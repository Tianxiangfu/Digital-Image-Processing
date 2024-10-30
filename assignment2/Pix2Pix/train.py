import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

def tensor_to_image(tensor):
    """
    Convert a tensor to a NumPy image array.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The output image as a NumPy array (H, W, C).
    """
    # 将 tensor 从 (C, H, W) 转换为 (H, W, C)
    img = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    # 如果是单通道灰度图像，扩展为 3 通道
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)  # 将 1 通道的灰度图扩展为 3 通道
    
    # 将像素值从 [-1, 1] 转换为 [0, 255]（根据需要调整）
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    
    return img

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images using the updated tensor_to_image function
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # 确保输入、目标和输出图像都有 3 通道
        if input_img_np.shape[-1] == 1:
            input_img_np = np.repeat(input_img_np, 3, axis=-1)
        if target_img_np.shape[-1] == 1:
            target_img_np = np.repeat(target_img_np, 3, axis=-1)
        if output_img_np.shape[-1] == 1:
            output_img_np = np.repeat(output_img_np, 3, axis=-1)

        # Concatenate the images horizontally (input -> target -> output)
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_rgb)

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

        # Compute the loss
        loss = criterion(outputs, image_semantic)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')


class FacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [file for file in os.listdir(root_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path)

        # 获取图片的宽度和高度
        width, height = image.size

        # 将图片从中间分成左右两部分
        input_image = image.crop((0, 0, width // 2, height))  # 左半部分为输入
        target_image = image.crop((width // 2, 0, width, height))  # 右半部分为目标

        if self.transform:
            input_image = self.transform(input_image.convert("L"))  # 转换为灰度图像
            target_image = self.transform(target_image.convert("RGB"))  # 保持彩色图像

        return input_image, target_image

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将所有图像调整为 256x256 的尺寸
        transforms.ToTensor(),          # 转换为张量
    ])

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(root_dir='datasets/facades/test', transform=transform)
    val_dataset = FacadesDataset(root_dir='datasets/facades/val', transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        validate(model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
