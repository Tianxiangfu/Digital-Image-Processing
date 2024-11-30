import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from facades_dataset import FacadesDataset
import torch.nn.functional as F
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.cuda.amp import autocast
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    
    batch_size = inputs.size(0)
    num_images = min(num_images, batch_size)  # 确保不会访问超出范围的索引

    for i in range(num_images):
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




def train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epoch, num_epochs, save_interval=5):
    generator.train()
    discriminator.train()
    
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        
        batch_size = real_images.size(0)
        
        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------------
        # 训练判别器
        # ---------------------------
        discriminator.zero_grad()

        # 判别器训练真实图像
        output_real = discriminator(real_images, real_images)
        loss_real = criterion(output_real, real_labels)

        # 生成假图像
        z = torch.randn(batch_size, 100).to(device)  # 随机噪声
        fake_images = generator(real_images, z)
        output_fake = discriminator(real_images, fake_images.detach())  # detach()避免更新生成器
        loss_fake = criterion(output_fake, fake_labels)

        # 总判别器损失
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # ---------------------------
        # 训练生成器
        # ---------------------------
        generator.zero_grad()

        output_fake = discriminator(real_images, fake_images)
        loss_g = criterion(output_fake, real_labels)  # 让判别器认为生成的图像为真实
        loss_g.backward()
        optimizer_g.step()

        # 打印损失
        if i % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

        # 每隔一定的步骤保存一些图像
        if i == 0 and epoch % save_interval == 0:
            # 这里的 `inputs` 就是训练中的真实图像，`targets` 通常为目标图像（例如，地面真值图像）
            save_images(real_images, real_images, fake_images, 'train_results', epoch)


def validate(generator, discriminator, val_loader, device, epoch, save_interval=5):
    generator.eval()  # 设置生成器为评估模式
    val_loss = 0
    with torch.no_grad():
        for step, (real_images, _) in enumerate(val_loader):
            real_images = real_images.to(device)
            z = torch.randn(real_images.size(0), 100).to(device)
            fake_images = generator(real_images, z)
            
            # 计算判别器的输出和损失
            output_fake = discriminator(real_images, fake_images)
            loss = nn.BCELoss()(output_fake, torch.ones_like(output_fake).to(device))  # 假设你的损失是这样计算的
            val_loss += loss.item()
            
            # 每隔一定的步数保存图像
            if epoch % save_interval == 0 and step == 0:
                save_images(real_images, real_images, fake_images, 'val_results', epoch)

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


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
        input_image = image.crop((0, 0, width // 2, height))  # 左半部分为目标
        target_image = image.crop((width // 2, 0, width, height))  # 右半部分为输入

        if self.transform:
            input_image = self.transform(input_image.convert("L"))  # 转换为灰度图像
            target_image = self.transform(target_image.convert("RGB"))  # 保持彩色图像

        return  target_image, input_image



class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # 输入：z_dim + 3*256*256
        self.fc1 = nn.Linear(z_dim + 3 * 256 * 256, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 3 * 256 * 256)
        
    def forward(self, x, z):
        with autocast():
            # 将条件图像x和噪声z拼接起来
            x = x.view(x.size(0), -1)  # 展平条件图像
            z = z.view(z.size(0), -1)  # 展平噪声z
            combined_input = torch.cat([x, z], dim=1)  # 拼接z和x
        
            x = F.relu(self.fc1(combined_input))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))  # 输出值[-1, 1]范围内
        
        return x.view(-1, 3, 256, 256)  # 将输出变为3*256*256的图像



    



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 使用卷积层来处理图像
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)  # 6个输入通道 (x + y)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)  # 将卷积的输出展平
        self.fc2 = nn.Linear(1024, 1)  # 输出一个概率
        
    def forward(self, x, y):
        # 将x和y拼接（即条件图像和待判别图像拼接）
        x = torch.cat([x, y], dim=1)  # 拼接
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        
        x = x.view(x.size(0), -1)  # 展平
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))  # 输出0到1之间的概率
        return x



def main():
    """
    Main function to set up the training and validation processes.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # 将灰度图转换为3通道
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    print(f"Using device: {device}")

    # 初始化模型
    z_dim = 100  # 噪声向量的维度
    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二进制交叉熵损失
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 学习率调度器：根据验证集损失动态调整学习率
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    # 数据加载器
    train_dataset = FacadesDataset(root_dir='cGAN/datasets/facades/train', transform=transform)
    val_dataset = FacadesDataset(root_dir='cGAN/datasets/facades/train', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    num_epochs = 300
    for epoch in range(num_epochs):
        # 训练一轮
        train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epoch, num_epochs, save_interval=5)
        
        # 验证后根据验证损失调整学习率
        val_loss = validate(generator, discriminator, val_loader, device, epoch)  # 你可以在validate函数中计算并返回验证损失
        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)

        # 可选：保存模型
        #if epoch % 10 == 0:
            #torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
            #torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()