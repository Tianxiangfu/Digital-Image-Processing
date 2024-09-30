import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    height, width = transformed_image.shape[:2]
    center_x, center_y = width // 2, height // 2


    translation_x_matrix = np.array([[1, 0, 0],
                               [0, 1, -translation_x],
                               [0, 0, 1]])
    translation_y_matrix = np.array([[1, 0, translation_y],
                               [0, 1, 0],
                               [0, 0, 1]])
    translation_center_matrix = np.array([[1, 0, -center_y],
                               [0, 1, -center_x],
                               [0, 0, 1]])
    translation_center_inverse_matrix = np.array([[1, 0, center_y],
                               [0, 1, center_x],
                               [0, 0, 1]])
    scale_matrix = np.array([[1 / scale, 0, 0],
                               [0, 1 / scale, 0],
                               [0, 0, 1]])
    flip_matrix = np.array([[1, 0, 0],
                               [0, -1, width],
                               [0, 0, 1]])
    theta = np.radians(rotation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta),  np.cos(theta), 0],
                            [0,              0,             1]])
    transformed_image = matrix_transform(transformed_image, [translation_x_matrix])

    transformed_image = matrix_transform(transformed_image, [translation_y_matrix])
    #缩放
    transformed_image = matrix_transform(transformed_image, [translation_center_matrix, scale_matrix, translation_center_inverse_matrix])
    #旋转
    transformed_image = matrix_transform(transformed_image, [translation_center_matrix, rotation_matrix, translation_center_inverse_matrix])
    if flip_horizontal:
        transformed_image = matrix_transform(transformed_image, [flip_matrix])
    return transformed_image

import numpy as np

def matrix_transform(image, transform_matrices):
    h, w = image.shape[:2]
    
    # 获取图像中每个像素点的坐标 (x, y, 1)
    coordinates = np.indices((h, w)).reshape(2, -1)
    ones = np.ones((1, coordinates.shape[1]))  # 添加 (x, y, 1) 中的 1
    homogenous_coords = np.vstack([coordinates, ones])  # (3, h*w)

    # 将所有变换矩阵进行连乘
    total_transform_matrix = np.eye(3)  # 初始化为单位矩阵
    for transform_matrix in transform_matrices:
        total_transform_matrix = transform_matrix @ total_transform_matrix

    # 应用总变换矩阵
    transformed_coords = total_transform_matrix @ homogenous_coords

    # 去齐次化 (x', y', z') -> (x'/z', y'/z')
    transformed_coords = transformed_coords[:2, :] / transformed_coords[2, :]

    # 将结果映射回图像坐标
    transformed_coords = np.round(transformed_coords).astype(int)

    # 创建空白图像，用于存储变换后的图像
    transformed_image = np.zeros_like(image)

    # 映射变换后的坐标到新图像
    valid_idx = (transformed_coords[0] >= 0) & (transformed_coords[0] < h) & (transformed_coords[1] >= 0) & (transformed_coords[1] < w)
    transformed_image[coordinates[0, valid_idx], coordinates[1, valid_idx]] = image[transformed_coords[0, valid_idx], transformed_coords[1, valid_idx]]

    return transformed_image



# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch('share=True')