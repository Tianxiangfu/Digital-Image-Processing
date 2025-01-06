# 实验报告：简化的3D高斯点云渲染

本次实验实现了简化版的3D高斯点云渲染，旨在通过结构光法（SfM）和3D高斯点云表示从多视角图像中重建3D场景。实验过程包含了数据处理、3D高斯初始化、图像投影、Gaussian值计算以及体积渲染等多个步骤。

## 目录
- [实验报告：简化的3D高斯点云渲染](#实验报告简化的3d高斯点云渲染)
  - [目录](#目录)
  - [实验概述](#实验概述)
  - [安装步骤](#安装步骤)
  - [使用方法](#使用方法)
  - [实验结果](#实验结果)


## 实验概述

本实验的目标是实现从多视角图像中重建3D场景，并对3D点进行高斯点云处理以进行体积渲染。实验主要分为以下几个步骤：

1. **结构光法（SfM）**：使用Colmap进行相机姿态估计与稀疏3D点云重建。
2. **简化版3D高斯点云渲染**：将3D点云转化为3D高斯点进行渲染，通过调整协方差矩阵、颜色和不透明度来改善渲染效果。
3. **高斯投影与值计算**：实现3D高斯到2D高斯的投影，并计算每个像素位置的高斯值。
4. **体积渲染与α混合**：基于已计算的高斯值进行α混合，最终得到渲染图像。

## 安装步骤

在本地运行此实验，请按照以下步骤操作：

1. **克隆仓库并下载数据**：

    ```bash
    git clone <仓库地址>
    cd <项目文件夹>
    ```

2. **运行程序**：
  使用Colmap进行相机姿态估计与3D点重建：
   ```
   python mvs_with_colmap.py --data_dir data/chair
   python debug_mvs_by_projecting_pts.py --data_dir data/chair
   ```
   得到结果如下：
   <p align="center">
     <figure style="display: inline-block; margin: 10px;">
       <img src="pics\1.png" alt="语义图" width="900" height="280">
       <figcaption>图1：高斯点云的初步渲染。</figcaption>
     </figure>
   
   
   初始化3D高斯点云并进行渲染：
   **填充代码部分**：
    1.计算并返回每个高斯分布的协方差矩阵，协方差矩阵由旋转矩阵 R 和缩放矩阵 S 共同决定。旋转矩阵控制高斯分布的方向，而尺度矩阵控制其在各个方向上的扩展程度。
    <p align="center">
     <figure style="display: inline-block; margin: 10px;">
       <img src="pics\2.png" alt="语义图" width="900" height="280">
       <figcaption>图2：计算每个高斯分布的协方差矩阵。</figcaption>
     </figure>
    
    2.计算投影的雅可比矩阵并将三维协方差变换到相机坐标系
    <p align="center">
     <figure style="display: inline-block; margin: 10px;">
       <img src="pics\3.png" alt="语义图" width="900" height="300">
       <figcaption>图3：计算投影的雅可比矩阵并将三维协方差变换到相机坐标系。</figcaption>
     </figure>
    
    3.实现了基于 alpha 合成（alpha compositing）进行图像渲染的过程
    <p align="center">
     <figure style="display: inline-block; margin: 10px;">
       <img src="pics\4.png" alt="语义图" width="900" height="240">
       <figcaption>图4：累积透明度（Alpha）并计算权重。</figcaption>
     </figure>

  开始训练模型，运行：
   ```
   python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
   ```

## 使用方法

**3D高斯点云渲染**：
运行程序后，系统会从重建结果中读取相机位置与3D点信息，并通过初始化3D高斯点云进行渲染。你需要提供多视角图像，并选择对应的相机参数进行处理。
**体积渲染**：
通过调整不透明度和颜色等属性，程序会使用高斯点云进行体积渲染，从而生成3D场景的投影。

## 实验结果

结构光法与3D点重建：
在运行结构光法后，我们得到了稀疏的3D点云，结合相机的姿态信息进行场景重建。
3D高斯点云初始化与投影：
初始3D点云经过高斯变换后，生成了覆盖更大空间的3D高斯点。在投影到2D平面时，成功生成了具有适当模糊效果的高斯分布。 例如，我们以乐高玩具图片训练，以下是渲染的效果图：

<img src="pics\result.gif" alt="对人的交互操作" width="600">

