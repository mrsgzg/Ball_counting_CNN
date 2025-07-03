import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import seaborn as sns

# 假设你的数据集加载器已经导入
from DataLoader_embodiment import BallCountingDataset, get_ball_counting_data_loaders

class DatasetVisualizer:
    """数据集可视化工具 - 服务器版本（保存到本地）"""
    
    def __init__(self, data_root, output_dir="visualization_output"):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"可视化结果将保存到: {self.output_dir}")
        
        # 设置matplotlib为非交互式后端
        plt.switch_backend('Agg')
    
    def load_original_image(self, image_path):
        """加载原始图像"""
        full_path = os.path.join(self.data_root, image_path)
        if os.path.exists(full_path):
            return Image.open(full_path).convert('RGB')
        return None
    
    def tensor_to_image(self, tensor):
        """将张量转换为可显示的图像"""
        if tensor.dim() == 3:
            # 如果是3D张量 (C, H, W)
            if tensor.shape[0] == 1:
                # 灰度图像
                return tensor.squeeze(0).numpy()
            elif tensor.shape[0] == 3:
                # RGB图像，需要转置为 (H, W, C)
                return tensor.permute(1, 2, 0).numpy()
        return tensor.numpy()
    
    def denormalize_image(self, tensor, mode='rgb'):
        """反归一化图像张量以便显示"""
        if mode == 'rgb':
            # ImageNet标准化参数
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            # 灰度标准化参数
            mean = torch.tensor([0.5]).view(1, 1, 1)
            std = torch.tensor([0.5]).view(1, 1, 1)
        
        # 反归一化
        denorm_tensor = tensor * std + mean
        # 限制在[0, 1]范围内
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        return denorm_tensor
    
    def visualize_joints(self, joints, ax, title="Joints Visualization"):
        """可视化关节点数据"""
        joints_np = joints.numpy() if isinstance(joints, torch.Tensor) else joints
        
        ax.clear()
        
        if len(joints_np) == 7:
            joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6', 'Joint7']
            
            bars = ax.bar(joint_names, joints_np, color='skyblue', alpha=0.7)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Joint Values')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, joints_np):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'Joints shape: {joints_np.shape}\nValues: {joints_np}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
    
    def visualize_sample(self, dataset, sample_idx=0, frame_idx=0, save_name=None):
        """可视化单个样本"""
        print(f"正在可视化样本 {sample_idx}, 帧 {frame_idx}...")
        
        sample = dataset[sample_idx]
        sequence_data = sample['sequence_data']
        
        print(f"样本ID: {sample['sample_id']}")
        print(f"标签: {sample['label']}")
        print(f"序列长度: {sequence_data['sequence_length']}")
        
        if frame_idx >= sequence_data['images'].shape[0]:
            frame_idx = 0
            print(f"帧索引超出范围，使用第0帧")
        
        image_tensor = sequence_data['images'][frame_idx]
        joints = sequence_data['joints'][frame_idx]
        timestamp = sequence_data['timestamps'][frame_idx]
        frame_label = sequence_data['labels'][frame_idx]
        
        print(f"图像张量形状: {image_tensor.shape}")
        print(f"图像值范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        # 确定图像模式
        if image_tensor.shape[0] == 3:
            image_mode = "RGB"
        elif image_tensor.shape[0] == 1:
            image_mode = "Grayscale"
        else:
            image_mode = "Unknown"
        
        # 创建可视化布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample {sample_idx} - Frame {frame_idx} - {image_mode} Mode', fontsize=16)
        
        # 1. 原始图像（从JSON获取）
        sample_row = dataset.csv_data.iloc[sample_idx]
        json_path = sample_row['json_path']
        
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            if frame_idx < len(json_data['frames']):
                frame_data = json_data['frames'][frame_idx]
                image_path = frame_data.get('image_path', '')
                
                if image_path:
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                    else:
                        relative_image_path = image_path
                    
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                    
                    original_image = self.load_original_image(relative_image_path)
                    
                    if original_image:
                        axes[0, 0].imshow(original_image)
                        axes[0, 0].set_title('Original Image')
                        axes[0, 0].axis('off')
                    else:
                        axes[0, 0].text(0.5, 0.5, 'Original Image\nNot Found', 
                                      transform=axes[0, 0].transAxes, ha='center', va='center')
                        axes[0, 0].set_title('Original Image (Not Found)')
                        axes[0, 0].axis('off')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Image Path', 
                                  transform=axes[0, 0].transAxes, ha='center', va='center')
                    axes[0, 0].set_title('Original Image (No Path)')
                    axes[0, 0].axis('off')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Error loading\noriginal image:\n{str(e)}', 
                          transform=axes[0, 0].transAxes, ha='center', va='center')
            axes[0, 0].set_title('Original Image (Error)')
            axes[0, 0].axis('off')
        
        # 2. 你的程序处理的图像
        if image_tensor.shape[0] == 3:
            # RGB模式
            rgb_denorm = self.denormalize_image(image_tensor, 'rgb')
            rgb_display = self.tensor_to_image(rgb_denorm)
            axes[0, 1].imshow(rgb_display)
            axes[0, 1].set_title('Your Program - RGB Image')
            axes[0, 1].axis('off')
        elif image_tensor.shape[0] == 1:
            # 灰度模式
            gray_denorm = self.denormalize_image(image_tensor, 'grayscale')
            gray_display = self.tensor_to_image(gray_denorm)
            axes[0, 1].imshow(gray_display, cmap='gray')
            axes[0, 1].set_title('Your Program - Grayscale Image')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, f'Unknown Format\nShape: {image_tensor.shape}', 
                          transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('Processed Image (Unknown)')
            axes[0, 1].axis('off')
        
        # 3. 图像统计信息
        axes[0, 2].text(0.1, 0.9, f'Image Mode: {image_mode}', fontsize=12, transform=axes[0, 2].transAxes, weight='bold')
        axes[0, 2].text(0.1, 0.8, f'Tensor Shape: {image_tensor.shape}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.7, f'Value Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.6, f'Mean: {image_tensor.mean():.3f}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.5, f'Std: {image_tensor.std():.3f}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.4, f'Sample ID: {sample["sample_id"]}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.3, f'Sample Label: {sample["label"]}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.2, f'Frame Label: {frame_label}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.1, f'Timestamp: {timestamp:.3f}', fontsize=11, transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Image Statistics')
        axes[0, 2].axis('off')
        
        # 4. 关节点可视化
        self.visualize_joints(joints, axes[1, 0], f'Joints (Frame {frame_idx})')
        
        # 5. 序列信息
        axes[1, 1].text(0.1, 0.9, f'Ball Count: {sequence_data["ball_count"]}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Sequence Length: {sequence_data["sequence_length"]}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Current Frame: {frame_idx}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Joints Shape: {joints.shape}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Joints Range: [{joints.min():.3f}, {joints.max():.3f}]', fontsize=12, transform=axes[1, 1].transAxes)
        
        # 显示关节点具体数值
        axes[1, 1].text(0.1, 0.3, 'Joint Values:', fontsize=11, transform=axes[1, 1].transAxes, weight='bold')
        for i, val in enumerate(joints.numpy()):
            y_pos = 0.25 - i * 0.03
            axes[1, 1].text(0.1, y_pos, f'  Joint{i+1}: {val:.4f}', fontsize=9, transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Sequence Information')
        axes[1, 1].axis('off')
        
        # 6. 整个序列的关节点变化
        all_joints = sequence_data['joints'].numpy()
        for i in range(7):
            axes[1, 2].plot(all_joints[:, i], label=f'Joint {i+1}', alpha=0.7, linewidth=2)
        axes[1, 2].axvline(x=frame_idx, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Current Frame')
        axes[1, 2].set_title('Joints Over Time')
        axes[1, 2].set_xlabel('Frame Index')
        axes[1, 2].set_ylabel('Joint Values')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = f"sample_{sample_idx}_frame_{frame_idx}_{image_mode.lower()}"
        
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存: {save_path}")
        return save_path
    
    def compare_rgb_vs_grayscale(self, rgb_dataset, gray_dataset, sample_idx=0, frame_idx=0):
        """比较RGB和灰度模式的处理结果"""
        print(f"比较RGB和灰度模式 - 样本 {sample_idx}, 帧 {frame_idx}")
        
        rgb_sample = rgb_dataset[sample_idx]
        gray_sample = gray_dataset[sample_idx]
        
        rgb_image = rgb_sample['sequence_data']['images'][frame_idx]
        gray_image = gray_sample['sequence_data']['images'][frame_idx]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'RGB vs Grayscale Comparison - Sample {sample_idx}, Frame {frame_idx}', fontsize=16)
        
        # RGB处理结果
        rgb_denorm = self.denormalize_image(rgb_image, 'rgb')
        rgb_display = self.tensor_to_image(rgb_denorm)
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title('Your Program - RGB Mode')
        axes[0, 0].axis('off')
        
        # 灰度处理结果
        gray_denorm = self.denormalize_image(gray_image, 'grayscale')
        gray_display = self.tensor_to_image(gray_denorm)
        axes[0, 1].imshow(gray_display, cmap='gray')
        axes[0, 1].set_title('Your Program - Grayscale Mode')
        axes[0, 1].axis('off')
        
        # 图像统计对比
        axes[0, 2].text(0.05, 0.9, 'RGB Statistics:', fontsize=12, transform=axes[0, 2].transAxes, weight='bold', color='red')
        axes[0, 2].text(0.05, 0.85, f'Shape: {rgb_image.shape}', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.8, f'Range: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.75, f'Mean: {rgb_image.mean():.3f}', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.7, f'Std: {rgb_image.std():.3f}', fontsize=10, transform=axes[0, 2].transAxes)
        
        axes[0, 2].text(0.05, 0.55, 'Grayscale Statistics:', fontsize=12, transform=axes[0, 2].transAxes, weight='bold', color='blue')
        axes[0, 2].text(0.05, 0.5, f'Shape: {gray_image.shape}', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.45, f'Range: [{gray_image.min():.3f}, {gray_image.max():.3f}]', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.4, f'Mean: {gray_image.mean():.3f}', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.05, 0.35, f'Std: {gray_image.std():.3f}', fontsize=10, transform=axes[0, 2].transAxes)
        
        axes[0, 2].set_title('Statistics Comparison')
        axes[0, 2].axis('off')
        
        # 像素值分布对比
        axes[1, 0].hist(rgb_display.flatten(), bins=50, alpha=0.7, color='red', label='RGB')
        axes[1, 0].set_title('RGB Pixel Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(gray_display.flatten(), bins=50, alpha=0.7, color='blue', label='Grayscale')
        axes[1, 1].set_title('Grayscale Pixel Distribution')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 关节点对比（应该相同）
        rgb_joints = rgb_sample['sequence_data']['joints'][frame_idx]
        gray_joints = gray_sample['sequence_data']['joints'][frame_idx]
        
        joint_names = [f'J{i+1}' for i in range(7)]
        x = np.arange(len(joint_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, rgb_joints.numpy(), width, label='RGB Mode', alpha=0.7, color='red')
        axes[1, 2].bar(x + width/2, gray_joints.numpy(), width, label='Grayscale Mode', alpha=0.7, color='blue')
        axes[1, 2].set_title('Joints Comparison (Should be Same)')
        axes[1, 2].set_xlabel('Joints')
        axes[1, 2].set_ylabel('Values')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(joint_names)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"rgb_vs_grayscale_sample_{sample_idx}_frame_{frame_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"RGB vs 灰度对比图已保存: {save_path}")
        return save_path
    
    def visualize_batch_comparison(self, rgb_loader, gray_loader, batch_idx=0, num_samples=4):
        """可视化批次数据对比"""
        print(f"正在可视化批次对比 {batch_idx}...")
        
        # 获取RGB批次数据
        for i, rgb_batch in enumerate(rgb_loader):
            if i == batch_idx:
                break
        else:
            print(f"RGB批次 {batch_idx} 不存在")
            return
        
        # 获取灰度批次数据
        for i, gray_batch in enumerate(gray_loader):
            if i == batch_idx:
                break
        else:
            print(f"灰度批次 {batch_idx} 不存在")
            return
        
        batch_size = len(rgb_batch['sample_id'])
        num_samples = min(num_samples, batch_size)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Batch {batch_idx} - RGB vs Grayscale Comparison', fontsize=16)
        
        for sample_idx in range(num_samples):
            # RGB图像 (第一帧)
            rgb_image = rgb_batch['sequence_data']['images'][sample_idx, 0]
            rgb_denorm = self.denormalize_image(rgb_image, 'rgb')
            rgb_display = self.tensor_to_image(rgb_denorm)
            
            axes[sample_idx, 0].imshow(rgb_display)
            axes[sample_idx, 0].set_title(f'RGB - Sample {rgb_batch["sample_id"][sample_idx]}')
            axes[sample_idx, 0].axis('off')
            
            # 灰度图像 (第一帧)
            gray_image = gray_batch['sequence_data']['images'][sample_idx, 0]
            gray_denorm = self.denormalize_image(gray_image, 'grayscale')
            gray_display = self.tensor_to_image(gray_denorm)
            
            axes[sample_idx, 1].imshow(gray_display, cmap='gray')
            axes[sample_idx, 1].set_title(f'Grayscale - Sample {gray_batch["sample_id"][sample_idx]}')
            axes[sample_idx, 1].axis('off')
            
            # 信息对比
            axes[sample_idx, 2].text(0.05, 0.9, 'RGB Mode:', fontsize=11, transform=axes[sample_idx, 2].transAxes, weight='bold', color='red')
            axes[sample_idx, 2].text(0.05, 0.85, f'Shape: {rgb_image.shape}', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            axes[sample_idx, 2].text(0.05, 0.8, f'Range: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            
            axes[sample_idx, 2].text(0.05, 0.65, 'Grayscale Mode:', fontsize=11, transform=axes[sample_idx, 2].transAxes, weight='bold', color='blue')
            axes[sample_idx, 2].text(0.05, 0.6, f'Shape: {gray_image.shape}', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            axes[sample_idx, 2].text(0.05, 0.55, f'Range: [{gray_image.min():.3f}, {gray_image.max():.3f}]', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            
            axes[sample_idx, 2].text(0.05, 0.4, f'Sample ID: {rgb_batch["sample_id"][sample_idx]}', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            axes[sample_idx, 2].text(0.05, 0.35, f'Label: {rgb_batch["label"][sample_idx]}', fontsize=9, transform=axes[sample_idx, 2].transAxes)
            
            axes[sample_idx, 2].set_title(f'Info - Sample {rgb_batch["sample_id"][sample_idx]}')
            axes[sample_idx, 2].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"batch_comparison_{batch_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"批次对比图已保存: {save_path}")
        return save_path


def test_dataset_visualization():
    """测试数据集可视化 - 服务器版本"""
    # 配置路径
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
    print("=== 数据集可视化测试 (服务器版本) ===")
    
    # 检查文件是否存在
    if not all(os.path.exists(path) for path in [train_csv, val_csv, data_root]):
        print("错误: 某些文件路径不存在，请检查配置")
        return
    
    # 创建可视化器
    visualizer = DatasetVisualizer(data_root, output_dir="dataset_visualization_results")
    
    try:
        # 创建RGB数据集
        print("创建RGB数据集...")
        rgb_dataset = BallCountingDataset(
            csv_path=train_csv,
            data_root=data_root,
            sequence_length=11,
            normalize=True,
            image_mode="rgb",
            normalize_images=True
        )
        
        # 创建灰度数据集
        print("创建灰度数据集...")
        gray_dataset = BallCountingDataset(
            csv_path=train_csv,
            data_root=data_root,
            sequence_length=11,
            normalize=True,
            image_mode="grayscale",
            normalize_images=True
        )
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        rgb_loader = DataLoader(rgb_dataset, batch_size=8, shuffle=False)
        gray_loader = DataLoader(gray_dataset, batch_size=8, shuffle=False)
        
        print(f"RGB数据集大小: {len(rgb_dataset)}")
        print(f"灰度数据集大小: {len(gray_dataset)}")
        
        # 测试样本可视化
        test_samples = [0, 1, 2]  # 测试多个样本
        test_frames = [0, 3, 6]   # 测试多个帧
        
        print("\n=== 测试RGB样本可视化 ===")
        for sample_idx in test_samples:
            for frame_idx in test_frames:
                try:
                    visualizer.visualize_sample(
                        rgb_dataset, 
                        sample_idx=sample_idx, 
                        frame_idx=frame_idx,
                        save_name=f"rgb_sample_{sample_idx}_frame_{frame_idx}"
                    )
                except Exception as e:
                    print(f"RGB样本 {sample_idx} 帧 {frame_idx} 可视化失败: {e}")
        
        print("\n=== 测试灰度样本可视化 ===")
        for sample_idx in test_samples:
            for frame_idx in test_frames:
                try:
                    visualizer.visualize_sample(
                        gray_dataset, 
                        sample_idx=sample_idx, 
                        frame_idx=frame_idx,
                        save_name=f"grayscale_sample_{sample_idx}_frame_{frame_idx}"
                    )
                except Exception as e:
                    print(f"灰度样本 {sample_idx} 帧 {frame_idx} 可视化失败: {e}")
        
        print("\n=== 测试RGB vs 灰度对比 ===")
        for sample_idx in [0, 1]:
            for frame_idx in [0, 3]:
                try:
                    visualizer.compare_rgb_vs_grayscale(
                        rgb_dataset, gray_dataset, 
                        sample_idx=sample_idx, 
                        frame_idx=frame_idx
                    )
                except Exception as e:
                    print(f"RGB vs 灰度对比 样本 {sample_idx} 帧 {frame_idx} 失败: {e}")
        
        print("\n=== 测试批次对比 ===")
        try:
            visualizer.visualize_batch_comparison(
                rgb_loader, gray_loader, 
                batch_idx=0, 
                num_samples=4
            )
        except Exception as e:
            print(f"批次对比可视化失败: {e}")
        
        print("\n=== 可视化测试完成 ===")
        print(f"所有结果已保存到: {visualizer.output_dir}")
        
        # 生成总结报告
        generate_summary_report(visualizer.output_dir)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def generate_summary_report(output_dir):
    """生成可视化结果总结报告"""
    print("\n=== 生成总结报告 ===")
    
    # 统计生成的文件
    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    report_content = f"""
# 数据集可视化结果总结报告

## 测试时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 生成文件统计
总共生成图像文件: {len(files)} 个

### 文件列表:
"""
    
    # 按类型分组
    rgb_samples = [f for f in files if f.startswith('rgb_sample_')]
    gray_samples = [f for f in files if f.startswith('grayscale_sample_')]
    comparisons = [f for f in files if f.startswith('rgb_vs_grayscale_')]
    batch_files = [f for f in files if f.startswith('batch_comparison_')]
    
    report_content += f"""
#### RGB样本可视化 ({len(rgb_samples)} 个):
"""
    for f in sorted(rgb_samples):
        report_content += f"- {f}\n"
    
    report_content += f"""
#### 灰度样本可视化 ({len(gray_samples)} 个):
"""
    for f in sorted(gray_samples):
        report_content += f"- {f}\n"
    
    report_content += f"""
#### RGB vs 灰度对比 ({len(comparisons)} 个):
"""
    for f in sorted(comparisons):
        report_content += f"- {f}\n"
    
    report_content += f"""
#### 批次对比 ({len(batch_files)} 个):
"""
    for f in sorted(batch_files):
        report_content += f"- {f}\n"
    
    report_content += """

## 可视化内容说明

### 单个样本可视化包含:
1. **原始图像**: 从JSON路径加载的原始图像
2. **你的程序处理结果**: 
   - RGB模式: 3通道 (3, 224, 224) 经过ImageNet标准化
   - 灰度模式: 1通道 (1, 224, 224) 经过灰度标准化
3. **图像统计信息**: 张量形状、数值范围、均值、标准差等
4. **关节点数据**: 7个关节点的数值柱状图
5. **序列信息**: 球数、序列长度、当前帧等
6. **时间序列**: 整个序列中关节点的变化趋势

### RGB vs 灰度对比包含:
1. **并排图像对比**: RGB处理结果 vs 灰度处理结果
2. **统计信息对比**: 两种模式的张量统计对比
3. **像素分布对比**: RGB和灰度的像素值分布直方图
4. **关节点对比**: 验证两种模式下关节点数据是否一致

### 批次对比包含:
1. **多样本并排**: 同一批次中RGB和灰度模式的并排对比
2. **详细信息**: 每个样本的张量形状和数值范围对比

## 注意事项
- 所有图像都经过了反归一化处理以便正确显示
- RGB使用ImageNet标准化参数: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- 灰度使用标准参数: mean=[0.5], std=[0.5]
- 关节点和时间戳数据根据设置进行了归一化处理
"""
    
    # 保存报告
    report_path = os.path.join(output_dir, "visualization_summary_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"总结报告已保存: {report_path}")
    print(f"请查看 {output_dir} 目录下的所有可视化结果")


# 快速测试函数
def quick_test(sample_indices=[0, 1], frame_indices=[0, 3]):
    """快速测试指定样本和帧"""
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    
    print("=== 快速测试 ===")
    
    if not all(os.path.exists(path) for path in [train_csv, data_root]):
        print("错误: 文件路径不存在")
        return
    
    visualizer = DatasetVisualizer(data_root, output_dir="quick_test_results")
    
    # 只创建RGB数据集进行快速测试
    rgb_dataset = BallCountingDataset(
        csv_path=train_csv,
        data_root=data_root,
        sequence_length=11,
        normalize=True,
        image_mode="rgb",
        normalize_images=True
    )
    
    gray_dataset = BallCountingDataset(
        csv_path=train_csv,
        data_root=data_root,
        sequence_length=11,
        normalize=True,
        image_mode="grayscale",
        normalize_images=True
    )
    
    for sample_idx in sample_indices:
        for frame_idx in frame_indices:
            print(f"测试样本 {sample_idx}, 帧 {frame_idx}")
            try:
                # RGB可视化
                visualizer.visualize_sample(rgb_dataset, sample_idx, frame_idx, 
                                          f"quick_rgb_s{sample_idx}_f{frame_idx}")
                
                # 灰度可视化
                visualizer.visualize_sample(gray_dataset, sample_idx, frame_idx, 
                                          f"quick_gray_s{sample_idx}_f{frame_idx}")
                
                # 对比
                visualizer.compare_rgb_vs_grayscale(rgb_dataset, gray_dataset, sample_idx, frame_idx)
                
            except Exception as e:
                print(f"测试失败: {e}")
    
    print(f"快速测试完成，结果保存在: {visualizer.output_dir}")


if __name__ == "__main__":
    # 运行完整测试
    test_dataset_visualization()
    
    # 或者运行快速测试
    # quick_test(sample_indices=[0], frame_indices=[0])