import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import random

class SingleImageDataset(Dataset):
    """单图像分类数据集 - 从序列数据中提取所有帧，标签使用ball_count"""
    
    def __init__(self, csv_path, data_root, 
                 image_format="jpg", 
                 image_mode="rgb", normalize_images=True, 
                 custom_image_norm_stats=None):
        """
        初始化单图像数据集
        
        Args:
            csv_path: CSV文件路径
            data_root: 数据根目录
            image_format: 图像格式
            image_mode: 图像处理模式，"rgb" 或 "grayscale"
            normalize_images: 是否对图像进行标准化
            custom_image_norm_stats: 自定义图像标准化统计值
        """
        self.csv_data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.image_format = image_format
        self.image_mode = image_mode.lower()
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # 验证图像模式
        assert self.image_mode in ["rgb", "grayscale"], "image_mode必须是'rgb'或'grayscale'"
        
        # 验证CSV格式
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in self.csv_data.columns, f"CSV必须包含{col}列"
        
        # 设置图像变换
        self._setup_image_transforms()
        
        # 构建单图像样本列表
        self.samples = self._build_sample_list()
        
        print(f"单图像数据集构建完成:")
        print(f"  原始序列数: {len(self.csv_data)}")
        print(f"  提取的单图像样本数: {len(self.samples)}")
        print(f"  图像模式: {self.image_mode}")
        print(f"  标签: 直接使用ball_count")
    
    def _setup_image_transforms(self):
        """设置图像变换流水线"""
        transform_list = []
        
        # 基础变换
        transform_list.append(transforms.Resize((224, 224)))
        
        # 根据模式处理颜色
        if self.image_mode == "grayscale":
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 图像标准化
        if self.normalize_images:
            if self.custom_image_norm_stats:
                mean = self.custom_image_norm_stats["mean"]
                std = self.custom_image_norm_stats["std"]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            elif self.image_mode == "rgb":
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
            else:
                transform_list.append(transforms.Normalize(
                    mean=[0.5], 
                    std=[0.5]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
    
    def _build_sample_list(self):
        """构建单图像样本列表 - 提取所有帧，标签使用ball_count"""
        samples = []
        
        for idx in range(len(self.csv_data)):
            sample_row = self.csv_data.iloc[idx]
            sample_id = sample_row['sample_id']
            ball_count = sample_row['ball_count']
            json_path = sample_row['json_path']
            

            
            try:
                # 加载JSON数据
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                frames = json_data['frames']
                
                # 提取所有帧
                for frame_idx, frame in enumerate(frames):
                    # 获取图像路径
                    image_path = frame.get('image_path', '')
                    if image_path:
                        # 处理图像路径
                        path_parts = image_path.split('/')
                        if 'ball_data_collection' in path_parts:
                            ball_data_idx = path_parts.index('ball_data_collection')
                            relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                        else:
                            relative_image_path = image_path
                        
                        # 修复路径命名不一致问题
                        if '1_ball' in relative_image_path:
                            relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                        
                        # 创建样本 - 标签直接使用ball_count
                        sample = {
                            'image_path': relative_image_path,
                            'label': int(ball_count),  # 直接使用ball_count作为标签
                            'sample_id': sample_id,
                            'frame_idx': frame_idx
                        }
                        samples.append(sample)
                
            except Exception as e:
                print(f"处理JSON文件失败 {json_path}: {e}")
                continue
        
        return samples
    
    def _load_image(self, image_path):
        """加载并处理单张图像"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found: {full_image_path}")
                channels = 3 if self.image_mode == "rgb" else 1
                return torch.zeros(channels, 224, 224)
            
            # 加载图像
            image = Image.open(full_image_path).convert('RGB')
            
            # 应用变换
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            channels = 3 if self.image_mode == "rgb" else 1
            return torch.zeros(channels, 224, 224)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'label': sample['label'],  # 1-10的球数
            'sample_id': sample['sample_id'],
            'frame_idx': sample['frame_idx']
        }
    
    def get_class_distribution(self):
        """获取类别分布"""
        labels = [sample['label'] for sample in self.samples]
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def debug_sample_loading(self, sample_idx=0):
        """调试样本加载"""
        print(f"调试样本 {sample_idx} 的加载...")
        
        if sample_idx >= len(self.samples):
            print(f"样本索引 {sample_idx} 超出范围 (最大: {len(self.samples)-1})")
            return
        
        sample_info = self.samples[sample_idx]
        print(f"样本信息: {sample_info}")
        
        try:
            sample = self[sample_idx]
            print(f"成功加载样本:")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  标签: {sample['label']}")
            print(f"  图像值范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        except Exception as e:
            print(f"加载样本失败: {e}")
            import traceback
            traceback.print_exc()


def get_single_image_data_loaders(train_csv_path, val_csv_path, data_root, 
                                  batch_size=32, 
                                  num_workers=4, 
                                  image_mode="rgb", normalize_images=True,
                                  custom_image_norm_stats=None):
    """
    创建单图像分类的训练和验证数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
        val_csv_path: 验证集CSV路径
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载器进程数
        image_mode: 图像处理模式，"rgb" 或 "grayscale"
        normalize_images: 是否对图像进行标准化
        custom_image_norm_stats: 自定义图像标准化参数
    
    Returns:
        train_loader, val_loader
    """
    
    print(f"=== 创建单图像数据加载器 - 图像模式: {image_mode.upper()} ===")
    print("标签: 直接使用ball_count")
    
    # 创建训练集
    train_dataset = SingleImageDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # 创建验证集
    val_dataset = SingleImageDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_dist = train_dataset.get_class_distribution()
    for label in sorted(train_dist.keys()):
        print(f"  球数 {label}: {train_dist[label]} 样本")
    
    print("\n验证集类别分布:")
    val_dist = val_dataset.get_class_distribution()
    for label in sorted(val_dist.keys()):
        print(f"  球数 {label}: {val_dist[label]} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 数据集路径配置
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== 单图像分类数据集测试 ===")
    
    # 检查文件是否存在
    if not os.path.exists(train_csv):
        print(f"错误: 训练集CSV文件不存在: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"错误: 验证集CSV文件不存在: {val_csv}")
        exit(1)
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        exit(1)
    
    try:
        # 测试RGB模式
        print("\n=== 测试RGB模式 ===")
        train_loader_rgb, val_loader_rgb = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            image_mode="rgb",
            normalize_images=True
        )
        
        print(f"训练集样本数: {len(train_loader_rgb.dataset)}")
        print(f"验证集样本数: {len(val_loader_rgb.dataset)}")
        
        # 测试batch数据
        for batch in train_loader_rgb:
            print(f"RGB Batch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Labels: {batch['label'].shape}")
            print(f"  图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"  标签范围: [{batch['label'].min()}, {batch['label'].max()}]")
            print(f"  样本标签示例: {batch['label'][:5].tolist()}")
            break
        
        # 测试灰度模式
        print("\n=== 测试灰度模式 ===")
        train_loader_gray, val_loader_gray = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            image_mode="grayscale",
            normalize_images=True
        )
        
        # 测试batch数据
        for batch in train_loader_gray:
            print(f"灰度 Batch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Labels: {batch['label'].shape}")
            print(f"  图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"  标签范围: [{batch['label'].min()}, {batch['label'].max()}]")
            print(f"  样本标签示例: {batch['label'][:5].tolist()}")
            break
        
        print("\n=== 使用示例 ===")
        print("# RGB模式:")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root, image_mode='rgb')")
        print()
        print("# 灰度模式:")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root, image_mode='grayscale')")
        
        print("\n=== 测试完成 ===")
        print("注意: 所有标签都直接使用ball_count值！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()