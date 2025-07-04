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
    """单图像分类数据集 - 从序列数据中提取所有帧"""
    
    def __init__(self, csv_path, data_root, 
                 image_format="jpg", 
                 image_mode="rgb", normalize_images=True, 
                 custom_image_norm_stats=None,
                 frame_selection="all"):
        """
        初始化单图像数据集
        
        Args:
            csv_path: CSV文件路径
            data_root: 数据根目录
            image_format: 图像格式
            image_mode: 图像处理模式，"rgb" 或 "grayscale"
            normalize_images: 是否对图像进行标准化
            custom_image_norm_stats: 自定义图像标准化统计值
            frame_selection: 帧选择策略 ("all", "final", "keyframes")
        """
        self.csv_data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.image_format = image_format
        self.image_mode = image_mode.lower()
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        self.frame_selection = frame_selection
        
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
        print(f"  帧选择策略: {self.frame_selection}")
    
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
        """构建单图像样本列表"""
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
                
                # 根据帧选择策略提取样本
                if self.frame_selection == "all":
                    # 使用所有帧
                    selected_frames = list(range(len(frames)))
                elif self.frame_selection == "final":
                    # 只使用最后一帧
                    selected_frames = [len(frames) - 1]
                elif self.frame_selection == "keyframes":
                    # 使用关键帧（计数发生变化的帧）
                    selected_frames = self._find_keyframes(frames)
                else:
                    raise ValueError(f"Unknown frame_selection: {self.frame_selection}")
                
                # 为每个选中的帧创建样本
                for frame_idx in selected_frames:
                    if frame_idx < len(frames):
                        frame = frames[frame_idx]
                        
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
                            
                            # 获取标签
                            if self.frame_selection == "all":
                                # 使用当前帧的标签
                                label = frame.get('label', ball_count)
                            else:
                                # 使用序列的最终球数
                                label = ball_count
                            
                            # 创建样本
                            sample = {
                                'image_path': relative_image_path,
                                'label': int(label),
                                'sample_id': sample_id,
                                'frame_idx': frame_idx,
                                'original_ball_count': ball_count
                            }
                            samples.append(sample)
                
            except Exception as e:
                print(f"处理JSON文件失败 {json_path}: {e}")
                continue
        
        return samples
    
    def _find_keyframes(self, frames):
        """找到关键帧（计数发生变化的帧）"""
        keyframes = [0]  # 总是包含第一帧
        
        prev_count = frames[0].get('label', 0)
        for i, frame in enumerate(frames[1:], 1):
            current_count = frame.get('label', 0)
            if current_count != prev_count:
                keyframes.append(i)
                prev_count = current_count
        
        # 总是包含最后一帧
        if len(frames) - 1 not in keyframes:
            keyframes.append(len(frames) - 1)
        
        return keyframes
    
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
            'label': sample['label'],
            'sample_id': sample['sample_id'],
            'frame_idx': sample['frame_idx'],
            'original_ball_count': sample['original_ball_count']
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
                                  custom_image_norm_stats=None,
                                  frame_selection="all"):
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
        frame_selection: 帧选择策略 ("all", "final", "keyframes")
    
    Returns:
        train_loader, val_loader
    """
    
    print(f"=== 创建单图像数据加载器 - 图像模式: {image_mode.upper()} ===")
    print(f"帧选择策略: {frame_selection}")
    
    # 创建训练集
    train_dataset = SingleImageDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        frame_selection=frame_selection
    )
    
    # 创建验证集
    val_dataset = SingleImageDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        frame_selection=frame_selection
    )
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_dist = train_dataset.get_class_distribution()
    for label, count in sorted(train_dist.items()):
        print(f"  球数 {label}: {count} 样本")
    
    print("\n验证集类别分布:")
    val_dist = val_dataset.get_class_distribution()
    for label, count in sorted(val_dist.items()):
        print(f"  球数 {label}: {count} 样本")
    
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
        # 测试不同的帧选择策略
        strategies = ["all", "final", "keyframes"]
        
        for strategy in strategies:
            print(f"\n=== 测试帧选择策略: {strategy} ===")
            
            train_loader, val_loader = get_single_image_data_loaders(
                train_csv_path=train_csv,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                image_mode="rgb",
                normalize_images=True,
                frame_selection=strategy
            )
            
            print(f"训练集样本数: {len(train_loader.dataset)}")
            print(f"验证集样本数: {len(val_loader.dataset)}")
            
            # 测试batch数据
            for batch in train_loader:
                print(f"Batch shapes:")
                print(f"  Images: {batch['image'].shape}")
                print(f"  Labels: {batch['label'].shape}")
                print(f"  图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
                print(f"  标签范围: [{batch['label'].min()}, {batch['label'].max()}]")
                break
        
        print("\n=== 使用示例 ===")
        print("# 使用所有帧:")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root, frame_selection='all')")
        print()
        print("# 只使用最终帧:")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root, frame_selection='final')")
        print()
        print("# 使用关键帧:")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root, frame_selection='keyframes')")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()