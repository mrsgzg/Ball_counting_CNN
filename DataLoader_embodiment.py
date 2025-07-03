import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from collections import defaultdict
import random

class DataNormalizer:
    """高效的数据归一化器，支持保存和加载统计信息"""
    
    def __init__(self):
        self.stats = {
            'joints': {'mean': None, 'std': None},
            'timestamps': {'mean': None, 'std': None}
        }
        self.is_fitted = False
    
    def compute_and_save_stats(self, csv_path, data_root, save_path, sample_size=1000):
        """计算并保存归一化统计信息"""
        print("计算归一化统计信息...")
        
        # 读取CSV并随机采样
        csv_data = pd.read_csv(csv_path)
        sample_indices = np.random.choice(len(csv_data), 
                                         min(sample_size, len(csv_data)), 
                                         replace=False)
        
        joints_all = []
        timestamps_all = []
        
        for idx in sample_indices:
            sample_row = csv_data.iloc[idx]
            json_path = sample_row['json_path']
            full_json_path = json_path
            
            try:
                with open(full_json_path, 'r') as f:
                    json_data = json.load(f)
                
                # 提取所有帧的joints和timestamps
                for frame in json_data['frames']:
                    # 安全地提取joints数据
                    joints = frame.get('joints', [])
                    if joints and all(isinstance(x, (int, float)) and x is not None for x in joints):
                        joints_all.append(joints)
                    
                    # 安全地提取timestamp数据
                    timestamp = frame.get('timestamp', None)
                    if timestamp is not None and isinstance(timestamp, (int, float)):
                        timestamps_all.append(timestamp)
                    
            except Exception as e:
                print(f"处理文件时出错 {json_path}: {e}")
                continue
        
        # 计算统计信息
        if joints_all:
            joints_array = np.array(joints_all)
            self.stats['joints']['mean'] = np.mean(joints_array, axis=0)
            self.stats['joints']['std'] = np.std(joints_array, axis=0) + 1e-8
        
        if timestamps_all:
            timestamps_array = np.array(timestamps_all)
            self.stats['timestamps']['mean'] = np.mean(timestamps_array)
            self.stats['timestamps']['std'] = np.std(timestamps_array) + 1e-8
        
        self.is_fitted = True
        self.save_stats(save_path)
        print(f"归一化统计信息已保存到: {save_path}")
    
    def normalize(self, data, modality):
        """标准化归一化数据"""
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，请先计算统计信息")
        
        is_tensor = isinstance(data, torch.Tensor)
        if not is_tensor:
            data = torch.tensor(data, dtype=torch.float32)
        
        mean = torch.tensor(self.stats[modality]['mean'], dtype=torch.float32)
        std = torch.tensor(self.stats[modality]['std'], dtype=torch.float32)
        
        # 处理维度匹配
        if len(data.shape) > len(mean.shape):
            # 如果data是多维的，需要适当扩展mean和std的维度
            for _ in range(len(data.shape) - len(mean.shape)):
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
        
        # 标准化
        normalized = (data - mean) / std
        
        if not is_tensor:
            normalized = normalized.numpy()
        
        return normalized
    
    def denormalize(self, data, modality):
        """反归一化数据"""
        if not self.is_fitted:
            raise ValueError("归一化器未拟合")
        
        is_tensor = isinstance(data, torch.Tensor)
        if not is_tensor:
            data = torch.tensor(data, dtype=torch.float32)
        
        mean = torch.tensor(self.stats[modality]['mean'], dtype=torch.float32)
        std = torch.tensor(self.stats[modality]['std'], dtype=torch.float32)
        
        # 处理维度匹配
        if len(data.shape) > len(mean.shape):
            for _ in range(len(data.shape) - len(mean.shape)):
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
        
        denormalized = data * std + mean
        
        if not is_tensor:
            denormalized = denormalized.numpy()
        
        return denormalized
    
    def save_stats(self, filepath):
        """保存统计信息到JSON文件"""
        stats_to_save = {}
        for modality, stats in self.stats.items():
            stats_to_save[modality] = {}
            for key, value in stats.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        stats_to_save[modality][key] = value.tolist()
                    else:
                        stats_to_save[modality][key] = value
                else:
                    stats_to_save[modality][key] = None
        
        with open(filepath, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def load_stats(self, filepath):
        """从JSON文件加载统计信息"""
        with open(filepath, 'r') as f:
            stats_data = json.load(f)
        
        for modality, stats in stats_data.items():
            self.stats[modality] = {}
            for key, value in stats.items():
                if value is not None and isinstance(value, list):
                    self.stats[modality][key] = np.array(value)
                else:
                    self.stats[modality][key] = value
        
        self.is_fitted = True
        print(f"归一化统计信息已从 {filepath} 加载")


class BallCountingDataset(Dataset):
    """球类计数多模态序列数据集"""
    
    def __init__(self, csv_path, data_root, sequence_length=6, 
                 normalize=True, norm_stats_path=None, 
                 image_format="jpg", 
                 image_mode="rgb", normalize_images=True, 
                 custom_image_norm_stats=None):
        """
        初始化数据集
        
        Args:
            csv_path: CSV文件路径
            data_root: 数据根目录
            sequence_length: 序列长度，默认6（根据你的数据）
            normalize: 是否归一化关节和时间戳数据
            norm_stats_path: 归一化统计信息文件路径
            image_format: 图像格式
            image_mode: 图像处理模式，"rgb" 或 "grayscale"
            normalize_images: 是否对图像进行标准化
            custom_image_norm_stats: 自定义图像标准化统计值 {"mean": [...], "std": [...]}
        """
        self.csv_data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.image_format = image_format
        self.normalize = normalize
        self.image_mode = image_mode.lower()
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # 验证图像模式
        assert self.image_mode in ["rgb", "grayscale"], "image_mode必须是'rgb'或'grayscale'"
        
        # 验证CSV格式
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in self.csv_data.columns, f"CSV必须包含{col}列"
        
        # 初始化归一化器（用于关节和时间戳）
        self.normalizer = None
        if self.normalize:
            self.normalizer = DataNormalizer()
            
            # 构建统计信息文件路径
            if norm_stats_path is None:
                norm_stats_path = os.path.join(data_root, 'ball_normalization_stats.json')
            
            # 检查是否存在统计信息文件
            if os.path.exists(norm_stats_path):
                self.normalizer.load_stats(norm_stats_path)
            else:
                # 计算并保存统计信息
                self.normalizer.compute_and_save_stats(csv_path, data_root, norm_stats_path)
        
        # 设置图像变换
        self._setup_image_transforms()
    
    def _setup_image_transforms(self):
        """设置图像变换流水线"""
        transform_list = []
        
        # 基础变换
        transform_list.append(transforms.Resize((224, 224)))
        
        # 根据模式处理颜色
        if self.image_mode == "grayscale":
            # 灰度模式：RGB → 灰度 → Tensor
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 图像标准化
        if self.normalize_images:
            if self.custom_image_norm_stats:
                # 使用自定义标准化参数
                mean = self.custom_image_norm_stats["mean"]
                std = self.custom_image_norm_stats["std"]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            elif self.image_mode == "rgb":
                # RGB模式：使用ImageNet标准化参数
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
            else:
                # 灰度模式：使用通用灰度标准化参数
                transform_list.append(transforms.Normalize(
                    mean=[0.5], 
                    std=[0.5]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
        
        print(f"图像处理模式: {self.image_mode}")
        print(f"图像标准化: {self.normalize_images}")
        if self.normalize_images:
            if self.custom_image_norm_stats:
                print(f"使用自定义标准化参数")
            elif self.image_mode == "rgb":
                print(f"使用ImageNet标准化参数")
            else:
                print(f"使用灰度标准化参数")
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample_row = self.csv_data.iloc[idx]
        sample_id = sample_row['sample_id']
        ball_count = sample_row['ball_count']
        json_path = sample_row['json_path']
        
        full_json_path = json_path
        sequence_data = self._load_sequence_data(full_json_path, json_path)
        
        return {
            'sample_id': sample_id,
            'sequence_data': sequence_data,
            'label': ball_count
        }
    
    def _load_image(self, image_path):
        """
        加载并处理单张图像
        处理流程：根据image_mode进行RGB或灰度处理 → 张量转换 → 标准化
        """
        try:
            # 构建完整的图像路径
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found: {full_image_path}")
                # 返回对应通道数的零张量
                channels = 3 if self.image_mode == "rgb" else 1
                return torch.zeros(channels, 224, 224)
            
            # 加载图像（保持RGB格式，变换管道会处理颜色转换）
            image = Image.open(full_image_path).convert('RGB')
            
            # 应用变换管道
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            channels = 3 if self.image_mode == "rgb" else 1
            return torch.zeros(channels, 224, 224)
    
    def _safe_extract_value(self, data, key, default_value, expected_type=None):
        """安全地从字典中提取值，处理None和类型检查"""
        value = data.get(key, default_value)
        
        # 如果值为None，返回默认值
        if value is None:
            return default_value
        
        # 如果指定了期望类型，进行类型检查
        if expected_type is not None:
            if not isinstance(value, expected_type):
                return default_value
        
        return value
    
    def _load_sequence_data(self, json_path, relative_json_path):
        """加载并处理序列数据"""
        try:
            # 加载JSON数据
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            original_length = len(frames)
            
            # 调整序列长度
            if len(frames) < self.sequence_length:
                # 不足时重复最后一帧
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                # 超过时取最后的几帧
                frames = frames[-self.sequence_length:]
            
            # 提取各类数据
            joints_list = []
            timestamps_list = []
            labels_list = []
            images_list = []
            
            for frame in frames:
                # 安全地提取关节位置 (7个关节值)
                joints = self._safe_extract_value(frame, 'joints', [0.0] * 7, list)
                # 确保joints列表中没有None值
                joints = [float(j) if j is not None else 0.0 for j in joints]
                # 确保长度为7
                if len(joints) != 7:
                    joints = joints[:7] + [0.0] * max(0, 7 - len(joints))
                joints_list.append(joints)
                # 安全地提取时间戳
                timestamp = self._safe_extract_value(frame, 'timestamp', 0.0, (int, float))
                timestamps_list.append(float(timestamp))
                
                # 安全地提取标签 (当前帧数数到的数量)
                label = self._safe_extract_value(frame, 'label', 0, (int, float))
                labels_list.append(float(label))
                
                # 图像路径处理 - 从JSON中的image_path提取相对路径
                image_path = self._safe_extract_value(frame, 'image_path', '', str)
                
                if image_path:
                    # 移除绝对路径前缀，保留相对路径
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                    else:
                        relative_image_path = image_path
                    
                    # 修复路径中的命名不一致问题
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                    
                    # 根据设定的image_mode加载对应格式的图像
                    image = self._load_image(relative_image_path)
                else:
                    # 如果没有图像路径，创建零张量
                    channels = 3 if self.image_mode == "rgb" else 1
                    image = torch.zeros(channels, 224, 224)
                
                images_list.append(image)
            
            # 转换为numpy数组，确保数据类型正确
            joints = np.array(joints_list, dtype=np.float32)
            timestamps = np.array(timestamps_list, dtype=np.float32)
            labels = np.array(labels_list, dtype=np.float32)
            images = torch.stack(images_list)
            
            # 归一化（只对非零数据进行归一化）
            if self.normalize and self.normalizer:
                # 检查数据是否有效
                if not np.any(np.isnan(joints)) and not np.any(np.isinf(joints)):
                    joints = self.normalizer.normalize(joints, 'joints')
                else:
                    print(f"警告: joints数据包含NaN或Inf值，跳过归一化")
                
                if not np.any(np.isnan(timestamps)) and not np.any(np.isinf(timestamps)):
                    timestamps = self.normalizer.normalize(timestamps, 'timestamps')
                else:
                    print(f"警告: timestamps数据包含NaN或Inf值，跳过归一化")
            
            # 安全地提取ball_count
            ball_count = self._safe_extract_value(json_data, 'ball_count', 0, (int, float))
            
            return {
                'joints': torch.tensor(joints, dtype=torch.float32),
                'timestamps': torch.tensor(timestamps, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.float32),
                'images': images,
                'sequence_length': original_length,
                'ball_count': int(ball_count)
            }
            
        except Exception as e:
            print(f"处理JSON文件失败 {json_path}: {e}")
            import traceback
            traceback.print_exc()
            # 返回零填充的默认数据
            channels = 3 if self.image_mode == "rgb" else 1
            return {
                'joints': torch.zeros(self.sequence_length, 7, dtype=torch.float32),
                'timestamps': torch.zeros(self.sequence_length, dtype=torch.float32),
                'labels': torch.zeros(self.sequence_length, dtype=torch.float32),
                'images': torch.zeros(self.sequence_length, channels, 224, 224),
                'sequence_length': 1,
                'ball_count': 0
            }
    
    def debug_sample_loading(self, sample_idx=0):
        """调试样本加载问题"""
        print(f"调试样本 {sample_idx} 的加载...")
        
        sample_row = self.csv_data.iloc[sample_idx]
        json_path = sample_row['json_path']
        full_json_path = json_path
        
        print(f"JSON路径: {full_json_path}")
        print(f"文件是否存在: {os.path.exists(full_json_path)}")
        
        if os.path.exists(full_json_path):
            try:
                # 加载JSON数据
                with open(full_json_path, 'r') as f:
                    json_data = json.load(f)
                
                print(f"样本ID: {json_data.get('sample_id', 'N/A')}")
                print(f"球数: {json_data.get('ball_count', 'N/A')}")
                print(f"序列长度: {json_data.get('sequence_length', 'N/A')}")
                print(f"帧数: {len(json_data.get('frames', []))}")
                
                # 检查前几帧的数据完整性
                frames = json_data.get('frames', [])
                for i, frame in enumerate(frames[:3]):  # 检查前3帧
                    print(f"\n帧 {i}:")
                    print(f"  joints: {frame.get('joints', 'Missing')}")
                    print(f"  timestamp: {frame.get('timestamp', 'Missing')}")
                    print(f"  label: {frame.get('label', 'Missing')}")
                    print(f"  balls_detected: {frame.get('balls_detected', 'Missing')}")
                    print(f"  image_path: {frame.get('image_path', 'Missing')}")
                    
                    # 检查joints中是否有None值
                    joints = frame.get('joints', [])
                    if joints:
                        none_count = sum(1 for j in joints if j is None)
                        if none_count > 0:
                            print(f"  !! joints中有{none_count}个None值")
                
            except Exception as e:
                print(f"读取JSON文件时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 尝试加载完整样本
        try:
            sample = self[sample_idx]
            print(f"\n成功加载样本，标签: {sample['label']}")
            print("序列数据形状:")
            for key, value in sample['sequence_data'].items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"\n加载样本失败: {e}")
            import traceback
            traceback.print_exc()
    
    def get_normalizer(self):
        """获取归一化器"""
        return self.normalizer
    
    def denormalize(self, data, modality):
        """反归一化数据"""
        if self.normalize and self.normalizer:
            return self.normalizer.denormalize(data, modality)
        return data


def get_ball_counting_data_loaders(train_csv_path, val_csv_path, data_root, 
                                   batch_size=32, sequence_length=6, 
                                   normalize=True, norm_stats_path=None,
                                   num_workers=4, 
                                   image_mode="rgb", normalize_images=True,
                                   custom_image_norm_stats=None):
    """
    创建球类计数的训练和验证数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
        val_csv_path: 验证集CSV路径
        data_root: 数据根目录
        batch_size: 批次大小
        sequence_length: 序列长度
        normalize: 是否归一化关节和时间戳数据
        norm_stats_path: 归一化统计信息文件路径
        num_workers: 数据加载器进程数
        image_mode: 图像处理模式，"rgb" 或 "grayscale" - 仅选择一种模式
        normalize_images: 是否对图像进行标准化
        custom_image_norm_stats: 自定义图像标准化参数 {"mean": [...], "std": [...]}
    
    Returns:
        train_loader, val_loader, normalizer
    """
    
    print(f"=== 创建数据加载器 - 图像模式: {image_mode.upper()} ===")
    
    # 创建训练集
    train_dataset = BallCountingDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize=normalize,
        norm_stats_path=norm_stats_path,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # 创建验证集（使用相同的参数）
    val_dataset = BallCountingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize=normalize,
        norm_stats_path=norm_stats_path,
        image_mode=image_mode,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
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
    
    return train_loader, val_loader, train_dataset.get_normalizer()


# 测试代码
if __name__ == "__main__":
    # 数据集路径配置
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== 球类计数数据集测试 ===")
    
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
        # 选项1: 测试RGB模式
        print("\n=== 测试RGB模式 ===")
        train_loader_rgb, val_loader_rgb, normalizer_rgb = get_ball_counting_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize=True,
            image_mode="rgb",           # 只加载RGB图像
            normalize_images=True
        )
        
        print(f"RGB模式 - 训练集大小: {len(train_loader_rgb.dataset)}")
        print(f"RGB模式 - 验证集大小: {len(val_loader_rgb.dataset)}")
        
        # 测试RGB batch数据
        for batch in train_loader_rgb:
            print("RGB Batch shapes:")
            print(f"  Images: {batch['sequence_data']['images'].shape}")
            print(f"  图像通道数: {batch['sequence_data']['images'].shape[2]} (应该是3)")
            print(f"  图像值范围: [{batch['sequence_data']['images'].min():.3f}, {batch['sequence_data']['images'].max():.3f}]")
            break
        
        # 选项2: 测试灰度模式
        print("\n=== 测试灰度模式 ===")
        train_loader_gray, val_loader_gray, normalizer_gray = get_ball_counting_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize=True,
            image_mode="grayscale",     # 只加载灰度图像
            normalize_images=True
        )
        
        print(f"灰度模式 - 训练集大小: {len(train_loader_gray.dataset)}")
        print(f"灰度模式 - 验证集大小: {len(val_loader_gray.dataset)}")
        
        # 测试灰度batch数据
        for batch in train_loader_gray:
            print("灰度 Batch shapes:")
            print(f"  Images: {batch['sequence_data']['images'].shape}")
            print(f"  图像通道数: {batch['sequence_data']['images'].shape[2]} (应该是1)")
            print(f"  图像值范围: [{batch['sequence_data']['images'].min():.3f}, {batch['sequence_data']['images'].max():.3f}]")
            break
        
        print("\n=== 使用示例 ===")
        print("# 如果你想要RGB模式:")
        print("train_loader, val_loader, normalizer = get_ball_counting_data_loaders(")
        print("    train_csv, val_csv, data_root, image_mode='rgb')")
        print()
        print("# 如果你想要灰度模式:")
        print("train_loader, val_loader, normalizer = get_ball_counting_data_loaders(")
        print("    train_csv, val_csv, data_root, image_mode='grayscale')")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()