# 基于信息理论的数字感知神经网络实验设计

本实验旨在使用深度神经网络验证Cheyette和Piantadosi提出的数字感知信息理论模型，通过控制网络信息处理的有限性，探究其是否自然产生人类数字感知的特性。

## 1. 数据生成与预处理

### 1.1 点阵图像生成

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def generate_dot_array(n, canvas_size=(200, 200), dot_radius=5, 
                      control_type='size', contrast=1.0, seed=None):
    """
    生成n个点的点阵图像
    
    参数:
    - n: 点的数量
    - canvas_size: 画布大小
    - dot_radius: 基础点半径
    - control_type: 控制类型('size', 'density', 'area')
    - contrast: 对比度(0.1-1.6)
    - seed: 随机种子
    
    返回:
    - image: 包含点阵的numpy数组
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 创建空白灰色画布(灰度值0.5)
    canvas = np.ones(canvas_size) * 0.5
    
    # 根据控制类型调整点大小
    if control_type == 'size':
        radii = [dot_radius] * n  # 固定大小
    elif control_type == 'area':
        # 总面积固定
        total_area = np.pi * (dot_radius**2) * n
        radius = np.sqrt(total_area / (np.pi * n))
        radii = [radius] * n
    elif control_type == 'density':
        # 固定密度
        radii = [dot_radius] * n
        # 限制点的分布区域
        canvas_radius = np.sqrt(900 * n)  # 根据密度计算限制区域
    
    # 生成随机位置
    positions = []
    for i in range(n):
        valid_position = False
        attempts = 0
        
        while not valid_position and attempts < 100:
            if control_type == 'density':
                # 在限制区域内随机选择位置
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.uniform(0, canvas_radius)
                x = canvas_size[0]//2 + distance * np.cos(angle)
                y = canvas_size[1]//2 + distance * np.sin(angle)
            else:
                # 全画布随机位置
                x = np.random.uniform(radii[i], canvas_size[0] - radii[i])
                y = np.random.uniform(radii[i], canvas_size[1] - radii[i])
            
            # 检查是否与已有点重叠
            if all(np.sqrt((x-px)**2 + (y-py)**2) > (radii[i] + pr) 
                  for px, py, pr in positions):
                valid_position = True
                positions.append((x, y, radii[i]))
            
            attempts += 1
        
        if not valid_position:
            # 如果无法找到不重叠的位置，强制放置
            x = np.random.uniform(radii[i], canvas_size[0] - radii[i])
            y = np.random.uniform(radii[i], canvas_size[1] - radii[i])
            positions.append((x, y, radii[i]))
    
    # 在画布上绘制点
    for x, y, r in positions:
        # 创建点的掩码
        y_grid, x_grid = np.ogrid[-r:r+1, -r:r+1]
        mask = x_grid**2 + y_grid**2 <= r**2
        
        # 计算点在画布上的位置
        x_start, x_end = max(0, int(x-r)), min(canvas_size[0], int(x+r+1))
        y_start, y_end = max(0, int(y-r)), min(canvas_size[1], int(y+r+1))
        
        # 获取掩码对应部分
        mask_x_start = max(0, -int(x-r))
        mask_x_end = mask.shape[1] - max(0, int(x+r+1) - canvas_size[0])
        mask_y_start = max(0, -int(y-r))
        mask_y_end = mask.shape[0] - max(0, int(y+r+1) - canvas_size[1])
        
        mask_portion = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        
        # 应用对比度
        # 计算点的灰度值(0.5为背景灰度，值越低越黑)
        dot_value = 0.5 - 0.5 * contrast
        
        # 在画布上绘制点
        canvas_portion = canvas[y_start:y_end, x_start:x_end]
        canvas_portion[mask_portion] = dot_value
    
    return canvas

def generate_dataset(n_samples_per_numerosity, numerosities=range(1, 16),
                    control_types=['size', 'density', 'area'],
                    contrasts=[0.1, 0.2, 0.4, 0.8, 1.6]):
    """
    生成完整的数据集
    
    参数:
    - n_samples_per_numerosity: 每个数量的样本数
    - numerosities: 要生成的数量列表
    - control_types: 控制类型列表
    - contrasts: 对比度列表
    
    返回:
    - dataset: 包含图像、标签和元数据的字典
    """
    dataset = {
        'images': [],
        'numerosities': [],
        'control_types': [],
        'contrasts': []
    }
    
    for n in numerosities:
        for control in control_types:
            for contrast in contrasts:
                for i in range(n_samples_per_numerosity):
                    # 使用不同的随机种子生成多样的样本
                    seed = hash(f"{n}_{control}_{contrast}_{i}") % 10000
                    
                    # 生成点阵图像
                    image = generate_dot_array(
                        n=n, 
                        control_type=control, 
                        contrast=contrast,
                        seed=seed
                    )
                    
                    # 将图像和元数据添加到数据集
                    dataset['images'].append(image)
                    dataset['numerosities'].append(n)
                    dataset['control_types'].append(control)
                    dataset['contrasts'].append(contrast)
    
    return dataset
```

### 1.2 数据采样与分布

```python
def sample_according_to_prior(dataset, n_samples, alpha=2.0):
    """
    根据先验分布P(n)∝1/n^alpha从数据集中采样
    
    参数:
    - dataset: 数据集字典
    - n_samples: 要采样的样本数
    - alpha: 先验分布的幂
    
    返回:
    - sampled_indices: 采样的索引列表
    """
    # 提取数量标签
    numerosities = np.array(dataset['numerosities'])
    
    # 计算每个数量的先验概率
    unique_nums = np.unique(numerosities)
    probs = 1 / (unique_nums ** alpha)
    probs = probs / np.sum(probs)  # 归一化
    
    # 按先验分布采样数量
    sampled_nums = np.random.choice(unique_nums, size=n_samples, p=probs)
    
    # 对于每个采样的数量，随机选择一个样本
    sampled_indices = []
    for n in sampled_nums:
        # 找到所有具有该数量的样本
        indices = np.where(numerosities == n)[0]
        # 随机选择一个
        sampled_idx = np.random.choice(indices)
        sampled_indices.append(sampled_idx)
    
    return sampled_indices
```

### 1.3 时间模拟变换

```python
def simulate_exposure_time(images, exposure_times=[40, 80, 160, 320, 640], max_time=640):
    """
    模拟不同曝光时间的图像处理
    
    参数:
    - images: 原始图像列表
    - exposure_times: 要模拟的曝光时间(毫秒)
    - max_time: 最大曝光时间
    
    返回:
    - time_simulated_images: 字典，键为曝光时间，值为处理后的图像
    """
    time_simulated_images = {}
    
    for t in exposure_times:
        # 计算信息比例
        info_ratio = min(1.0, t / max_time)
        
        # 处理每个图像
        processed_images = []
        for img in images:
            # 1. 增加与曝光时间相关的噪声
            noise_level = 1.0 - info_ratio
            noise = np.random.normal(0, noise_level * 0.1, img.shape)
            
            # 2. 模糊程度与曝光时间相关
            if info_ratio < 1.0:
                from scipy.ndimage import gaussian_filter
                blur_sigma = max(0, 1.0 - info_ratio) * 2
                processed = gaussian_filter(img, sigma=blur_sigma)
            else:
                processed = img.copy()
            
            # 3. 应用噪声
            processed = np.clip(processed + noise, 0, 1)
            
            processed_images.append(processed)
        
        time_simulated_images[t] = processed_images
    
    return time_simulated_images
```

## 2. 模型架构

### 2.1 基础视觉特征提取器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 基础CNN特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # 输出特征维度: 128 x 7 x 7
        self.feature_dim = 128 * 7 * 7
    
    def forward(self, x):
        # 期望输入: (batch_size, 1, height, width)
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平为特征向量
        return x
```

### 2.2 信息约束VAE模型

```python
class InfoConstrainedVAE(nn.Module):
    def __init__(self, feature_dim=128*7*7, latent_dim=32, target_capacity=2.0):
        """
        带有信息容量约束的变分自编码器
        
        参数:
        - feature_dim: 输入特征维度
        - latent_dim: 潜在空间维度
        - target_capacity: 目标信息容量(比特)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.target_capacity = target_capacity
        
        # 特征提取器
        self.feature_extractor = VisualFeatureExtractor()
        
        # 编码器(将特征映射到潜在空间)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 均值和方差预测
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 解码器(将潜在表示解码为数字估计)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # 确保输出为非负数
        )
        
        # 用于动态调整β的参数
        self.beta = nn.Parameter(torch.ones(1))
        self.register_buffer('kl_target', torch.tensor([target_capacity]))
    
    def encode(self, x):
        """
        将输入编码为潜在表示
        """
        # 提取视觉特征
        features = self.feature_extractor(x)
        
        # 编码特征
        h = self.encoder(features)
        
        # 生成均值和对数方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        使用重参数化技巧进行采样
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        从潜在表示解码为数字估计
        """
        return self.decoder(z)
    
    def forward(self, x, return_all=False):
        """
        前向传播
        
        参数:
        - x: 输入图像 (batch_size, 1, height, width)
        - return_all: 是否返回所有中间结果
        
        返回:
        - numerosity: 预测的数字
        - mu, logvar: 潜在空间的参数(如果return_all=True)
        - kl_divergence: KL散度值(如果return_all=True)
        """
        # 编码
        mu, logvar = self.encode(x)
        
        # KL散度
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # 采样潜在变量
        z = self.reparameterize(mu, logvar)
        
        # 解码为数字估计
        numerosity = self.decode(z).squeeze()
        
        if return_all:
            return numerosity, mu, logvar, kl_divergence
        else:
            return numerosity
    
    def get_capacity_loss(self, kl_divergence):
        """
        计算容量约束损失
        """
        # 计算KL散度与目标容量的差异
        kl_mean = torch.mean(kl_divergence)
        capacity_loss = self.beta * torch.abs(kl_mean - self.kl_target)
        return capacity_loss, kl_mean
```

### 2.3 时间约束RNN模型

```python
class TimeConstrainedRNN(nn.Module):
    def __init__(self, feature_dim=128*7*7, hidden_dim=256, target_capacity=2.0):
        """
        时间约束的递归神经网络模型
        
        参数:
        - feature_dim: 输入特征维度
        - hidden_dim: RNN隐藏状态维度
        - target_capacity: 目标信息容量(比特)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.target_capacity = target_capacity
        
        # 特征提取器
        self.feature_extractor = VisualFeatureExtractor()
        
        # 特征转换(将提取的特征变换为序列)
        self.feature_transform = nn.Linear(feature_dim, hidden_dim)
        
        # RNN层
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 数字预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # 确保输出为非负数
        )
    
    def forward(self, x, timesteps=None):
        """
        前向传播
        
        参数:
        - x: 输入图像 (batch_size, 1, height, width)
        - timesteps: 处理的时间步数，模拟不同的曝光时间
        
        返回:
        - numerosity: 预测的数字
        """
        batch_size = x.size(0)
        
        # 提取视觉特征
        features = self.feature_extractor(x)
        
        # 转换为序列特征(拆分为10个时间步)
        sequence_length = 10
        transformed = self.feature_transform(features)
        sequence = transformed.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # 如果指定了时间步数，则限制处理的序列长度
        if timesteps is not None:
            max_steps = int(timesteps / 64)  # 将时间(ms)映射到步数(0-10)
            sequence = sequence[:, :max_steps, :]
        
        # 通过RNN处理序列
        _, hidden = self.rnn(sequence)
        
        # 从最终隐藏状态预测数字
        numerosity = self.predictor(hidden.squeeze(0))
        
        return numerosity.squeeze()
```

### 2.4 模型组合

```python
class InfoNumerosityModel(nn.Module):
    """
    完整的数字感知模型，结合VAE和RNN
    """
    def __init__(self, model_type='vae', target_capacity=2.0):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'vae':
            self.model = InfoConstrainedVAE(target_capacity=target_capacity)
        elif model_type == 'rnn':
            self.model = TimeConstrainedRNN(target_capacity=target_capacity)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def forward(self, x, timesteps=None, return_all=False):
        if self.model_type == 'vae':
            return self.model(x, return_all)
        else:  # rnn
            return self.model(x, timesteps)
```

## 3. 训练过程

### 3.1 损失函数设计

```python
def numerosity_loss_function(prediction, target, kl_divergence=None, 
                            beta=1.0, target_capacity=2.0):
    """
    数字感知模型的损失函数
    
    参数:
    - prediction: 模型预测的数字
    - target: 真实数字
    - kl_divergence: KL散度(用于VAE)
    - beta: KL散度的权重
    - target_capacity: 目标信息容量(比特)
    
    返回:
    - total_loss: 总损失
    - loss_components: 损失的各个组成部分
    """
    # MSE损失
    mse_loss = F.mse_loss(prediction, target)
    
    # 初始化损失组件字典
    loss_components = {'mse': mse_loss.item()}
    
    if kl_divergence is not None:
        # 计算KL散度的均值
        kl_mean = torch.mean(kl_divergence)
        
        # 容量损失
        capacity_loss = beta * torch.abs(kl_mean - target_capacity)
        
        # 添加损失组件
        loss_components['kl'] = kl_mean.item()
        loss_components['capacity'] = capacity_loss.item()
        
        # 总损失
        total_loss = mse_loss + capacity_loss
    else:
        total_loss = mse_loss
    
    return total_loss, loss_components
```

### 3.2 训练循环

```python
def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    训练一个epoch
    
    参数:
    - model: 模型
    - dataloader: 数据加载器
    - optimizer: 优化器
    - device: 设备
    - scheduler: 学习率调度器(可选)
    
    返回:
    - avg_loss: 平均损失
    - loss_stats: 损失统计信息
    """
    model.train()
    running_loss = 0.0
    loss_stats = {'total': 0, 'mse': 0, 'kl': 0, 'capacity': 0}
    
    for batch_idx, (data, target, metadata) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if model.model_type == 'vae':
            # 对于VAE模型，需要获取KL散度
            prediction, _, _, kl_divergence = model(data, return_all=True)
            
            # 计算损失
            loss, loss_components = numerosity_loss_function(
                prediction, target, kl_divergence, 
                beta=model.model.beta, 
                target_capacity=model.model.target_capacity
            )
        else:  # rnn模型
            # 获取当前批次的时间步数
            timesteps = metadata.get('exposure_time', None)
            if timesteps is not None:
                timesteps = timesteps.to(device)
            
            # 前向传播
            prediction = model(data, timesteps)
            
            # 计算损失
            loss, loss_components = numerosity_loss_function(
                prediction, target
            )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        running_loss += loss.item()
        for k, v in loss_components.items():
            loss_stats[k] += v
    
    # 学习率调整
    if scheduler is not None:
        scheduler.step()
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    for k in loss_stats:
        loss_stats[k] /= len(dataloader)
    
    return avg_loss, loss_stats
```

### 3.3 验证循环

```python
def validate(model, dataloader, device):
    """
    在验证集上评估模型
    
    参数:
    - model: 模型
    - dataloader: 数据加载器
    - device: 设备
    
    返回:
    - avg_loss: 平均损失
    - predictions: 预测结果
    - analysis_data: 分析数据
    """
    model.eval()
    running_loss = 0.0
    
    # 收集预测和真实值
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        for data, target, metadata in dataloader:
            data, target = data.to(device), target.to(device)
            
            if model.model_type == 'vae':
                prediction = model(data)
            else:  # rnn模型
                timesteps = metadata.get('exposure_time', None)
                if timesteps is not None:
                    timesteps = timesteps.to(device)
                prediction = model(data, timesteps)
            
            # 计算MSE损失
            loss = F.mse_loss(prediction, target)
            running_loss += loss.item()
            
            # 收集结果
            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # 收集元数据(如数量、对比度、控制类型等)
            batch_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, torch.Tensor):
                    batch_metadata[k] = v.cpu().numpy()
                else:
                    batch_metadata[k] = v
            all_metadata.append(batch_metadata)
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    
    # 合并结果
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # 合并元数据
    metadata_dict = {}
    for k in all_metadata[0].keys():
        metadata_dict[k] = np.concatenate([m[k] for m in all_metadata])
    
    # 创建分析数据字典
    analysis_data = {
        'predictions': predictions,
        'targets': targets,
        'metadata': metadata_dict
    }
    
    return avg_loss, predictions, analysis_data
```

## 4. 实验分析

### 4.1 主要评估指标

```python
def analyze_numerosity_perception(analysis_data, log_dir=None):
    """
    分析数字感知结果
    
    参数:
    - analysis_data: 分析数据字典
    - log_dir: 保存结果的目录
    
    返回:
    - results: 分析结果字典
    """
    predictions = analysis_data['predictions']
    targets = analysis_data['targets']
    metadata = analysis_data['metadata']
    
    # 创建结果字典
    results = {}
    
    # 1. 计算全局度量
    results['global'] = {
        'mse': np.mean((predictions - targets) ** 2),
        'mae': np.mean(np.abs(predictions - targets)),
        'r2': r2_score(targets, predictions)
    }
    
    # 2. 分析不同数量的表现
    unique_nums = np.unique(targets)
    num_results = {}
    
    for n in unique_nums:
        mask = (targets == n)
        n_preds = predictions[mask]
        
        # 计算平均估计和标准差
        mean_estimate = np.mean(n_preds)
        std_estimate = np.std(n_preds)
        
        # 计算绝对误差
        abs_error = np.mean(np.abs(n_preds - n))
        
        num_results[int(n)] = {
            'mean_estimate': mean_estimate,
            'std_estimate': std_estimate,
            'abs_error': abs_error,
            'coefficient_variation': std_estimate / mean_estimate if mean_estimate > 0 else np.nan
        }
    
    results['by_numerosity'] = num_results
    
    # 3. 分析不同曝光时间/对比度的表现
    if 'exposure_time' in metadata:
        time_results = {}
        times = np.unique(metadata['exposure_time'])
        
        for t in times:
            mask = (metadata['exposure_time'] == t)
            t_preds = predictions[mask]
            t_targets = targets[mask]
            
            # 计算每个时间条件下不同数量的表现
            t_num_results = {}
            for n in unique_nums:
                n_mask = (t_targets == n)
                tn_preds = t_preds[n_mask]
                tn_targets = t_targets[n_mask]
                
                if len(tn_preds) > 0:
                    t_num_results[int(n)] = {
                        'mean_estimate': np.mean(tn_preds),
                        'std_estimate': np.std(tn_preds),
                        'abs_error': np.mean(np.abs(tn_preds - n))
                    }
            
            time_results[int(t)] = {
                'overall_mae': np.mean(np.abs(t_preds - t_targets)),
                'by_numerosity': t_num_results
            }
        
        results['by_exposure_time'] = time_results
    
    if 'contrast' in metadata:
        contrast_results = {}
        contrasts = np.unique(metadata['contrast'])
        
        for c in contrasts:
            mask = (metadata['contrast'] == c)
            c_preds = predictions[mask]
            c_targets = targets[mask]
            
            # 类似地处理对比度
            c_num_results = {}
            for n in unique_nums:
                n_mask = (c_targets == n)
                cn_preds = c_preds[n_mask]
                
                if len(cn_preds) > 0:
                    c_num_results[int(n)] = {
                        'mean_estimate': np.mean(cn_preds),
                        'std_estimate': np.std(cn_preds),
                        'abs_error': np.mean(np.abs(cn_preds - n))
                    }
            
            contrast_results[float(c)] = {
                'overall_mae': np.mean(np.abs(c_preds - c_targets)),
                'by_numerosity': c_num_results
            }
        
        results['by_contrast'] = contrast_results
    
    # 4. 确定计数范围(精确表示的最大数量)
    # 定义"精确"为平均绝对误差小于0.5
    subitizing_range = []
    for n in sorted(unique_nums):
        if num_results[int(n)]['abs_error'] < 0.5:
            subitizing_range.append(int(n))
        else:
            break
    
    if subitizing_range:
        results['subitizing_range'] = max(subitizing_range)
    else:
        results['subitizing_range'] = 0
    
    # 如果提供了日志目录，保存结果
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results
```

### 4.2 可视化函数

```python
def plot_numerosity_perception_results(results, output_dir=None):
    """
    绘制数字感知结果
    
    参数:
    - results: 分析结果字典
    - output_dir: 保存图表的目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置样式
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 创建输出目录
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 绘制平均估计值与真实数量的关系
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_data = results['by_numerosity']
    numerosities = sorted(list(map(int, num_data.keys())))
    mean_estimates = [num_data[n]['mean_estimate'] for n in numerosities]
    
    ax.plot(numerosities, mean_estimates, 'o-', label='Model Estimates')
    ax.plot(numerosities, numerosities, 'k--', label='Veridical')
    
    ax.set_xlabel('Actual Numerosity')
    ax.set_ylabel('Mean Estimate')
    ax.set_title('Mean Estimates vs. Actual Numerosity')
    ax.legend()
    ax.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'mean_estimates.png'), dpi=300, bbox_inches='tight')
    
    # 2. 绘制绝对误差与数量的关系
    fig, ax = plt.subplots(figsize=(10, 6))
    
    abs_errors = [num_data[n]['abs_error'] for n in numerosities]
    
    ax.plot(numerosities, abs_errors, 'o-')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Subitizing Threshold')
    
    ax.set_xlabel('Numerosity')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Absolute Error vs. Numerosity (Subitizing Range: {results["subitizing_range"]})')
    ax.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'absolute_error.png'), dpi=300, bbox_inches='tight')
    
    # 3. 如果有曝光时间数据，绘制不同时间条件下的结果
    if 'by_exposure_time' in results:
        time_data = results['by_exposure_time']
        times = sorted(list(map(int, time_data.keys())))
        
        # 3.1 不同时间条件下的平均估计
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for t in times:
            t_num_data = time_data[t]['by_numerosity']
            t_numerosities = sorted([n for n in numerosities if n in t_num_data])
            t_means = [t_num_data[n]['mean_estimate'] for n in t_numerosities]
            
            ax.plot(t_numerosities, t_means, 'o-', label=f'{t} ms')
        
        ax.plot(numerosities, numerosities, 'k--', label='Veridical')
        
        ax.set_xlabel('Actual Numerosity')
        ax.set_ylabel('Mean Estimate')
        ax.set_title('Mean Estimates vs. Actual Numerosity by Exposure Time')
        ax.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'mean_estimates_by_time.png'), dpi=300, bbox_inches='tight')
        
        # 3.2 不同时间条件下的绝对误差
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for t in times:
            t_num_data = time_data[t]['by_numerosity']
            t_numerosities = sorted([n for n in numerosities if n in t_num_data])
            t_errors = [t_num_data[n]['abs_error'] for n in t_numerosities]
            
            ax.plot(t_numerosities, t_errors, 'o-', label=f'{t} ms')
        
        ax.axhline(y=0.5, color='r', linestyle='--', label='Subitizing Threshold')
        
        ax.set_xlabel('Numerosity')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error vs. Numerosity by Exposure Time')
        ax.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'absolute_error_by_time.png'), dpi=300, bbox_inches='tight')
    
    # 4. 标量可变性分析(标准差与均值的比例)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 忽略很小的数(避免除以接近零的值)
    cv_data = [(n, num_data[n]['coefficient_variation']) for n in numerosities if n > 1]
    ns, cvs = zip(*cv_data)
    
    ax.plot(ns, cvs, 'o-')
    ax.set_xlabel('Numerosity')
    ax.set_ylabel('Coefficient of Variation (σ/μ)')
    ax.set_title('Scalar Variability Analysis')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'scalar_variability.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
```

## 5. 完整实验流程

```python
def run_info_numerosity_experiment(capacities=[2.0, 4.0, 6.0], model_type='vae', 
                                  num_epochs=50, batch_size=64, learning_rate=1e-4):
    """
    运行完整的信息数字感知实验
    
    参数:
    - capacities: 要测试的信息容量列表(比特)
    - model_type: 模型类型('vae'或'rnn')
    - num_epochs: 训练轮数
    - batch_size: 批次大小
    - learning_rate: 学习率
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备输出目录
    base_output_dir = f'results/info_numerosity_{model_type}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 生成数据集
    print("生成数据集...")
    dataset = generate_dataset(
        n_samples_per_numerosity=50,  # 每个数量50个样本
        numerosities=range(1, 16),    # 1到15个点
        control_types=['size'],       # 简化为仅使用大小控制
        contrasts=[0.8]               # 简化为仅使用标准对比度
    )
    
    # 模拟不同的曝光时间
    exposure_times = [40, 80, 160, 320, 640]  # 毫秒
    time_simulated_images = simulate_exposure_time(
        dataset['images'], 
        exposure_times=exposure_times
    )
    
    # 准备训练和测试数据
    for capacity in capacities:
        print(f"\n------- 训练容量为 {capacity} 比特的模型 -------")
        
        # 创建输出目录
        output_dir = os.path.join(base_output_dir, f'capacity_{capacity}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建模型
        model = InfoNumerosityModel(model_type=model_type, target_capacity=capacity).to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # 训练记录
        train_losses = []
        val_losses = []
        
        # 准备数据加载器(使用所有时间条件)
        train_data = []
        val_data = []
        
        for t in exposure_times:
            # 获取该时间条件下的图像
            t_images = time_simulated_images[t]
            
            # 构建数据集
            for idx, img in enumerate(t_images):
                sample = {
                    'image': torch.tensor(img, dtype=torch.float32).unsqueeze(0),  # 添加通道维度
                    'numerosity': torch.tensor(dataset['numerosities'][idx], dtype=torch.float32),
                    'metadata': {
                        'exposure_time': t,
                        'contrast': dataset['contrasts'][idx],
                        'control_type': dataset['control_types'][idx]
                    }
                }
                
                # 80%用于训练，20%用于验证
                if np.random.rand() < 0.8:
                    train_data.append(sample)
                else:
                    val_data.append(sample)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([x['image'] for x in batch]),
                torch.stack([x['numerosity'] for x in batch]),
                {k: [x['metadata'][k] for x in batch] for k in batch[0]['metadata']}
            )
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([x['image'] for x in batch]),
                torch.stack([x['numerosity'] for x in batch]),
                {k: [x['metadata'][k] for x in batch] for k in batch[0]['metadata']}
            )
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练
            train_loss, loss_stats = train_epoch(
                model, train_loader, optimizer, device, scheduler
            )
            
            # 验证
            val_loss, _, _ = validate(model, val_loader, device)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if model_type == 'vae':
                    print(f"KL: {loss_stats['kl']:.4f}, Capacity: {loss_stats['capacity']:.4f}")
        
        # 保存训练损失图
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress (Capacity: {capacity} bits)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # 在每个时间条件下分别评估模型
        for t in exposure_times:
            print(f"\n评估时间条件: {t} ms")
            
            # 创建该时间条件的测试数据
            t_test_data = []
            for idx, img in enumerate(time_simulated_images[t]):
                sample = {
                    'image': torch.tensor(img, dtype=torch.float32).unsqueeze(0),
                    'numerosity': torch.tensor(dataset['numerosities'][idx], dtype=torch.float32),
                    'metadata': {
                        'exposure_time': t,
                        'contrast': dataset['contrasts'][idx],
                        'control_type': dataset['control_types'][idx]
                    }
                }
                t_test_data.append(sample)
            
            # 创建数据加载器
            t_test_loader = torch.utils.data.DataLoader(
                t_test_data, 
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([x['image'] for x in batch]),
                    torch.stack([x['numerosity'] for x in batch]),
                    {k: [x['metadata'][k] for x in batch] for k in batch[0]['metadata']}
                )
            )
            
            # 评估模型
            _, _, analysis_data = validate(model, t_test_loader, device)
            
            # 分析结果
            t_results_dir = os.path.join(output_dir, f'time_{t}ms')
            os.makedirs(t_results_dir, exist_ok=True)
            
            # 分析预测结果
            results = analyze_numerosity_perception(analysis_data, t_results_dir)
            
            # 绘制结果
            plot_numerosity_perception_results(results, t_results_dir)
        
        # 对比模型在不同时间条件下的表现
        plot_comparison_across_times(capacities, exposure_times, base_output_dir)
        
    print("\n实验完成!")
```

### 5.1 比较不同信息容量的结果

```python
def plot_comparison_across_times(capacities, exposure_times, base_output_dir):
    """
    比较不同信息容量和时间条件下的结果
    
    参数:
    - capacities: 信息容量列表
    - exposure_times: 曝光时间列表
    - base_output_dir: 基础输出目录
    """
    import matplotlib.pyplot as plt
    import json
    import os
    
    # 创建比较结果目录
    comparison_dir = os.path.join(base_output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. 比较不同容量下的计数范围
    subitizing_ranges = {}
    
    for capacity in capacities:
        capacity_subitizing = {}
        capacity_dir = os.path.join(base_output_dir, f'capacity_{capacity}')
        
        for t in exposure_times:
            results_file = os.path.join(capacity_dir, f'time_{t}ms', 'analysis_results.json')
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    capacity_subitizing[t] = results.get('subitizing_range', 0)
        
        subitizing_ranges[capacity] = capacity_subitizing
    
    # 绘制计数范围与时间的关系
    plt.figure(figsize=(10, 6))
    
    for capacity, time_data in subitizing_ranges.items():
        times = sorted(time_data.keys())
        ranges = [time_data[t] for t in times]
        
        plt.plot(times, ranges, 'o-', label=f'{capacity} bits')
    
    plt.xlabel('Exposure Time (ms)')
    plt.ylabel('Subitizing Range')
    plt.title('Subitizing Range vs. Exposure Time for Different Information Capacities')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(comparison_dir, 'subitizing_range_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. 比较不同容量下的标量可变性(11-15的平均CV)
    scalar_variability = {}
    
    for capacity in capacities:
        capacity_cv = {}
        capacity_dir = os.path.join(base_output_dir, f'capacity_{capacity}')
        
        for t in exposure_times:
            results_file = os.path.join(capacity_dir, f'time_{t}ms', 'analysis_results.json')
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                    # 计算大数(11-15)的平均CV
                    cv_values = []
                    for n in range(11, 16):
                        if str(n) in results['by_numerosity']:
                            cv = results['by_numerosity'][str(n)].get('coefficient_variation', np.nan)
                            if not np.isnan(cv):
                                cv_values.append(cv)
                    
                    if cv_values:
                        capacity_cv[t] = np.mean(cv_values)
        
        scalar_variability[capacity] = capacity_cv
    
    # 绘制标量可变性与时间的关系
    plt.figure(figsize=(10, 6))
    
    for capacity, time_data in scalar_variability.items():
        times = sorted(time_data.keys())
        cvs = [time_data[t] for t in times]
        
        plt.plot(times, cvs, 'o-', label=f'{capacity} bits')
    
    plt.xlabel('Exposure Time (ms)')
    plt.ylabel('Average Coefficient of Variation (11-15)')
    plt.title('Scalar Variability vs. Exposure Time for Different Information Capacities')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(comparison_dir, 'scalar_variability_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
```

## 6. 主函数

```python
def main():
    """
    主函数
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 运行VAE模型实验
    print("运行VAE模型实验...")
    run_info_numerosity_experiment(
        capacities=[2.0, 4.0, 6.0],
        model_type='vae',
        num_epochs=30,
        batch_size=64
    )
    
    # 运行RNN模型实验
    print("\n运行RNN模型实验...")
    run_info_numerosity_experiment(
        capacities=[2.0, 4.0, 6.0],
        model_type='rnn',
        num_epochs=30,
        batch_size=64
    )
    
    print("\n所有实验完成!")

if __name__ == "__main__":
    main()
```

## 7. 预期结果与讨论

我们预期这个实验将能够验证以下几点：

1. **信息约束效应**：随着信息容量的增加(2→4→6比特)，模型在数字感知任务上的表现应该会呈现质的变化：
   - 低信息容量(2比特)：即使小数量也会表现出估计特性和高误差
   - 中等信息容量(4比特)：小数量(1-4)能够精确表示，而大数量表现出标量可变性
   - 高信息容量(6比特)：精确表示范围扩大到5-6，大数量仍然表现出标量可变性

2. **时间依赖性**：展示持续时间越短，可用信息越少，导致：
   - 精确表示的数量范围缩小
   - 对大数量的估计误差增加
   - 标量可变性出现在更小的数量上

3. **统一系统证据**：我们期望看到在同一个神经网络中，仅通过改变信息约束，就能自然产生分别对小数量的精确表示和对大数量的近似表示，而不需要两个分离的子系统。

这一实验设计不仅可以验证Cheyette和Piantadosi的信息理论模型，还可以提供对大脑如何在信息约束下实现数字感知的新见解。更重要的是，它展示了深度学习模型如何自然地在有限信息条件下产生类似人类的行为模式，支持了"资源理性"(resource-rational)的认知理论框架。