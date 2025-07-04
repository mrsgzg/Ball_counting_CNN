"""
简洁的交互式PCA/t-SNE可视化工具
点击散点图上的点可以查看对应的原始图像
专门针对EmbodiedCountingModel优化
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 服务器环境的非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys
import json
import argparse
import time
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimpleFeatureExtractor:
    """简化的特征提取器 - 只关注核心功能"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.features = {}
        self.hooks = []
        
    def auto_detect_key_layers(self):
        """自动检测关键层"""
        # 精确的层名称匹配
        target_layers = [
            'fusion',              # 多模态融合层
            'lstm',                # 时序处理层
            'counting_decoder',    # 计数解码层
            'visual_encoder',      # 视觉编码层
            'embodiment_encoder'   # 具身编码层
        ]
        
        detected_layers = []
        
        # 检查哪些层真实存在
        for target in target_layers:
            try:
                module = self.model
                for part in target.split('.'):
                    module = getattr(module, part)
                detected_layers.append(target)
                print(f"✅ 检测到关键层: {target}")
            except AttributeError:
                print(f"❌ 未找到层: {target}")
        
        return detected_layers
        
    def register_hooks(self, layers_to_extract):
        """注册钩子函数来提取中间层特征"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0].detach().cpu()
                else:
                    self.features[name] = output.detach().cpu()
            return hook
        
        successful_hooks = []
        
        # 注册钩子
        for layer_name in layers_to_extract:
            try:
                module = self.model
                for part in layer_name.split('.'):
                    module = getattr(module, part)
                
                handle = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(handle)
                successful_hooks.append(layer_name)
                print(f"✅ 成功注册钩子: {layer_name}")
                
            except AttributeError as e:
                print(f"❌ 注册钩子失败: {layer_name} - {e}")
        
        return successful_hooks
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        
    def _process_feature_tensor(self, feature_tensor):
        """处理不同形状的特征张量"""
        if len(feature_tensor.shape) == 3:  # [batch, seq, dim]
            # 取最后一个时刻的特征
            return feature_tensor[:, -1, :].cpu().numpy()
        elif len(feature_tensor.shape) == 4:  # [batch, seq, h, w] or [batch, channel, h, w]
            # 全局平均池化
            pooled = feature_tensor.mean(dim=(-2, -1))
            if len(pooled.shape) == 3:  # 如果还有seq维度
                pooled = pooled[:, -1, :]
            return pooled.cpu().numpy()
        elif len(feature_tensor.shape) == 2:  # [batch, dim]
            return feature_tensor.cpu().numpy()
        else:
            # 其他情况，展平
            return feature_tensor.view(feature_tensor.shape[0], -1).cpu().numpy()
    
    def extract_features(self, data_loader, max_samples=1000):
        """提取特征 - 简化版"""
        all_features = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_sample_ids = []
        all_image_info = []  # 新增：保存图像信息
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                if sample_count >= max_samples:
                    break
                
                # 适配新的数据格式
                sequence_data = {
                    'images': batch['sequence_data']['images'].to(self.device),
                    'joints': batch['sequence_data']['joints'].to(self.device),
                    'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                    'labels': batch['sequence_data']['labels'].to(self.device)
                }
                
                # 获取批次信息
                labels = batch['label'].cpu().numpy()  # CSV中的标签
                sample_ids = batch['sample_id']
                
                # 计算实际处理的样本数
                remaining_samples = max_samples - sample_count
                actual_batch_size = min(len(labels), remaining_samples)
                
                # 截断批次（如果需要）
                if actual_batch_size < len(labels):
                    for key in sequence_data:
                        sequence_data[key] = sequence_data[key][:actual_batch_size]
                    labels = labels[:actual_batch_size]
                    sample_ids = sample_ids[:actual_batch_size]
                
                # 提取图像信息 - 保存原始图像张量用于后续显示
                original_images = sequence_data['images'][:, 0].cpu()  # 取第一帧
                
                # 清空特征字典
                self.features = {}
                
                # 前向传播
                outputs = self.model(
                    sequence_data=sequence_data,
                    use_teacher_forcing=False
                )
                
                # 收集预测结果
                count_logits = outputs['counts']  # [batch, seq_len, 11]
                pred_labels = torch.argmax(count_logits, dim=-1)  # [batch, seq_len]
                final_pred = pred_labels[:, -1].cpu().numpy()  # 最终时刻的预测
                
                all_predictions.extend(final_pred)
                all_labels.extend(labels)
                all_sample_ids.extend(sample_ids)
                
                # 保存图像信息
                for i in range(actual_batch_size):
                    image_info = {
                        'image_tensor': original_images[i],  # [C, H, W]
                        'sample_id': sample_ids[i],
                        'true_label': labels[i],
                        'pred_label': final_pred[i],
                        'global_index': sample_count + i  # 全局索引用于映射
                    }
                    all_image_info.append(image_info)
                
                # 收集中间层特征
                for feature_name, feature_tensor in self.features.items():
                    processed_features = self._process_feature_tensor(feature_tensor)
                    all_features[feature_name].append(processed_features)
                
                sample_count += actual_batch_size
                
                if sample_count >= max_samples:
                    break
        
        # 合并所有特征
        final_features = {}
        for feature_name, feature_list in all_features.items():
            if feature_list:
                final_features[feature_name] = np.vstack(feature_list)
        
        result = {
            'features': final_features,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'sample_ids': all_sample_ids,
            'image_info': all_image_info  # 新增
        }
        
        print(f"特征提取完成:")
        print(f"  实际样本数: {len(result['labels'])}")
        print(f"  提取的特征层: {list(final_features.keys())}")
        print(f"  图像信息数: {len(all_image_info)}")
        
        return result


class InteractiveVisualizer:
    """交互式可视化器 - 简化版本，专注于PCA和t-SNE"""
    
    def __init__(self, image_info, figsize=(15, 10)):
        self.image_info = image_info
        self.figsize = figsize
        self.current_fig = None
        self.current_ax = None
        
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
    
    def tensor_to_image(self, tensor):
        """将张量转换为可显示的图像"""
        if tensor.dim() == 3:
            if tensor.shape[0] == 1:
                # 灰度图像
                return tensor.squeeze(0).numpy()
            elif tensor.shape[0] == 3:
                # RGB图像，需要转置为 (H, W, C)
                return tensor.permute(1, 2, 0).numpy()
        return tensor.numpy()
    
    def reduce_dimensions(self, features, method='pca', n_components=2):
        """降维"""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(features)//4))
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        reduced_features = reducer.fit_transform(features)
        return reduced_features
    
    def create_interactive_plot(self, features_2d, labels, layer_name, method, save_dir):
        """创建交互式散点图 - 保存版本，带点击信息"""
        
        # 创建主散点图
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制散点图
        scatter_plots = []
        for i, label in enumerate(unique_labels):
            mask = labels == label
            scatter = ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                               c=[colors[i]], label=f'Class {label}', 
                               alpha=0.7, s=60, picker=True)
            scatter_plots.append((scatter, mask, label))
        
        ax.set_title(f'{layer_name} - {method.upper()} Visualization\n(Check point_info.txt for detailed sample information)', fontsize=14)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 保存主图
        plt.tight_layout()
        main_plot_path = os.path.join(save_dir, f'{layer_name}_{method}_interactive.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成点信息文件
        self.generate_point_info_file(features_2d, labels, layer_name, method, save_dir)
        
        # 创建示例图像网格 - 显示每个类别的代表性样本
        self.create_sample_grid(labels, layer_name, method, save_dir)
        
        print(f"✅ 交互式可视化已保存: {main_plot_path}")
        print(f"✅ 点信息文件已生成: 查看 point_info.txt")
        print(f"✅ 示例图像网格已生成")
        
        return main_plot_path
    
    def generate_point_info_file(self, features_2d, labels, layer_name, method, save_dir):
        """生成点信息文件，包含每个点的详细信息"""
        
        info_file_path = os.path.join(save_dir, f'{layer_name}_{method}_point_info.txt')
        
        with open(info_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {layer_name} - {method.upper()} 可视化点信息 ===\n\n")
            f.write("格式: 点索引 | 2D坐标 | 样本ID | 真实标签 | 预测标签\n")
            f.write("-" * 80 + "\n")
            
            for i, (x, y) in enumerate(features_2d):
                if i < len(self.image_info):
                    img_info = self.image_info[i]
                    f.write(f"点{i:3d} | ({x:8.3f}, {y:8.3f}) | {img_info['sample_id']:>10} | "
                           f"真实:{img_info['true_label']:2d} | 预测:{img_info['pred_label']:2d}\n")
                else:
                    f.write(f"点{i:3d} | ({x:8.3f}, {y:8.3f}) | 信息缺失\n")
        
        print(f"✅ 点信息文件已保存: {info_file_path}")
    
    def create_sample_grid(self, labels, layer_name, method, save_dir, samples_per_class=3):
        """创建每个类别的代表性样本网格"""
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 计算网格大小
        grid_cols = samples_per_class
        grid_rows = n_classes
        
        fig, axes = plt.subplots(grid_rows, grid_cols, 
                                figsize=(4*grid_cols, 3*grid_rows))
        
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        if grid_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{layer_name} - {method.upper()} Sample Grid\n'
                     f'Representative samples from each class', fontsize=16)
        
        for class_idx, label in enumerate(unique_labels):
            # 找到该类别的所有样本
            class_mask = labels == label
            class_indices = np.where(class_mask)[0]
            
            # 随机选择几个样本（或取前几个）
            selected_indices = class_indices[:samples_per_class] if len(class_indices) >= samples_per_class else class_indices
            
            for sample_idx in range(samples_per_class):
                ax = axes[class_idx, sample_idx]
                
                if sample_idx < len(selected_indices):
                    # 获取对应的图像信息
                    global_idx = selected_indices[sample_idx]
                    if global_idx < len(self.image_info):
                        img_info = self.image_info[global_idx]
                        
                        # 反归一化并显示图像
                        image_tensor = img_info['image_tensor']
                        
                        # 判断图像模式
                        if image_tensor.shape[0] == 3:
                            denorm_image = self.denormalize_image(image_tensor, 'rgb')
                            display_image = self.tensor_to_image(denorm_image)
                            ax.imshow(display_image)
                        elif image_tensor.shape[0] == 1:
                            denorm_image = self.denormalize_image(image_tensor, 'grayscale')
                            display_image = self.tensor_to_image(denorm_image)
                            ax.imshow(display_image, cmap='gray')
                        else:
                            ax.text(0.5, 0.5, 'Unknown\nFormat', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10)
                        
                        # 添加标题信息
                        ax.set_title(f'ID:{img_info["sample_id"]}\n'
                                   f'T:{img_info["true_label"]} P:{img_info["pred_label"]}', 
                                   fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'No Image', ha='center', va='center',
                               transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'Class {label} - Sample {sample_idx+1}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No More\nSamples', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'Class {label} - N/A', fontsize=10)
                
                ax.axis('off')
        
        plt.tight_layout()
        
        # 保存网格图
        grid_path = os.path.join(save_dir, f'{layer_name}_{method}_sample_grid.png')
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 样本网格已保存: {grid_path}")
    
    def create_detailed_sample_view(self, selected_indices, features_2d, labels, 
                                   layer_name, method, save_dir, max_samples=20):
        """创建选定样本的详细视图"""
        
        # 限制样本数量
        if len(selected_indices) > max_samples:
            selected_indices = selected_indices[:max_samples]
        
        n_samples = len(selected_indices)
        cols = min(5, n_samples)
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Detailed View - {layer_name} {method.upper()}\n'
                     f'Selected {n_samples} samples', fontsize=16)
        
        for i, idx in enumerate(selected_indices):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # 获取图像信息
            if idx < len(self.image_info):
                img_info = self.image_info[idx]
                
                # 显示图像
                image_tensor = img_info['image_tensor']
                
                if image_tensor.shape[0] == 3:
                    denorm_image = self.denormalize_image(image_tensor, 'rgb')
                    display_image = self.tensor_to_image(denorm_image)
                    ax.imshow(display_image)
                elif image_tensor.shape[0] == 1:
                    denorm_image = self.denormalize_image(image_tensor, 'grayscale')
                    display_image = self.tensor_to_image(denorm_image)
                    ax.imshow(display_image, cmap='gray')
                
                # 添加详细信息
                x, y = features_2d[idx]
                title = (f'Point {idx}\n'
                        f'ID: {img_info["sample_id"]}\n'
                        f'2D: ({x:.2f}, {y:.2f})\n'
                        f'True: {img_info["true_label"]}, Pred: {img_info["pred_label"]}')
                ax.set_title(title, fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Point {idx}\nNo Image Data', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存详细视图
        detail_path = os.path.join(save_dir, f'{layer_name}_{method}_selected_samples.png')
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 详细样本视图已保存: {detail_path}")
        return detail_path


def load_model_and_data(checkpoint_path, val_csv, data_root, batch_size=8):
    """加载模型和数据"""
    print("📥 加载模型和数据...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 导入模型类
    try:
        from Model_embodiment import EmbodiedCountingModel
    except ImportError:
        print("❌ 无法导入EmbodiedCountingModel，请确保Model_embodiment.py在Python路径中")
        raise
    
    # 确定图像模式
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    # 重建模型
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    model = EmbodiedCountingModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成 (图像模式: {image_mode}, 设备: {device})")
    
    # 创建数据加载器
    try:
        from DataLoader_embodiment import get_ball_counting_data_loaders
    except ImportError:
        print("❌ 无法导入get_ball_counting_data_loaders，请确保DataLoader_embodiment.py在Python路径中")
        raise
    
    _, val_loader, _ = get_ball_counting_data_loaders(
        train_csv_path=config['train_csv'],
        val_csv_path=val_csv,
        data_root=data_root,
        batch_size=batch_size,
        sequence_length=config['sequence_length'],
        normalize=config['normalize'],
        num_workers=2,
        image_mode=image_mode
    )
    
    print(f"✅ 数据加载器创建完成，验证集大小: {len(val_loader.dataset)}")
    
    return model, val_loader, device, config


def simple_interactive_analysis(checkpoint_path, val_csv, data_root, 
                               save_dir='./simple_analysis', 
                               max_samples=300, specific_layers=None):
    """简洁的交互式分析主函数"""
    
    print("🎯 开始简洁交互式分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建特征提取器
        extractor = SimpleFeatureExtractor(model, device)
        
        # 3. 确定要分析的层
        if specific_layers is None:
            print("🔍 自动检测关键层...")
            key_layers = extractor.auto_detect_key_layers()
            if not key_layers:
                print("❌ 未检测到任何层，使用默认层")
                key_layers = ['fusion']
        else:
            key_layers = specific_layers
        
        print(f"📋 准备分析的层: {key_layers}")
        
        # 4. 注册钩子并提取特征
        successful_layers = extractor.register_hooks(key_layers)
        
        if not successful_layers:
            print("❌ 没有成功注册任何钩子！")
            return None
        
        try:
            # 5. 提取特征
            print("🎯 提取特征...")
            data = extractor.extract_features(val_loader, max_samples)
            
            features = data['features']
            predictions = data['predictions']
            true_labels = data['labels']
            image_info = data['image_info']
            
            print(f"✅ 特征提取完成:")
            print(f"   样本数: {len(true_labels)}")
            print(f"   提取层: {list(features.keys())}")
            print(f"   图像信息数: {len(image_info)}")
            
            # 6. 创建可视化器
            visualizer = InteractiveVisualizer(image_info)
            
            # 7. 对每个层进行PCA和t-SNE分析
            print("📊 开始可视化分析...")
            
            analysis_results = {}
            
            for layer_name, layer_features in features.items():
                if layer_features is None:
                    continue
                    
                print(f"   分析 {layer_name} 层...")
                layer_results = {}
                
                # PCA分析
                try:
                    print(f"     执行PCA...")
                    pca_features = visualizer.reduce_dimensions(layer_features, 'pca')
                    
                    pca_plot_path = visualizer.create_interactive_plot(
                        pca_features, true_labels, layer_name, 'pca', save_dir
                    )
                    
                    layer_results['pca'] = {
                        'features_2d': pca_features,
                        'plot_path': pca_plot_path
                    }
                    
                except Exception as e:
                    print(f"     PCA分析失败: {e}")
                
                # t-SNE分析
                try:
                    print(f"     执行t-SNE...")
                    tsne_features = visualizer.reduce_dimensions(layer_features, 'tsne')
                    
                    tsne_plot_path = visualizer.create_interactive_plot(
                        tsne_features, true_labels, layer_name, 'tsne', save_dir
                    )
                    
                    layer_results['tsne'] = {
                        'features_2d': tsne_features,
                        'plot_path': tsne_plot_path
                    }
                    
                except Exception as e:
                    print(f"     t-SNE分析失败: {e}")
                
                analysis_results[layer_name] = layer_results
            
            # 8. 生成使用说明
            generate_usage_guide(analysis_results, save_dir, image_info)
            
            print(f"🎉 简洁分析完成！结果保存在: {save_dir}")
            
            return analysis_results
            
        finally:
            extractor.remove_hooks()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_usage_guide(analysis_results, save_dir, image_info):
    """生成使用说明文件"""
    
    guide_path = os.path.join(save_dir, 'README.md')
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("# 简洁交互式可视化分析结果\n\n")
        f.write("## 🎯 功能说明\n\n")
        f.write("本工具提供了简洁的PCA和t-SNE可视化，让你可以了解降维图上每个点对应的原始样本信息。\n\n")
        
        f.write("## 📁 文件说明\n\n")
        
        for layer_name, layer_results in analysis_results.items():
            f.write(f"### {layer_name} 层分析结果\n\n")
            
            for method in ['pca', 'tsne']:
                if method in layer_results:
                    f.write(f"- `{layer_name}_{method}_interactive.png`: {method.upper()}降维散点图\n")
                    f.write(f"- `{layer_name}_{method}_point_info.txt`: 每个点的详细信息\n")
                    f.write(f"- `{layer_name}_{method}_sample_grid.png`: 每个类别的代表性样本\n")
            f.write("\n")
        
        f.write("## 🔍 如何查看点的详细信息\n\n")
        f.write("1. **查看散点图**: 打开 `*_interactive.png` 文件查看降维结果\n")
        f.write("2. **查找点信息**: 打开对应的 `*_point_info.txt` 文件\n")
        f.write("3. **定位具体点**: 在文件中搜索你感兴趣的区域坐标或样本ID\n")
        f.write("4. **查看样本图像**: 参考 `*_sample_grid.png` 了解各类别的典型样本\n\n")
        
        f.write("## 📊 点信息文件格式\n\n")
        f.write("```\n")
        f.write("点索引 | 2D坐标 | 样本ID | 真实标签 | 预测标签\n")
        f.write("点  0 | ( -2.156,   1.423) |    sample_1 | 真实: 3 | 预测: 3\n")
        f.write("点  1 | (  0.892,  -0.756) |    sample_2 | 真实: 5 | 预测: 4\n")
        f.write("...\n")
        f.write("```\n\n")
        
        f.write("## 🎯 使用技巧\n\n")
        f.write("1. **查找异常点**: 在散点图中找到离群或错误分类的点\n")
        f.write("2. **查找坐标**: 记录异常点的大致坐标位置\n")
        f.write("3. **定位样本**: 在point_info.txt中搜索相近的坐标\n")
        f.write("4. **分析样本**: 根据样本ID进一步分析具体原因\n\n")
        
        f.write("## 📈 分析建议\n\n")
        f.write("- **PCA**: 显示线性主成分，适合理解特征的主要变化方向\n")
        f.write("- **t-SNE**: 显示非线性结构，适合发现聚类和局部邻域关系\n")
        f.write("- **对比分析**: 同时查看PCA和t-SNE结果，获得更全面的理解\n")
        f.write("- **多层分析**: 比较不同层的可视化结果，理解特征演化过程\n\n")
        
        f.write(f"## 📋 本次分析统计\n\n")
        f.write(f"- 总样本数: {len(image_info)}\n")
        f.write(f"- 分析层数: {len(analysis_results)}\n")
        f.write(f"- 生成文件数: {sum(len(lr) * 3 for lr in analysis_results.values())}\n")
        
    print(f"✅ 使用说明已保存: {guide_path}")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='简洁交互式可视化工具')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str, 
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 分析选项
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=300,
                       help='最大分析样本数')
    parser.add_argument('--layers', nargs='+', default=None,
                       help='指定要分析的层名称')
    
    # 其他选项
    parser.add_argument('--batch_size', type=int, default=8,
                       help='数据加载批次大小')
    
    args = parser.parse_args()
    
    # 设置默认保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./simple_analysis_{timestamp}'
    
    # 检查输入文件
    for path, name in [(args.checkpoint, '检查点文件'), 
                       (args.val_csv, '验证CSV文件'), 
                       (args.data_root, '数据根目录')]:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    print("🎯 简洁交互式可视化工具")
    print("="*50)
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print(f"最大样本数: {args.max_samples}")
    if args.layers:
        print(f"指定层: {args.layers}")
    print("="*50)
    
    start_time = time.time()
    
    try:
        results = simple_interactive_analysis(
            args.checkpoint, args.val_csv, args.data_root, 
            args.save_dir, args.max_samples, args.layers
        )
        
        if results:
            elapsed_time = time.time() - start_time
            print(f"\n🎉 分析完成！")
            print(f"⏱️ 总耗时: {elapsed_time:.2f} 秒")
            print(f"📁 结果保存在: {args.save_dir}")
            print(f"📖 查看 README.md 了解如何使用结果")
        else:
            print(f"\n❌ 分析失败")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    if len(sys.argv) == 1:
        print("🎯 简洁交互式可视化工具")
        print("="*50)
        print("功能: PCA和t-SNE降维可视化，可查看每个点的原始图像信息")
        print("优势: 简洁、专注、易用")
        print("\n使用方法:")
        print("python simple_interactive_viz.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
        print("\n基本示例:")
        print("python simple_interactive_viz.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data")
        print("\n高级示例:")
        print("python simple_interactive_viz.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --layers fusion lstm \\")
        print("    --max_samples 500 \\")
        print("    --save_dir ./my_viz")
        print("\n输出文件:")
        print("- *_pca_interactive.png     # PCA降维散点图")
        print("- *_tsne_interactive.png    # t-SNE降维散点图")
        print("- *_point_info.txt          # 每个点的详细信息")
        print("- *_sample_grid.png         # 每类的代表性样本")
        print("- README.md                 # 详细使用说明")
        print("\n💡 如何查看点的原始图像:")
        print("1. 在散点图中找到感兴趣的点")
        print("2. 记录该点的大致坐标")
        print("3. 在对应的 point_info.txt 中搜索相近坐标")
        print("4. 获得样本ID和标签信息")
        print("5. 参考 sample_grid.png 查看该类的典型样本")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷函数，供其他脚本调用
# =============================================================================

def quick_viz(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=200):
    """快速可视化接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./quick_viz_{timestamp}'
    
    return simple_interactive_analysis(
        checkpoint_path, val_csv, data_root, save_dir, max_samples
    )


def custom_viz(checkpoint_path, val_csv, data_root, layers, save_dir=None, max_samples=300):
    """自定义层可视化接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./custom_viz_{timestamp}'
    
    return simple_interactive_analysis(
        checkpoint_path, val_csv, data_root, save_dir, max_samples, layers
    )


# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

1. 命令行使用:
   python simple_interactive_viz.py \\
       --checkpoint ./best_model.pth \\
       --val_csv ./val.csv \\
       --data_root ./data \\
       --max_samples 300

2. 在Python脚本中使用:
   from simple_interactive_viz import quick_viz, custom_viz
   
   # 快速可视化
   results = quick_viz(
       checkpoint_path='./best_model.pth',
       val_csv='./val.csv', 
       data_root='./data'
   )
   
   # 自定义层可视化
   results = custom_viz(
       checkpoint_path='./best_model.pth',
       val_csv='./val.csv',
       data_root='./data',
       layers=['fusion', 'lstm']
   )

3. 查看结果:
   - 打开生成的 *_interactive.png 查看散点图
   - 打开对应的 *_point_info.txt 查看点的详细信息
   - 参考 README.md 了解具体使用方法

4. 定位特定点的图像:
   - 在散点图中找到感兴趣的区域或异常点
   - 记录该点的坐标 (x, y)
   - 在 point_info.txt 中搜索相近的坐标
   - 获得对应的样本ID和标签信息
   - 在 sample_grid.png 中查看该类的典型样本
"""