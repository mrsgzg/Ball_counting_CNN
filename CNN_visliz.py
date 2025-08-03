"""
Single Image CNN Model Feature Analysis Tool
针对SingleImageClassifier的特征降维可视化分析工具
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
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

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SingleImageFeatureExtractor:
    """Single Image Model特征提取器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.features = {}
        self.hooks = []
        
    def inspect_model_structure(self):
        """检查SingleImageClassifier模型结构"""
        print("=== Single Image Model 结构检查 ===")
        available_layers = []
        
        def print_module_structure(module, prefix="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                print(f"{'  ' * current_depth}{full_name}: {child.__class__.__name__}")
                available_layers.append(full_name)
                
                # 递归打印子模块
                if len(list(child.children())) > 0:
                    print_module_structure(child, full_name, max_depth, current_depth + 1)
        
        print_module_structure(self.model)
        print(f"\n总共发现 {len(available_layers)} 个层")
        
        return available_layers
    
    def auto_detect_key_layers(self):
        """自动检测SingleImageClassifier的关键层"""
        available_layers = self.inspect_model_structure()
        
        # SingleImageClassifier的关键层
        target_layers = [
            'visual_encoder',      # 视觉编码器
            'visual_encoder.cnn',  # CNN特征提取
            'spatial_attention',   # 空间注意力
            'classifier'           # 分类器
        ]
        
        detected_layers = []
        
        # 精确匹配
        for target in target_layers:
            if target in available_layers:
                detected_layers.append(target)
                print(f"✅ 检测到关键层: {target}")
        
        # 模糊匹配CNN子层
        cnn_layers = []
        for layer_name in available_layers:
            if 'visual_encoder.cnn' in layer_name and any(x in layer_name.lower() for x in ['conv', 'relu', 'pool']):
                cnn_layers.append(layer_name)
        
        if cnn_layers:
            # 选择一些代表性的CNN层
            selected_cnn = cnn_layers[::max(1, len(cnn_layers)//3)][:3]  # 选择3个具有代表性的层
            detected_layers.extend(selected_cnn)
            print(f"🔍 检测到CNN子层: {selected_cnn}")
        
        print(f"\n建议提取的层: {detected_layers}")
        return detected_layers
        
    def register_hooks(self, layers_to_extract):
        """注册钩子函数来提取中间层特征"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    # 如果输出是元组，取第一个元素
                    self.features[name] = output[0].detach().cpu()
                else:
                    self.features[name] = output.detach().cpu()
            return hook
        
        successful_hooks = []
        failed_hooks = []
        
        # 注册钩子
        for layer_name in layers_to_extract:
            try:
                # 通过点号分割的路径访问嵌套模块
                module = self.model
                for part in layer_name.split('.'):
                    module = getattr(module, part)
                
                handle = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(handle)
                successful_hooks.append(layer_name)
                print(f"✅ 成功注册钩子: {layer_name}")
                
            except AttributeError as e:
                failed_hooks.append(layer_name)
                print(f"❌ 注册钩子失败: {layer_name} - {e}")
        
        print(f"\n钩子注册结果: 成功 {len(successful_hooks)}, 失败 {len(failed_hooks)}")
        
        if not successful_hooks and failed_hooks:
            print("⚠️ 没有成功注册任何钩子，尝试自动检测...")
            auto_layers = self.auto_detect_key_layers()
            if auto_layers:
                print(f"使用自动检测的层: {auto_layers}")
                return self.register_hooks(auto_layers[:4])  # 只取前4个避免太多
        
        return successful_hooks
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        
    def _process_feature_tensor(self, feature_tensor):
        """处理不同形状的特征张量"""
        if len(feature_tensor.shape) == 4:  # [batch, channel, h, w]
            # 全局平均池化
            pooled = feature_tensor.mean(dim=(-2, -1))  # [batch, channel]
            return pooled.cpu().numpy()
        elif len(feature_tensor.shape) == 3:  # [batch, seq, dim] 或其他3D
            # 如果有序列维度，取最后一个
            if feature_tensor.shape[1] > feature_tensor.shape[2]:
                # 假设是 [batch, spatial, channel]
                pooled = feature_tensor.mean(dim=1)
            else:
                # 假设是 [batch, channel, spatial]
                pooled = feature_tensor.mean(dim=-1)
            return pooled.cpu().numpy()
        elif len(feature_tensor.shape) == 2:  # [batch, dim]
            return feature_tensor.cpu().numpy()
        else:
            # 其他情况，展平
            return feature_tensor.view(feature_tensor.shape[0], -1).cpu().numpy()
    
    def extract_features(self, data_loader, max_samples=1000):
        """提取特征并收集预测结果 - 适配SingleImageDataset"""
        all_features = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_sample_ids = []
        all_attention_weights = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取单图像特征"):
                if sample_count >= max_samples:
                    break
                
                # 单图像数据格式
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                sample_ids = batch['sample_id']
                
                # 计算实际处理的样本数
                remaining_samples = max_samples - sample_count
                actual_batch_size = min(len(labels), remaining_samples)
                
                # 截断批次（如果需要）
                if actual_batch_size < len(labels):
                    images = images[:actual_batch_size]
                    labels = labels[:actual_batch_size]
                    sample_ids = sample_ids[:actual_batch_size]
                
                # 清空特征字典
                self.features = {}
                
                # 前向传播
                if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                    # 如果模型支持返回注意力权重
                    outputs = self.model(images, return_attention=True)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        if len(outputs) > 1:
                            attention_weights = outputs[1]
                            if attention_weights is not None:
                                all_attention_weights.extend(attention_weights.cpu().numpy())
                    else:
                        logits = outputs
                else:
                    logits = self.model(images)
                
                # 收集预测结果
                pred_labels = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_predictions.extend(pred_labels)
                all_labels.extend(labels)
                all_sample_ids.extend(sample_ids)
                
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
            'sample_ids': all_sample_ids
        }
        
        # 添加注意力权重（如果有）
        if all_attention_weights:
            result['attention_weights'] = np.array(all_attention_weights)
        
        # 打印调试信息
        print(f"单图像特征提取完成:")
        print(f"  实际样本数: {len(result['labels'])}")
        print(f"  真实标签范围: {result['labels'].min()} - {result['labels'].max()}")
        print(f"  预测标签范围: {result['predictions'].min()} - {result['predictions'].max()}")
        print(f"  真实标签唯一值: {sorted(np.unique(result['labels']))}")
        print(f"  预测标签唯一值: {sorted(np.unique(result['predictions']))}")
        print(f"  提取的特征层: {list(final_features.keys())}")
        
        return result


class SingleImageVisualizationEngine:
    """Single Image Model可视化引擎"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def reduce_dimensions(self, features, method='tsne', n_components=2):
        """降维"""
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(features)//4))
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        reduced_features = reducer.fit_transform(features)
        return reduced_features
    
    def plot_scatter(self, features_2d, labels, title, save_path=None, alpha=0.7):
        """绘制散点图"""
        plt.figure(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Count {label}', alpha=alpha, s=50)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_clustering_comparison(self, features_2d, true_labels, cluster_results, save_dir=None):
        """比较不同聚类方法的结果"""
        n_methods = len(cluster_results) + 1  # +1 for true labels
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        # 绘制真实标签
        unique_labels = np.unique(true_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = true_labels == label
            axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=f'Count {label}', alpha=0.7, s=30)
        axes[0].set_title('True Labels')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 绘制聚类结果
        for idx, (method, result) in enumerate(cluster_results.items()):
            cluster_labels = result['labels']
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                if cluster == -1:  # DBSCAN的噪声点
                    axes[idx+1].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                      c='black', label='Noise', alpha=0.5, s=30, marker='x')
                else:
                    axes[idx+1].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                      c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=30)
            
            title = f'{method.upper()}\nSilhouette: {result["silhouette_score"]:.3f}'
            axes[idx+1].set_title(title)
            axes[idx+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'clustering_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights, save_path=None):
        """可视化注意力权重热力图"""
        if attention_weights is None or len(attention_weights) == 0:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 处理注意力权重
        if len(attention_weights.shape) == 4:  # [batch, heads, H, W]
            # 取平均
            avg_attention = np.mean(attention_weights, axis=(0, 1))  # [H, W]
            sns.heatmap(avg_attention, cmap='viridis', cbar=True, square=True)
            plt.title('Average Spatial Attention Map')
            plt.axis('off')
        elif len(attention_weights.shape) == 3:  # [batch, heads, spatial]
            # 取平均并重塑
            avg_attention = np.mean(attention_weights, axis=0)  # [heads, spatial]
            sns.heatmap(avg_attention, cmap='viridis', cbar=True)
            plt.title('Multi-Head Attention Weights')
            plt.xlabel('Spatial Location')
            plt.ylabel('Attention Head')
        elif len(attention_weights.shape) == 2:  # [batch, spatial]
            # 取平均
            avg_attention = np.mean(attention_weights, axis=0)
            spatial_size = len(avg_attention)
            grid_size = int(np.sqrt(spatial_size))
            
            if grid_size * grid_size == spatial_size:
                # 可以重塑为正方形
                attention_map = avg_attention.reshape(grid_size, grid_size)
                sns.heatmap(attention_map, cmap='viridis', cbar=True, square=True)
                plt.title('Average Attention Map')
                plt.axis('off')
            else:
                # 绘制1D图
                plt.bar(range(len(avg_attention)), avg_attention)
                plt.title('Attention Weights')
                plt.xlabel('Spatial Location')
                plt.ylabel('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_hierarchy(self, features_dict, labels, save_path=None):
        """可视化特征层次结构"""
        n_layers = len(features_dict)
        if n_layers == 0:
            return
        
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(6 * ((n_layers + 1) // 2), 12))
        if n_layers == 1:
            axes = np.array([axes]).flatten()
        elif (n_layers + 1) // 2 == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for idx, (layer_name, features) in enumerate(features_dict.items()):
            if idx >= len(axes):
                break
                
            # t-SNE降维
            features_2d = self.reduce_dimensions(features, 'tsne')
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[idx].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                c=[colors[i]], label=f'Count {label}', alpha=0.7, s=30)
            
            axes[idx].set_title(f'{layer_name}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            if idx == 0:
                axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 隐藏多余的子图
        for idx in range(len(features_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_layer_statistics(self, features_dict, save_path=None):
        """绘制各层特征统计信息"""
        n_layers = len(features_dict)
        if n_layers == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        layer_names = list(features_dict.keys())
        
        # 1. 特征维度
        feature_dims = [features.shape[1] for features in features_dict.values()]
        axes[0].bar(range(len(layer_names)), feature_dims, color='skyblue')
        axes[0].set_title('Feature Dimensions by Layer')
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0].set_ylabel('Dimension')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 特征方差
        feature_vars = [np.var(features) for features in features_dict.values()]
        axes[1].bar(range(len(layer_names)), feature_vars, color='lightcoral')
        axes[1].set_title('Feature Variance by Layer')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[1].set_ylabel('Variance')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 特征范围
        feature_ranges = [(np.max(features) - np.min(features)) for features in features_dict.values()]
        axes[2].bar(range(len(layer_names)), feature_ranges, color='lightgreen')
        axes[2].set_title('Feature Range by Layer')
        axes[2].set_xticks(range(len(layer_names)))
        axes[2].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[2].set_ylabel('Range')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 有效秩（特征复杂度）
        effective_ranks = []
        for features in features_dict.values():
            try:
                rank = np.linalg.matrix_rank(features)
                effective_ranks.append(rank)
            except:
                effective_ranks.append(0)
        
        axes[3].bar(range(len(layer_names)), effective_ranks, color='orange')
        axes[3].set_title('Effective Rank by Layer')
        axes[3].set_xticks(range(len(layer_names)))
        axes[3].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[3].set_ylabel('Rank')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ClusterAnalyzer:
    """聚类分析器"""
    
    def __init__(self):
        self.results = {}
    
    def perform_clustering(self, features, methods=['kmeans', 'dbscan'], n_clusters=None, true_labels=None):
        """执行多种聚类算法"""
        results = {}
        
        # 如果没有指定聚类数，从真实标签推断
        if n_clusters is None and true_labels is not None:
            n_clusters = len(np.unique(true_labels))
        elif n_clusters is None:
            n_clusters = min(10, int(np.sqrt(len(features))))
        
        for method in methods:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(features)
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(features, cluster_labels)
                else:
                    silhouette = -1
                results[method] = {
                    'labels': cluster_labels,
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters
                }
                
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(features)
                n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters_found > 1:
                    mask = cluster_labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(features[mask], cluster_labels[mask])
                    else:
                        silhouette = -1
                else:
                    silhouette = -1
                    
                results[method] = {
                    'labels': cluster_labels,
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters_found
                }
        
        return results


def load_single_image_model_and_data(checkpoint_path, val_csv, data_root, batch_size=16):
    """加载单图像模型和数据"""
    print("📥 加载单图像模型和数据...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 导入模型类
    try:
        from Model_single_image import create_single_image_model
        from DataLoader_single_image import get_single_image_data_loaders
    except ImportError:
        print("❌ 无法导入单图像模型相关模块")
        raise
    
    # 重建模型
    model = create_single_image_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    image_mode = config.get('image_mode', 'rgb')
    print(f"✅ 单图像模型加载完成 (图像模式: {image_mode}, 设备: {device})")
    
    # 创建数据加载器
    _, val_loader = get_single_image_data_loaders(
        train_csv_path=config['train_csv'],
        val_csv_path=val_csv,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=2,
        image_mode=image_mode,
        normalize_images=True
    )
    
    print(f"✅ 数据加载器创建完成，验证集大小: {len(val_loader.dataset)}")
    
    return model, val_loader, device, config


def analyze_single_image_model(checkpoint_path, val_csv, data_root, 
                              save_dir='./single_image_analysis', 
                              max_samples=500, specific_layers=None):
    """分析单图像CNN模型"""
    
    print("🖼️ 开始单图像CNN模型特征分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_single_image_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建特征提取器
        feature_extractor = SingleImageFeatureExtractor(model, device)
        
        # 3. 确定要分析的层
        if specific_layers is None:
            print("🔍 自动检测关键层...")
            key_layers = feature_extractor.auto_detect_key_layers()
            if not key_layers:
                key_layers = ['visual_encoder', 'classifier']
        else:
            key_layers = specific_layers
        
        print(f"📋 准备分析的层: {key_layers}")
        
        # 4. 注册钩子并提取特征
        successful_layers = feature_extractor.register_hooks(key_layers)
        
        if not successful_layers:
            print("❌ 没有成功注册任何钩子！")
            return None
        
        try:
            # 5. 提取特征
            print("🎯 提取特征...")
            data = feature_extractor.extract_features(val_loader, max_samples)
            
            features = data['features']
            predictions = data['predictions']
            true_labels = data['labels']
            attention_weights = data.get('attention_weights', None)
            
            print(f"✅ 特征提取完成:")
            print(f"   样本数: {len(true_labels)}")
            print(f"   提取层: {list(features.keys())}")
            print(f"   注意力权重: {'有' if attention_weights is not None else '无'}")
            
            # 6. 创建可视化引擎
            visualizer = SingleImageVisualizationEngine()
            cluster_analyzer = ClusterAnalyzer()
            
            # 7. 特征层次结构可视化
            print("🎨 生成特征层次结构可视化...")
            visualizer.plot_feature_hierarchy(
                features, true_labels,
                save_path=os.path.join(save_dir, 'feature_hierarchy.png')
            )
            
            # 8. 各层统计信息
            print("📊 生成各层统计信息...")
            visualizer.plot_layer_statistics(
                features,
                save_path=os.path.join(save_dir, 'layer_statistics.png')
            )
            
            # 9. 注意力机制可视化（如果有）
            if attention_weights is not None:
                print("👁️ 可视化注意力机制...")
                visualizer.plot_attention_heatmap(
                    attention_weights,
                    save_path=os.path.join(save_dir, 'attention_heatmap.png')
                )
            
            # 10. 各层降维和聚类分析
            print("🔬 执行降维和聚类分析...")
            
            analysis_results = {}
            
            for layer_name, layer_features in features.items():
                if layer_features is None:
                    continue
                    
                print(f"   分析 {layer_name} 层...")
                layer_results = {}
                
                # 多种降维方法
                for dim_method in ['tsne', 'pca', 'umap']:
                    try:
                        # 降维
                        features_2d = visualizer.reduce_dimensions(layer_features, dim_method)
                        
                        # 可视化真实标签
                        visualizer.plot_scatter(
                            features_2d, true_labels,
                            f'{layer_name} - True Labels ({dim_method.upper()})',
                            save_path=os.path.join(save_dir, f'{layer_name}_{dim_method}_true.png')
                        )
                        
                        # 可视化预测结果
                        visualizer.plot_scatter(
                            features_2d, predictions,
                            f'{layer_name} - Predictions ({dim_method.upper()})',
                            save_path=os.path.join(save_dir, f'{layer_name}_{dim_method}_pred.png')
                        )
                        
                        # 聚类分析
                        cluster_results = cluster_analyzer.perform_clustering(
                            layer_features, methods=['kmeans', 'dbscan'], true_labels=true_labels
                        )
                        
                        # 聚类对比可视化
                        visualizer.plot_clustering_comparison(
                            features_2d, true_labels, cluster_results,
                            save_dir=os.path.join(save_dir, f'{layer_name}_{dim_method}_clustering')
                        )
                        
                        layer_results[dim_method] = {
                            'clustering': cluster_results,
                            'features_2d': features_2d
                        }
                        
                    except Exception as e:
                        print(f"     {dim_method} 分析失败: {e}")
                        continue
                
                analysis_results[layer_name] = layer_results
            
            # 11. 生成分析报告
            print("📝 生成分析报告...")
            generate_single_image_report(
                analysis_results, features, true_labels, predictions, 
                attention_weights, config, save_dir
            )
            
            print(f"🎉 单图像模型分析完成！结果保存在: {save_dir}")
            return analysis_results
            
        finally:
            feature_extractor.remove_hooks()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_single_image_report(analysis_results, features, true_labels, predictions, 
                               attention_weights, config, save_dir):
    """生成单图像模型分析报告"""
    
    from sklearn.metrics import accuracy_score
    
    # 基础统计
    accuracy = accuracy_score(true_labels, predictions)
    unique_true = len(np.unique(true_labels))
    unique_pred = len(np.unique(predictions))
    
    # 层级特征复杂度
    layer_complexity = {}
    for layer_name, layer_features in features.items():
        if layer_features is not None:
            layer_complexity[layer_name] = {
                'feature_dim': int(layer_features.shape[1]),
                'feature_variance': float(np.var(layer_features)),
                'feature_range': [float(np.min(layer_features)), float(np.max(layer_features))],
                'feature_mean': float(np.mean(layer_features)),
                'feature_std': float(np.std(layer_features)),
                'effective_rank': float(np.linalg.matrix_rank(layer_features))
            }
    
    # 注意力分析
    attention_analysis = {}
    if attention_weights is not None:
        attention_analysis = {
            'mean_attention': float(np.mean(attention_weights)),
            'attention_std': float(np.std(attention_weights)),
            'attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1).mean()) if attention_weights.ndim > 1 else 0,
            'attention_sparsity': float(np.sum(attention_weights < 0.1) / attention_weights.size)
        }
    
    # 聚类质量总结
    clustering_summary = {}
    for layer_name, layer_results in analysis_results.items():
        layer_clustering = {}
        for dim_method, method_results in layer_results.items():
            if 'clustering' in method_results:
                clustering_results = method_results['clustering']
                best_silhouette = max([r['silhouette_score'] for r in clustering_results.values() 
                                     if r['silhouette_score'] != -1], default=0)
                layer_clustering[dim_method] = {
                    'best_silhouette': float(best_silhouette),
                    'kmeans_silhouette': float(clustering_results.get('kmeans', {}).get('silhouette_score', 0)),
                    'dbscan_silhouette': float(clustering_results.get('dbscan', {}).get('silhouette_score', 0))
                }
        clustering_summary[layer_name] = layer_clustering
    
    # 生成完整报告
    report = {
        'analysis_type': 'Single Image CNN Model Feature Analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_config': {
            'image_mode': config.get('image_mode', 'rgb'),
            'use_attention': config.get('use_attention', True),
            'num_classes': config.get('num_classes', 10)
        },
        'summary': {
            'total_samples': int(len(true_labels)),
            'overall_accuracy': float(accuracy),
            'unique_true_labels': int(unique_true),
            'unique_predictions': int(unique_pred),
            'analyzed_layers': list(features.keys())
        },
        'layer_complexity': layer_complexity,
        'attention_analysis': attention_analysis,
        'clustering_summary': clustering_summary,
        'recommendations': generate_single_image_recommendations(analysis_results, features, accuracy, config)
    }
    
    # 保存JSON报告
    try:
        with open(os.path.join(save_dir, 'single_image_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ JSON报告已保存")
    except Exception as e:
        print(f"⚠️ JSON报告保存失败: {e}")
    
    # 生成可读报告
    try:
        with open(os.path.join(save_dir, 'single_image_analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== 单图像CNN模型特征分析报告 ===\n\n")
            f.write(f"分析时间: {report['timestamp']}\n")
            f.write(f"总样本数: {report['summary']['total_samples']}\n")
            f.write(f"整体准确率: {report['summary']['overall_accuracy']:.4f}\n")
            f.write(f"分析层数: {len(report['summary']['analyzed_layers'])}\n\n")
            
            # 模型配置
            f.write("=== 模型配置 ===\n")
            for key, value in report['model_config'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 层级复杂度
            f.write("=== 层级特征复杂度 ===\n")
            for layer_name, complexity in layer_complexity.items():
                f.write(f"{layer_name}:\n")
                f.write(f"  特征维度: {complexity['feature_dim']}\n")
                f.write(f"  特征方差: {complexity['feature_variance']:.4f}\n")
                f.write(f"  特征均值: {complexity['feature_mean']:.4f}\n")
                f.write(f"  特征标准差: {complexity['feature_std']:.4f}\n")
                f.write(f"  有效秩: {complexity['effective_rank']:.1f}\n")
                f.write(f"  数值范围: [{complexity['feature_range'][0]:.4f}, {complexity['feature_range'][1]:.4f}]\n")
            f.write("\n")
            
            # 注意力分析
            if attention_analysis:
                f.write("=== 注意力机制分析 ===\n")
                f.write(f"平均注意力强度: {attention_analysis['mean_attention']:.4f}\n")
                f.write(f"注意力标准差: {attention_analysis['attention_std']:.4f}\n")
                if attention_analysis['attention_entropy'] > 0:
                    f.write(f"注意力熵: {attention_analysis['attention_entropy']:.4f}\n")
                f.write(f"注意力稀疏性: {attention_analysis['attention_sparsity']:.4f}\n\n")
            
            # 聚类质量总结
            f.write("=== 聚类质量总结 ===\n")
            for layer_name, clustering in clustering_summary.items():
                f.write(f"{layer_name}:\n")
                for dim_method, scores in clustering.items():
                    f.write(f"  {dim_method}: 最佳轮廓系数={scores['best_silhouette']:.3f}\n")
                    f.write(f"    K-means: {scores['kmeans_silhouette']:.3f}\n")
                    f.write(f"    DBSCAN: {scores['dbscan_silhouette']:.3f}\n")
            f.write("\n")
            
            # 建议
            f.write("=== 分析建议 ===\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"✅ 文本报告已保存")
        
    except Exception as e:
        print(f"⚠️ 文本报告保存失败: {e}")


def generate_single_image_recommendations(analysis_results, features, accuracy, config):
    """基于分析结果生成建议"""
    recommendations = []
    
    # 准确率建议
    if accuracy < 0.6:
        recommendations.append("模型准确率较低，建议检查数据质量、增加训练轮数或调整学习率")
    elif accuracy > 0.95:
        recommendations.append("模型准确率很高，可能存在过拟合，建议在测试集上验证")
    elif accuracy > 0.85:
        recommendations.append("模型准确率良好，可以考虑在更复杂的任务上测试")
    
    # 特征维度建议
    high_dim_layers = [name for name, feats in features.items() 
                      if feats is not None and feats.shape[1] > 1024]
    if high_dim_layers:
        recommendations.append(f"层 {high_dim_layers} 特征维度很高，可考虑降维或正则化防止过拟合")
    
    low_dim_layers = [name for name, feats in features.items() 
                     if feats is not None and feats.shape[1] < 64]
    if low_dim_layers:
        recommendations.append(f"层 {low_dim_layers} 特征维度较低，可能限制了表达能力")
    
    # 特征复杂度建议
    low_rank_layers = []
    for layer_name, feats in features.items():
        if feats is not None:
            rank = np.linalg.matrix_rank(feats)
            dim = feats.shape[1]
            if rank < dim * 0.5:  # 有效秩小于维度的50%
                low_rank_layers.append(layer_name)
    
    if low_rank_layers:
        recommendations.append(f"层 {low_rank_layers} 的有效秩较低，特征可能存在冗余")
    
    # 聚类质量建议
    poor_clustering_layers = []
    for layer_name, layer_results in analysis_results.items():
        avg_silhouette = []
        for dim_method, method_results in layer_results.items():
            if 'clustering' in method_results:
                silhouettes = [r['silhouette_score'] for r in method_results['clustering'].values() 
                             if r['silhouette_score'] != -1]
                if silhouettes:
                    avg_silhouette.extend(silhouettes)
        
        if avg_silhouette and np.mean(avg_silhouette) < 0.2:
            poor_clustering_layers.append(layer_name)
    
    if poor_clustering_layers:
        recommendations.append(f"层 {poor_clustering_layers} 的聚类质量较差，不同类别的特征分离度不够")
    
    # 模型架构建议
    if not config.get('use_attention', True):
        recommendations.append("模型未使用注意力机制，建议尝试添加注意力模块提升性能")
    
    return recommendations


def inspect_single_image_model_structure(checkpoint_path):
    """检查单图像模型结构"""
    print("🔬 检查单图像模型结构...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        from Model_single_image import create_single_image_model
        
        model = create_single_image_model(config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        feature_extractor = SingleImageFeatureExtractor(model, device)
        available_layers = feature_extractor.inspect_model_structure()
        auto_layers = feature_extractor.auto_detect_key_layers()
        
        print(f"\n✨ 自动检测的关键层: {auto_layers}")
        print(f"\n📋 推荐的分析策略:")
        print("  --mode quick --max_samples 100              # 快速分析")
        print("  --mode full --max_samples 500               # 完整分析") 
        print("  --layers visual_encoder classifier --max_samples 300  # 自定义层分析")
        
        return available_layers, auto_layers
        
    except Exception as e:
        print(f"❌ 模型结构检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='单图像CNN模型特征分析工具')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='单图像模型检查点路径')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 分析选项
    parser.add_argument('--mode', type=str, default='full',
                       choices=['inspect', 'quick', 'full'],
                       help='分析模式')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=1100,
                       help='最大分析样本数')
    parser.add_argument('--layers', nargs='+', default=None,
                       help='指定要分析的层名称')
    
    # 其他选项
    parser.add_argument('--batch_size', type=int, default=32,
                       help='数据加载批次大小')
    
    args = parser.parse_args()
    
    # 设置默认保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./single_image_analysis_{args.mode}_{timestamp}'
    
    # 检查输入文件
    for path, name in [(args.checkpoint, '检查点文件'), 
                       (args.val_csv, '验证CSV文件'), 
                       (args.data_root, '数据根目录')]:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    print("🖼️ 单图像CNN模型特征分析工具")
    print("="*50)
    print(f"模式: {args.mode}")
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print("="*50)
    
    start_time = time.time()
    
    try:
        if args.mode == 'inspect':
            print("🔬 检查模型结构...")
            inspect_single_image_model_structure(args.checkpoint)
        
        elif args.mode == 'quick':
            print("⚡ 快速分析...")
            results = analyze_single_image_model(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, max_samples=min(100, args.max_samples),
                specific_layers=args.layers
            )
        
        elif args.mode == 'full':
            print("🖼️ 完整分析...")
            results = analyze_single_image_model(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, args.max_samples, args.layers
            )
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 分析完成！")
        print(f"⏱️ 总耗时: {elapsed_time:.2f} 秒")
        print(f"📁 结果保存在: {args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    if len(sys.argv) == 1:
        print("🖼️ 单图像CNN模型特征分析工具")
        print("="*50)
        print("使用方法:")
        print("python single_image_analysis.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
        print("\n可用模式:")
        print("  --mode inspect      # 检查模型结构")
        print("  --mode quick        # 快速分析（少量样本）")
        print("  --mode full         # 完整分析（推荐）")
        print("\n基本示例:")
        print("python single_image_analysis.py \\")
        print("    --checkpoint ./best_single_image_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode full")
        print("\n高级示例:")
        print("python single_image_analysis.py \\")
        print("    --checkpoint ./best_single_image_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode full \\")
        print("    --layers visual_encoder spatial_attention classifier \\")
        print("    --max_samples 1000 \\")
        print("    --save_dir ./my_single_image_analysis")
        print("\n📋 推荐工作流:")
        print("1. 首先运行: --mode inspect    # 查看模型结构")
        print("2. 然后运行: --mode quick      # 快速验证")
        print("3. 最后运行: --mode full       # 完整分析")
        print("\n💡 提示:")
        print("- 这个工具专门针对SingleImageClassifier模型")
        print("- 会自动提取CNN各层、注意力层、分类器的特征")
        print("- 生成t-SNE/PCA/UMAP降维可视化")
        print("- 包含聚类分析和特征统计")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷函数，供其他脚本调用
# =============================================================================

def quick_single_image_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=100):
    """快速单图像分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./quick_single_image_analysis_{timestamp}'
    
    return analyze_single_image_model(
        checkpoint_path, val_csv, data_root, save_dir, max_samples
    )


def full_single_image_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=500, layers=None):
    """完整单图像分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./full_single_image_analysis_{timestamp}'
    
    return analyze_single_image_model(
        checkpoint_path, val_csv, data_root, save_dir, max_samples, layers
    )


# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

1. 命令行使用:
   python single_image_analysis.py \\
       --checkpoint ./best_single_image_model.pth \\
       --val_csv ./val.csv \\
       --data_root ./data \\
       --mode full \\
       --max_samples 500

2. 在Python脚本中使用:
   from single_image_analysis import quick_single_image_analysis, full_single_image_analysis
   
   # 快速分析
   results = quick_single_image_analysis(
       checkpoint_path='./best_single_image_model.pth',
       val_csv='./val.csv', 
       data_root='./data'
   )
   
   # 完整分析
   results = full_single_image_analysis(
       checkpoint_path='./best_single_image_model.pth',
       val_csv='./val.csv',
       data_root='./data',
       max_samples=1000,
       layers=['visual_encoder', 'spatial_attention', 'classifier']
   )

3. 对比具身模型和单图像模型:
   # 分别运行两个分析工具
   embodied_results = full_analysis('./embodied_model.pth', ...)
   single_results = full_single_image_analysis('./single_model.pth', ...)
   
   # 手动对比分析结果
"""