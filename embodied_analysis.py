"""
完整的具身计数模型分析工具
支持多种特征提取、聚类分析和可视化方法
专门针对EmbodiedCountingModel优化
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
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, accuracy_score
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


class FeatureExtractor:
    """特征提取器 - 适配EmbodiedCountingModel的新数据格式"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.features = {}
        self.hooks = []
        
    def inspect_model_structure(self):
        """检查模型结构，返回所有可用的层名称"""
        print("=== 模型结构检查 ===")
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
        """自动检测关键层 - 针对EmbodiedCountingModel优化"""
        available_layers = self.inspect_model_structure()
        
        # 精确的层名称匹配
        target_layers = [
            'fusion',              # 多模态融合层
            'lstm',                # 时序处理层
            'counting_decoder',    # 计数解码层
            'visual_encoder',      # 视觉编码层
            'embodiment_encoder'   # 具身编码层
        ]
        
        detected_layers = []
        
        # 精确匹配
        for target in target_layers:
            if target in available_layers:
                detected_layers.append(target)
                print(f"✅ 检测到关键层: {target}")
        
        # 如果没找到，尝试子模块匹配
        if not detected_layers:
            print("🔍 尝试模糊匹配...")
            patterns = {
                'fusion': ['fusion', 'multimodal', 'cross_modal'],
                'lstm': ['lstm', 'rnn', 'gru'],
                'counting': ['counting', 'decoder', 'classifier', 'head'],
                'visual': ['visual', 'cnn', 'encoder'],
                'embodiment': ['embodiment', 'joint', 'pose']
            }
            
            for category, pattern_list in patterns.items():
                for layer_name in available_layers:
                    for pattern in pattern_list:
                        if pattern.lower() in layer_name.lower():
                            detected_layers.append(layer_name)
                            print(f"🔍 模糊匹配到: {layer_name} (类别: {category})")
                            break
                    if layer_name in detected_layers:
                        break
        
        print(f"\n建议提取的层: {detected_layers}")
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
                return self.register_hooks(auto_layers[:3])  # 只取前3个避免太多
        
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
        """提取特征并收集预测结果 - 适配新的数据格式"""
        all_features = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_sample_ids = []
        all_attention_weights = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                if sample_count >= max_samples:
                    break
                
                # 适配新的数据格式 - 从batch['sequence_data']中提取
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
                
                # 清空特征字典
                self.features = {}
                
                # 前向传播 - 使用新的数据格式
                outputs = self.model(
                    sequence_data=sequence_data,
                    use_teacher_forcing=False,
                    return_attention=True  # 返回注意力权重
                )
                
                # 收集预测结果
                count_logits = outputs['counts']  # [batch, seq_len, 11]
                pred_labels = torch.argmax(count_logits, dim=-1)  # [batch, seq_len]
                final_pred = pred_labels[:, -1].cpu().numpy()  # 最终时刻的预测
                
                all_predictions.extend(final_pred)
                all_labels.extend(labels)
                all_sample_ids.extend(sample_ids)
                
                # 收集注意力权重（如果可用）
                if 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights']  # [batch, seq_len, spatial_size]
                    # 取最后一个时刻的注意力权重
                    final_attention = attention_weights[:, -1, :].cpu().numpy()
                    all_attention_weights.extend(final_attention)
                
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
        print(f"特征提取完成:")
        print(f"  实际样本数: {len(result['labels'])}")
        print(f"  真实标签范围: {result['labels'].min()} - {result['labels'].max()}")
        print(f"  预测标签范围: {result['predictions'].min()} - {result['predictions'].max()}")
        print(f"  真实标签唯一值: {sorted(np.unique(result['labels']))}")
        print(f"  预测标签唯一值: {sorted(np.unique(result['predictions']))}")
        print(f"  提取的特征层: {list(final_features.keys())}")
        
        return result


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
            n_clusters = min(10, int(np.sqrt(len(features))))  # 默认启发式
        
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
                    # 只对非噪声点计算轮廓系数
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
    
    def evaluate_clustering(self, cluster_labels, true_labels):
        """评估聚类质量"""
        try:
            # 计算调整兰德指数
            ari = adjusted_rand_score(true_labels, cluster_labels)
            
            # 计算聚类纯度
            def purity_score(y_true, y_pred):
                # 处理负标签（DBSCAN的噪声点）
                y_pred_clean = y_pred.copy()
                noise_mask = y_pred_clean == -1
                if np.any(noise_mask):
                    # 将噪声点分配给自己的类别
                    max_cluster = np.max(y_pred_clean[~noise_mask]) if np.any(~noise_mask) else 0
                    y_pred_clean[noise_mask] = np.arange(max_cluster + 1, max_cluster + 1 + np.sum(noise_mask))
                
                unique_true = np.unique(y_true)
                unique_pred = np.unique(y_pred_clean)
                
                contingency_matrix = np.zeros((len(unique_true), len(unique_pred)))
                
                for i, true_label in enumerate(unique_true):
                    for j, pred_label in enumerate(unique_pred):
                        contingency_matrix[i, j] = np.sum((y_true == true_label) & (y_pred_clean == pred_label))
                
                return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            
            purity = purity_score(true_labels, cluster_labels)
            
            return {'ari': ari, 'purity': purity}
            
        except Exception as e:
            print(f"聚类评估失败: {e}")
            return {'ari': -1, 'purity': -1}


class EnhancedVisualizationEngine:
    """增强的可视化引擎"""
    
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
                       c=[colors[i]], label=f'Class {label}', alpha=alpha, s=50)
        
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
                          c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
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
    
    def plot_confusion_heatmap(self, true_labels, pred_labels, save_path=None):
        """绘制混淆矩阵热力图"""
        # 动态确定标签范围
        all_labels = np.concatenate([true_labels, pred_labels])
        unique_labels = sorted(np.unique(all_labels))
        
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Prediction Confusion Matrix')
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, features_2d, true_labels, pred_labels, save_path=None):
        """错误分析可视化"""
        plt.figure(figsize=self.figsize)
        
        # 正确和错误预测的mask
        correct_mask = true_labels == pred_labels
        error_mask = ~correct_mask
        
        # 绘制正确预测（灰色背景）
        plt.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1], 
                   c='lightgray', alpha=0.3, s=30, label='Correct')
        
        # 绘制错误预测，按真实标签着色
        if np.any(error_mask):
            unique_labels = np.unique(true_labels[error_mask])
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = error_mask & (true_labels == label)
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[colors[i]], label=f'Error Class {label}', alpha=0.8, s=60, 
                           edgecolors='black', linewidth=0.5)
        
        plt.title('Error Analysis: Misclassified Samples', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights, save_path=None):
        """可视化注意力权重热力图"""
        if attention_weights is None or len(attention_weights) == 0:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 如果是3D数据 [samples, seq_len, spatial_size]，取平均
        if len(attention_weights.shape) == 3:
            avg_attention = np.mean(attention_weights, axis=0)  # [seq_len, spatial_size]
        else:
            avg_attention = attention_weights
        
        # 重新整形为2D网格（假设spatial_size是正方形）
        spatial_size = avg_attention.shape[-1]
        grid_size = int(np.sqrt(spatial_size))
        
        if grid_size * grid_size == spatial_size:
            # 可以重新整形为正方形网格
            if len(avg_attention.shape) == 2:
                # 对每个时刻绘制注意力图
                seq_len = avg_attention.shape[0]
                cols = min(4, seq_len)
                rows = (seq_len + cols - 1) // cols
                
                for t in range(min(seq_len, 12)):  # 最多显示12个时刻
                    plt.subplot(rows, cols, t+1)
                    attention_map = avg_attention[t].reshape(grid_size, grid_size)
                    sns.heatmap(attention_map, cmap='viridis', cbar=True, square=True)
                    plt.title(f'Attention at t={t}')
                    plt.axis('off')
            else:
                # 单个注意力图
                attention_map = avg_attention.reshape(grid_size, grid_size)
                sns.heatmap(attention_map, cmap='viridis', cbar=True, square=True)
                plt.title('Average Attention Map')
                plt.axis('off')
        else:
            # 不能重新整形，绘制1D条形图
            if len(avg_attention.shape) == 1:
                plt.bar(range(len(avg_attention)), avg_attention)
                plt.title('Attention Weights')
                plt.xlabel('Spatial Location')
                plt.ylabel('Attention Weight')
            else:
                # 多个序列，绘制热力图
                sns.heatmap(avg_attention, cmap='viridis', cbar=True)
                plt.title('Attention Weights Heatmap')
                plt.xlabel('Spatial Location')
                plt.ylabel('Time Step')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embodiment_analysis(self, visual_features, embodiment_features, 
                                fusion_features, labels, save_path=None):
        """具身学习特定的分析可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 1. 视觉特征分布
        if visual_features is not None:
            visual_2d = self.reduce_dimensions(visual_features, 'tsne')
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0, 0].scatter(visual_2d[mask, 0], visual_2d[mask, 1], 
                                 c=[colors[i]], label=f'Class {label}', alpha=0.7)
            axes[0, 0].set_title('Visual Features')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 具身特征分布
        if embodiment_features is not None:
            embodiment_2d = self.reduce_dimensions(embodiment_features, 'tsne')
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0, 1].scatter(embodiment_2d[mask, 0], embodiment_2d[mask, 1], 
                                 c=[colors[i]], label=f'Class {label}', alpha=0.7)
            axes[0, 1].set_title('Embodiment Features')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 融合特征分布
        if fusion_features is not None:
            fusion_2d = self.reduce_dimensions(fusion_features, 'tsne')
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0, 2].scatter(fusion_2d[mask, 0], fusion_2d[mask, 1], 
                                 c=[colors[i]], label=f'Class {label}', alpha=0.7)
            axes[0, 2].set_title('Fusion Features')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 特征相关性分析
        if visual_features is not None and embodiment_features is not None:
            visual_mean = np.mean(visual_features, axis=1)
            embodiment_mean = np.mean(embodiment_features, axis=1)
            
            scatter = axes[1, 0].scatter(visual_mean, embodiment_mean, c=labels, 
                             cmap='tab10', alpha=0.7)
            axes[1, 0].set_xlabel('Visual Feature Mean')
            axes[1, 0].set_ylabel('Embodiment Feature Mean')
            axes[1, 0].set_title('Visual vs Embodiment Correlation')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # 5. 模态贡献分析
        if all(f is not None for f in [visual_features, embodiment_features, fusion_features]):
            visual_contribution = np.corrcoef(
                np.mean(visual_features, axis=1), 
                np.mean(fusion_features, axis=1)
            )[0, 1]
            embodiment_contribution = np.corrcoef(
                np.mean(embodiment_features, axis=1), 
                np.mean(fusion_features, axis=1)
            )[0, 1]
            
            contributions = [visual_contribution, embodiment_contribution]
            modal_names = ['Visual', 'Embodiment']
            
            axes[1, 1].bar(modal_names, contributions, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_title('Modal Contribution to Fusion')
            axes[1, 1].set_ylabel('Correlation with Fusion')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 决策置信度分析
        if fusion_features is not None:
            uncertainty = np.std(fusion_features, axis=1)
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[1, 2].scatter(np.arange(np.sum(mask)), uncertainty[mask], 
                                 c=[colors[i]], label=f'Class {label}', alpha=0.7)
            
            axes[1, 2].set_title('Decision Uncertainty by Class')
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Feature Uncertainty (std)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ModelAnalyzer:
    """主分析器 - 整合所有分析功能"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.feature_extractor = FeatureExtractor(model, device)
        self.cluster_analyzer = ClusterAnalyzer()
        self.visualizer = EnhancedVisualizationEngine()


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


def debug_labels_and_model(checkpoint_path, val_csv, data_root):
    """调试标签范围和模型输出"""
    print("=== 调试标签和模型输出 ===")
    
    try:
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root, batch_size=4
        )
        
        print(f"模型配置: {config['model_config']}")
        print(f"数据集大小: {len(val_loader.dataset)}")
        
        # 检查几个批次的数据
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:  # 只检查前两个批次
                    break
                
                print(f"\n--- 批次 {i+1} ---")
                
                # 输入数据
                sequence_data = {
                    'images': batch['sequence_data']['images'].to(device),
                    'joints': batch['sequence_data']['joints'].to(device),
                    'timestamps': batch['sequence_data']['timestamps'].to(device),
                    'labels': batch['sequence_data']['labels'].to(device)
                }
                labels = batch['label'].cpu().numpy()
                
                print(f"批次大小: {len(labels)}")
                print(f"图像形状: {sequence_data['images'].shape}")
                print(f"关节形状: {sequence_data['joints'].shape}")
                print(f"序列标签形状: {sequence_data['labels'].shape}")
                print(f"CSV标签: {labels}")
                print(f"CSV标签范围: {labels.min()} - {labels.max()}")
                
                # 模型前向传播
                outputs = model(
                    sequence_data=sequence_data,
                    use_teacher_forcing=False
                )
                
                count_logits = outputs['counts']  # [batch, seq_len, num_classes]
                print(f"模型输出logits形状: {count_logits.shape}")
                
                pred_labels = torch.argmax(count_logits, dim=-1)  # [batch, seq_len]
                final_pred = pred_labels[:, -1].cpu().numpy()  # 最终时刻的预测
                
                print(f"预测序列形状: {pred_labels.shape}")
                print(f"最终预测: {final_pred}")
                print(f"最终预测范围: {final_pred.min()} - {final_pred.max()}")
                
                # 对比最后几个时刻的目标和预测
                print("最后5个时刻的序列标签:")
                for j in range(len(labels)):
                    print(f"  样本{j}: {sequence_data['labels'][j, -5:].cpu().numpy()}")
                
                print("最后5个时刻的预测计数:")
                for j in range(len(labels)):
                    print(f"  样本{j}: {pred_labels[j, -5:].cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_model_structure(checkpoint_path):
    """检查模型结构"""
    print("🔬 检查模型结构...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        from Model_embodiment import EmbodiedCountingModel
        
        # 确定图像模式
        image_mode = config.get('image_mode', 'rgb')
        input_channels = 3 if image_mode == 'rgb' else 1
        
        model_config = config['model_config'].copy()
        model_config['input_channels'] = input_channels
        model = EmbodiedCountingModel(**model_config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        feature_extractor = FeatureExtractor(model, device)
        available_layers = feature_extractor.inspect_model_structure()
        auto_layers = feature_extractor.auto_detect_key_layers()
        
        print(f"\n✨ 自动检测的关键层: {auto_layers}")
        print(f"\n📋 推荐的分析策略:")
        print("  --mode quick --max_samples 100              # 最快分析")
        print("  --mode enhanced --max_samples 500           # 推荐的完整分析")
        print("  --layers fusion lstm --max_samples 300      # 自定义层分析")
        
        return available_layers, auto_layers
        
    except Exception as e:
        print(f"❌ 模型结构检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_embodied_counting_model_enhanced(checkpoint_path, val_csv, data_root, 
                                           save_dir='./enhanced_analysis', 
                                           max_samples=500, specific_layers=None):
    """增强版具身计数模型分析"""
    
    print("🤖 开始增强版具身计数模型分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建分析器
        analyzer = ModelAnalyzer(model, device)
        
        # 3. 确定要分析的层
        if specific_layers is None:
            print("🔍 自动检测关键层...")
            key_layers = analyzer.feature_extractor.auto_detect_key_layers()
            if not key_layers:
                key_layers = ['fusion', 'lstm', 'counting_decoder', 'visual_encoder', 'embodiment_encoder']
        else:
            key_layers = specific_layers
        
        print(f"📋 准备分析的层: {key_layers}")
        
        # 4. 注册钩子并提取特征
        successful_layers = analyzer.feature_extractor.register_hooks(key_layers)
        
        if not successful_layers:
            print("❌ 没有成功注册任何钩子！")
            return None
        
        try:
            # 5. 提取特征
            print("🎯 提取多层特征...")
            data = analyzer.feature_extractor.extract_features(val_loader, max_samples)
            
            features = data['features']
            predictions = data['predictions']
            true_labels = data['labels']
            attention_weights = data.get('attention_weights', None)
            
            print(f"✅ 特征提取完成:")
            print(f"   样本数: {len(true_labels)}")
            print(f"   提取层: {list(features.keys())}")
            print(f"   注意力权重: {'有' if attention_weights is not None else '无'}")
            
            # 6. 具身学习特定分析
            print("🧠 执行具身学习特定分析...")
            
            # 提取不同模态的特征
            visual_features = features.get('visual_encoder', None)
            embodiment_features = features.get('embodiment_encoder', None)
            fusion_features = features.get('fusion', None)
            lstm_features = features.get('lstm', None)
            
            # 具身学习分析可视化
            if any(f is not None for f in [visual_features, embodiment_features, fusion_features]):
                analyzer.visualizer.plot_embodiment_analysis(
                    visual_features, embodiment_features, fusion_features, true_labels,
                    save_path=os.path.join(save_dir, 'embodiment_analysis.png')
                )
                print("✅ 具身学习分析完成")
            
            # 7. 注意力机制可视化
            if attention_weights is not None:
                print("👁️ 可视化注意力机制...")
                analyzer.visualizer.plot_attention_heatmap(
                    attention_weights,
                    save_path=os.path.join(save_dir, 'attention_heatmap.png')
                )
                print("✅ 注意力可视化完成")
            
            # 8. 传统聚类和降维分析
            print("📊 执行传统聚类分析...")
            
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
                        features_2d = analyzer.visualizer.reduce_dimensions(layer_features, dim_method)
                        
                        # 可视化真实标签
                        analyzer.visualizer.plot_scatter(
                            features_2d, true_labels,
                            f'{layer_name} - True Labels ({dim_method.upper()})',
                            save_path=os.path.join(save_dir, f'{layer_name}_{dim_method}_true.png')
                        )
                        
                        # 可视化预测结果
                        analyzer.visualizer.plot_scatter(
                            features_2d, predictions,
                            f'{layer_name} - Predictions ({dim_method.upper()})',
                            save_path=os.path.join(save_dir, f'{layer_name}_{dim_method}_pred.png')
                        )
                        
                        # 错误分析
                        analyzer.visualizer.plot_error_analysis(
                            features_2d, true_labels, predictions,
                            save_path=os.path.join(save_dir, f'{layer_name}_{dim_method}_errors.png')
                        )
                        
                        # 聚类分析
                        cluster_results = analyzer.cluster_analyzer.perform_clustering(
                            layer_features, methods=['kmeans', 'dbscan'], true_labels=true_labels
                        )
                        
                        # 评估聚类质量
                        for method, result in cluster_results.items():
                            evaluation = analyzer.cluster_analyzer.evaluate_clustering(
                                result['labels'], true_labels
                            )
                            result.update(evaluation)
                        
                        # 聚类对比可视化
                        analyzer.visualizer.plot_clustering_comparison(
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
            
            # 9. 绘制混淆矩阵
            print("📈 生成混淆矩阵...")
            analyzer.visualizer.plot_confusion_heatmap(
                true_labels, predictions,
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )
            
            # 10. 生成综合报告
            print("📝 生成综合分析报告...")
            generate_enhanced_report(
                analysis_results, features, true_labels, predictions, 
                attention_weights, save_dir
            )
            
            print(f"🎉 增强版分析完成！结果保存在: {save_dir}")
            return analysis_results
            
        finally:
            analyzer.feature_extractor.remove_hooks()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_enhanced_report(analysis_results, features, true_labels, predictions, 
                           attention_weights, save_dir):
    """生成增强版分析报告"""
    
    # 基础统计
    accuracy = accuracy_score(true_labels, predictions)
    unique_true = len(np.unique(true_labels))
    unique_pred = len(np.unique(predictions))
    
    # 模态分析
    modality_analysis = {}
    if 'visual_encoder' in features and 'embodiment_encoder' in features:
        visual_var = np.var(features['visual_encoder'])
        embodiment_var = np.var(features['embodiment_encoder'])
        modality_analysis = {
            'visual_variance': float(visual_var),
            'embodiment_variance': float(embodiment_var),
            'modality_balance': float(visual_var / (visual_var + embodiment_var))
        }
    
    # 层级特征复杂度
    layer_complexity = {}
    for layer_name, layer_features in features.items():
        if layer_features is not None:
            layer_complexity[layer_name] = {
                'feature_dim': int(layer_features.shape[1]),
                'feature_variance': float(np.var(layer_features)),
                'feature_range': [float(np.min(layer_features)), float(np.max(layer_features))],
                'effective_rank': float(np.linalg.matrix_rank(layer_features))
            }
    
    # 注意力分析
    attention_analysis = {}
    if attention_weights is not None:
        attention_analysis = {
            'mean_attention': float(np.mean(attention_weights)),
            'attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1).mean()),
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
                best_ari = max([r.get('ari', 0) for r in clustering_results.values()], default=0)
                layer_clustering[dim_method] = {
                    'best_silhouette': float(best_silhouette),
                    'best_ari': float(best_ari)
                }
        clustering_summary[layer_name] = layer_clustering
    
    # 生成完整报告
    enhanced_report = {
        'analysis_type': 'Enhanced Embodied Counting Model Analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'summary': {
            'total_samples': int(len(true_labels)),
            'overall_accuracy': float(accuracy),
            'unique_true_labels': int(unique_true),
            'unique_predictions': int(unique_pred),
            'analyzed_layers': list(features.keys())
        },
        'modality_analysis': modality_analysis,
        'layer_complexity': layer_complexity,
        'attention_analysis': attention_analysis,
        'clustering_summary': clustering_summary,
        'recommendations': generate_recommendations(analysis_results, features, accuracy)
    }
    
    # 保存JSON报告
    try:
        with open(os.path.join(save_dir, 'enhanced_analysis_report.json'), 'w') as f:
            json.dump(enhanced_report, f, indent=2)
        print(f"✅ JSON报告已保存")
    except Exception as e:
        print(f"⚠️ JSON报告保存失败: {e}")
    
    # 生成可读报告
    try:
        with open(os.path.join(save_dir, 'enhanced_analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== 增强版具身计数模型分析报告 ===\n\n")
            f.write(f"分析时间: {enhanced_report['timestamp']}\n")
            f.write(f"总样本数: {enhanced_report['summary']['total_samples']}\n")
            f.write(f"整体准确率: {enhanced_report['summary']['overall_accuracy']:.4f}\n")
            f.write(f"分析层数: {len(enhanced_report['summary']['analyzed_layers'])}\n\n")
            
            # 模态分析
            if modality_analysis:
                f.write("=== 多模态分析 ===\n")
                f.write(f"视觉特征方差: {modality_analysis['visual_variance']:.4f}\n")
                f.write(f"具身特征方差: {modality_analysis['embodiment_variance']:.4f}\n")
                f.write(f"模态平衡度: {modality_analysis['modality_balance']:.4f}\n\n")
            
            # 层级复杂度
            f.write("=== 层级特征复杂度 ===\n")
            for layer_name, complexity in layer_complexity.items():
                f.write(f"{layer_name}:\n")
                f.write(f"  特征维度: {complexity['feature_dim']}\n")
                f.write(f"  特征方差: {complexity['feature_variance']:.4f}\n")
                f.write(f"  有效秩: {complexity['effective_rank']:.1f}\n")
            f.write("\n")
            
            # 注意力分析
            if attention_analysis:
                f.write("=== 注意力机制分析 ===\n")
                f.write(f"平均注意力强度: {attention_analysis['mean_attention']:.4f}\n")
                f.write(f"注意力熵: {attention_analysis['attention_entropy']:.4f}\n")
                f.write(f"注意力稀疏性: {attention_analysis['attention_sparsity']:.4f}\n\n")
            
            # 聚类质量总结
            f.write("=== 聚类质量总结 ===\n")
            for layer_name, clustering in clustering_summary.items():
                f.write(f"{layer_name}:\n")
                for dim_method, scores in clustering.items():
                    f.write(f"  {dim_method}: 轮廓系数={scores['best_silhouette']:.3f}, ARI={scores['best_ari']:.3f}\n")
            f.write("\n")
            
            # 建议
            f.write("=== 分析建议 ===\n")
            for rec in enhanced_report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"✅ 文本报告已保存")
        
    except Exception as e:
        print(f"⚠️ 文本报告保存失败: {e}")


def generate_recommendations(analysis_results, features, accuracy):
    """基于分析结果生成建议"""
    recommendations = []
    
    # 准确率建议
    if accuracy < 0.7:
        recommendations.append("模型准确率较低，建议检查数据质量或调整模型架构")
    elif accuracy > 0.9:
        recommendations.append("模型准确率很高，可以考虑在更复杂的任务上测试")
    
    # 特征维度建议
    high_dim_layers = [name for name, feats in features.items() 
                      if feats is not None and feats.shape[1] > 512]
    if high_dim_layers:
        recommendations.append(f"层 {high_dim_layers} 特征维度较高，可考虑降维优化")
    
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
        
        if avg_silhouette and np.mean(avg_silhouette) < 0.3:
            poor_clustering_layers.append(layer_name)
    
    if poor_clustering_layers:
        recommendations.append(f"层 {poor_clustering_layers} 的聚类质量较差，特征可分离性不强")
    
    # 模态平衡建议
    if 'visual_encoder' in features and 'embodiment_encoder' in features:
        visual_var = np.var(features['visual_encoder'])
        embodiment_var = np.var(features['embodiment_encoder'])
        ratio = visual_var / (embodiment_var + 1e-8)
        
        if ratio > 10:
            recommendations.append("视觉特征占主导地位，建议加强具身特征的表达能力")
        elif ratio < 0.1:
            recommendations.append("具身特征占主导地位，建议平衡多模态特征")
    
    return recommendations


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='具身计数模型分析工具')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 分析选项
    parser.add_argument('--mode', type=str, default='enhanced',
                       choices=['debug', 'inspect', 'quick', 'enhanced'],
                       help='分析模式')
    parser.add_argument('--save_dir', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/analysis_results',
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=500,
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
        args.save_dir = f'./analysis_results_{args.mode}_{timestamp}'
    
    # 检查输入文件
    for path, name in [(args.checkpoint, '检查点文件'), 
                       (args.val_csv, '验证CSV文件'), 
                       (args.data_root, '数据根目录')]:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    print("🚀 具身计数模型分析工具")
    print("="*50)
    print(f"模式: {args.mode}")
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print("="*50)
    
    start_time = time.time()
    
    try:
        if args.mode == 'debug':
            print("🔍 运行调试模式...")
            success = debug_labels_and_model(args.checkpoint, args.val_csv, args.data_root)
            if success:
                print("✅ 调试完成")
            else:
                print("❌ 调试失败")
        
        elif args.mode == 'inspect':
            print("🔬 检查模型结构...")
            inspect_model_structure(args.checkpoint)
        
        elif args.mode == 'quick':
            print("⚡ 快速分析...")
            results = analyze_embodied_counting_model_enhanced(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, max_samples=min(100, args.max_samples),
                specific_layers=args.layers
            )
        
        elif args.mode == 'enhanced':
            print("🤖 增强版分析...")
            results = analyze_embodied_counting_model_enhanced(
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
        print("🤖 具身计数模型分析工具")
        print("="*50)
        print("使用方法:")
        print("python embodied_analysis.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
        print("\n可用模式:")
        print("  --mode debug        # 调试模型和数据")
        print("  --mode inspect      # 检查模型结构")
        print("  --mode quick        # 快速分析（少量样本）")
        print("  --mode enhanced     # 增强分析（推荐）")
        print("\n基本示例:")
        print("python embodied_analysis.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode enhanced")
        print("\n高级示例:")
        print("python embodied_analysis.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode enhanced \\")
        print("    --layers fusion lstm counting_decoder \\")
        print("    --max_samples 1000 \\")
        print("    --save_dir ./my_analysis")
        print("\n📋 推荐工作流:")
        print("1. 首先运行: --mode inspect    # 查看模型结构")
        print("2. 然后运行: --mode debug      # 验证数据加载")
        print("3. 最后运行: --mode enhanced   # 完整分析")
        print("\n💡 提示:")
        print("- 如果是第一次使用，建议先用 --mode quick 测试")
        print("- enhanced 模式提供最全面的分析")
        print("- 可以用 --layers 参数指定特定层进行分析")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷函数，供其他脚本调用
# =============================================================================

def quick_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=100):
    """快速分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./quick_analysis_{timestamp}'
    
    return analyze_embodied_counting_model_enhanced(
        checkpoint_path, val_csv, data_root, save_dir, max_samples
    )


def full_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=500, layers=None):
    """完整分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./full_analysis_{timestamp}'
    
    return analyze_embodied_counting_model_enhanced(
        checkpoint_path, val_csv, data_root, save_dir, max_samples, layers
    )


def batch_analysis(checkpoint_paths, val_csv, data_root, base_save_dir='./batch_analysis'):
    """批量分析多个模型"""
    results = {}
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"分析模型 {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        print(f"{'='*60}")
        
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        save_dir = os.path.join(base_save_dir, f'model_{i+1}_{model_name}')
        
        try:
            result = analyze_embodied_counting_model_enhanced(
                checkpoint_path, val_csv, data_root, save_dir, max_samples=300
            )
            results[model_name] = result
            print(f"✅ 模型 {model_name} 分析完成")
        except Exception as e:
            print(f"❌ 模型 {model_name} 分析失败: {e}")
            results[model_name] = None
    
    # 生成对比报告
    generate_comparison_report(results, base_save_dir)
    
    return results


def generate_comparison_report(results, save_dir):
    """生成多模型对比报告"""
    print("\n📊 生成多模型对比报告...")
    
    comparison_data = []
    
    for model_name, result in results.items():
        if result is None:
            continue
        
        # 这里可以提取每个模型的关键指标进行对比
        # 由于结果结构比较复杂，这里提供一个框架
        model_summary = {
            'model_name': model_name,
            'status': 'success' if result else 'failed'
        }
        comparison_data.append(model_summary)
    
    # 保存对比结果
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    print(f"✅ 对比报告保存在: {os.path.join(save_dir, 'model_comparison.csv')}")


# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

1. 命令行使用:
   python embodied_analysis.py \\
       --checkpoint ./best_model.pth \\
       --val_csv ./val.csv \\
       --data_root ./data \\
       --mode enhanced \\
       --max_samples 500

2. 在Python脚本中使用:
   from embodied_analysis import quick_analysis, full_analysis
   
   # 快速分析
   results = quick_analysis(
       checkpoint_path='./best_model.pth',
       val_csv='./val.csv', 
       data_root='./data'
   )
   
   # 完整分析
   results = full_analysis(
       checkpoint_path='./best_model.pth',
       val_csv='./val.csv',
       data_root='./data',
       max_samples=1000,
       layers=['fusion', 'lstm', 'counting_decoder']
   )

3. 批量分析多个模型:
   from embodied_analysis import batch_analysis
   
   checkpoint_paths = [
       './model1.pth',
       './model2.pth', 
       './model3.pth'
   ]
   
   results = batch_analysis(
       checkpoint_paths=checkpoint_paths,
       val_csv='./val.csv',
       data_root='./data'
   )
"""