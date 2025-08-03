"""
Integrated Single Image CNN Model Analysis Tool
集成的单图像CNN模型分析工具 - 包含降维可视化和数值神经元分析
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
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
        
        # 打印调试信息
        print(f"单图像特征提取完成:")
        print(f"  实际样本数: {len(result['labels'])}")
        print(f"  真实标签范围: {result['labels'].min()} - {result['labels'].max()}")
        print(f"  预测标签范围: {result['predictions'].min()} - {result['predictions'].max()}")
        print(f"  真实标签唯一值: {sorted(np.unique(result['labels']))}")
        print(f"  预测标签唯一值: {sorted(np.unique(result['predictions']))}")
        print(f"  提取的特征层: {list(final_features.keys())}")
        
        return result


class NumberLineAnalyzer:
    """Number Line分析器"""
    
    def __init__(self, features_dict, labels, layer_names=None):
        """
        Args:
            features_dict: {layer_name: features_array} 各层特征
            labels: 真实数值标签 (1-10)
            layer_names: 要分析的层名称列表
        """
        self.features_dict = features_dict
        self.labels = np.array(labels)
        self.layer_names = layer_names or list(features_dict.keys())
        self.results = {}
        
    def find_number_line_neurons(self, layer_name, min_r2=0.5, method='linear'):
        """
        寻找具有number line特性的神经元
        
        Args:
            layer_name: 层名称
            min_r2: 最小R²阈值
            method: 'linear', 'log', 'sqrt' - 不同的数值编码假设
        
        Returns:
            dict: number line神经元的分析结果
        """
        features = self.features_dict[layer_name]  # [samples, neurons]
        
        if method == 'linear':
            target = self.labels
        elif method == 'log':
            target = np.log(self.labels)
        elif method == 'sqrt':
            target = np.sqrt(self.labels)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_neurons = features.shape[1]
        number_line_neurons = []
        
        print(f"🔍 分析 {layer_name} 层的 {n_neurons} 个神经元...")
        
        for neuron_idx in tqdm(range(n_neurons), desc="寻找number line神经元"):
            neuron_response = features[:, neuron_idx]
            
            # 线性回归拟合
            reg = LinearRegression()
            reg.fit(target.reshape(-1, 1), neuron_response)
            predicted = reg.predict(target.reshape(-1, 1))
            
            # 计算拟合质量
            r2 = r2_score(neuron_response, predicted)
            correlation, p_value = pearsonr(target, neuron_response)
            
            if r2 >= min_r2 and p_value < 0.05:
                number_line_neurons.append({
                    'neuron_idx': neuron_idx,
                    'r2_score': r2,
                    'correlation': correlation,
                    'p_value': p_value,
                    'slope': reg.coef_[0],
                    'intercept': reg.intercept_,
                    'response': neuron_response.copy(),
                    'target_values': target.copy()
                })
        
        # 按R²分数排序
        number_line_neurons.sort(key=lambda x: x['r2_score'], reverse=True)
        
        result = {
            'layer_name': layer_name,
            'method': method,
            'total_neurons': n_neurons,
            'number_line_neurons': number_line_neurons,
            'proportion': len(number_line_neurons) / n_neurons
        }
        
        print(f"✅ 找到 {len(number_line_neurons)} 个number line神经元 "
              f"({result['proportion']:.1%} of total)")
        
        return result
    
    def find_number_selective_neurons(self, layer_name, selectivity_threshold=0.3):
        """
        寻找数值选择性神经元
        
        Args:
            layer_name: 层名称
            selectivity_threshold: 选择性阈值
        
        Returns:
            dict: 数值选择性神经元分析结果
        """
        features = self.features_dict[layer_name]
        n_neurons = features.shape[1]
        
        # 计算每个神经元对每个数值的平均响应
        unique_numbers = np.unique(self.labels)
        response_matrix = np.zeros((len(unique_numbers), n_neurons))
        
        for i, num in enumerate(unique_numbers):
            mask = self.labels == num
            if np.sum(mask) > 0:
                response_matrix[i, :] = np.mean(features[mask, :], axis=0)
        
        selective_neurons = []
        
        print(f"🔍 分析 {layer_name} 层的数值选择性...")
        
        for neuron_idx in tqdm(range(n_neurons), desc="计算选择性"):
            responses = response_matrix[:, neuron_idx]
            
            # 计算选择性指数
            max_response = np.max(responses)
            min_response = np.min(responses)
            
            if max_response != min_response:
                selectivity_index = (max_response - min_response) / (max_response + min_response + 1e-8)
            else:
                selectivity_index = 0
            
            # 找到最佳数值
            preferred_number = unique_numbers[np.argmax(responses)]
            
            # 计算调谐曲线的锐度
            tuning_width = self._calculate_tuning_width(responses, unique_numbers)
            
            if selectivity_index >= selectivity_threshold:
                selective_neurons.append({
                    'neuron_idx': neuron_idx,
                    'selectivity_index': selectivity_index,
                    'preferred_number': preferred_number,
                    'tuning_width': tuning_width,
                    'response_profile': responses.copy(),
                    'max_response': max_response,
                    'response_ratio': max_response / (min_response + 1e-8)
                })
        
        # 按选择性指数排序
        selective_neurons.sort(key=lambda x: x['selectivity_index'], reverse=True)
        
        result = {
            'layer_name': layer_name,
            'total_neurons': n_neurons,
            'selective_neurons': selective_neurons,
            'proportion': len(selective_neurons) / n_neurons,
            'unique_numbers': unique_numbers,
            'response_matrix': response_matrix
        }
        
        print(f"✅ 找到 {len(selective_neurons)} 个数值选择性神经元 "
              f"({result['proportion']:.1%} of total)")
        
        return result
    
    def _calculate_tuning_width(self, responses, numbers):
        """计算调谐曲线宽度"""
        # 标准化响应
        responses_norm = (responses - np.min(responses)) / (np.max(responses) - np.min(responses) + 1e-8)
        
        # 计算半高宽度
        max_idx = np.argmax(responses_norm)
        half_max = responses_norm[max_idx] / 2
        
        # 找到半高点
        left_half = np.where(responses_norm[:max_idx] <= half_max)[0]
        right_half = np.where(responses_norm[max_idx:] <= half_max)[0]
        
        if len(left_half) > 0 and len(right_half) > 0:
            left_bound = left_half[-1] if len(left_half) > 0 else 0
            right_bound = max_idx + right_half[0] if len(right_half) > 0 else len(responses) - 1
            tuning_width = numbers[right_bound] - numbers[left_bound]
        else:
            tuning_width = len(numbers)  # 很宽的调谐
        
        return tuning_width


class IntegratedVisualizationEngine:
    """集成的可视化引擎 - 只包含必要的分析"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def reduce_dimensions(self, features, method='tsne', n_components=2):
        """降维"""
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(features)//4))
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
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
    
    def plot_number_line_neurons(self, number_line_result, save_path=None, top_n=6):
        """可视化number line神经元"""
        neurons = number_line_result['number_line_neurons'][:top_n]
        layer_name = number_line_result['layer_name']
        
        if not neurons:
            print(f"⚠️ {layer_name} 层没有找到number line神经元")
            return
        
        n_cols = min(3, len(neurons))
        n_rows = (len(neurons) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if len(neurons) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        unique_numbers = np.unique(neurons[0]['target_values'])
        
        for i, neuron in enumerate(neurons):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 数值 vs 神经元响应
            target_values = neuron['target_values']
            responses = neuron['response']
            
            # 计算每个数值的平均响应和标准差
            avg_responses = []
            std_responses = []
            for num in unique_numbers:
                mask = target_values == num
                if np.sum(mask) > 0:
                    avg_responses.append(np.mean(responses[mask]))
                    std_responses.append(np.std(responses[mask]))
                else:
                    avg_responses.append(0)
                    std_responses.append(0)
            
            ax.errorbar(unique_numbers, avg_responses, yerr=std_responses, 
                       marker='o', capsize=5, linewidth=2, markersize=8)
            
            # 添加拟合线
            reg_line = neuron['slope'] * unique_numbers + neuron['intercept']
            ax.plot(unique_numbers, reg_line, '--', color='red', linewidth=2, alpha=0.7)
            
            ax.set_title(f'Neuron {neuron["neuron_idx"]}\n'
                        f'R² = {neuron["r2_score"]:.3f}, r = {neuron["correlation"]:.3f}')
            ax.set_xlabel('Number')
            ax.set_ylabel('Neural Response')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(neurons), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Number Line Neurons - {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_number_selective_neurons(self, selective_result, save_path=None, top_n=6):
        """可视化数值选择性神经元"""
        neurons = selective_result['selective_neurons'][:top_n]
        layer_name = selective_result['layer_name']
        unique_numbers = selective_result['unique_numbers']
        
        if not neurons:
            print(f"⚠️ {layer_name} 层没有找到number selective神经元")
            return
        
        n_cols = min(3, len(neurons))
        n_rows = (len(neurons) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if len(neurons) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, neuron in enumerate(neurons):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 调谐曲线
            responses = neuron['response_profile']
            ax.bar(unique_numbers, responses, alpha=0.7, 
                  color='skyblue', edgecolor='navy', linewidth=1)
            
            # 标记偏好数值
            preferred_idx = np.argmax(responses)
            ax.bar(unique_numbers[preferred_idx], responses[preferred_idx], 
                  color='red', alpha=0.8, label=f'Preferred: {neuron["preferred_number"]}')
            
            ax.set_title(f'Neuron {neuron["neuron_idx"]}\n'
                        f'Selectivity = {neuron["selectivity_index"]:.3f}')
            ax.set_xlabel('Number')
            ax.set_ylabel('Average Response')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(neurons), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Number Selective Neurons - {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_layer_comparison(self, all_results, save_path=None):
        """对比不同层的number神经元比例"""
        layer_names = []
        number_line_props = []
        selective_props = []
        
        for layer_name, results in all_results.items():
            layer_names.append(layer_name)
            
            if 'number_line' in results:
                number_line_props.append(results['number_line']['proportion'])
            else:
                number_line_props.append(0)
                
            if 'selective' in results:
                selective_props.append(results['selective']['proportion'])
            else:
                selective_props.append(0)
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width/2, number_line_props, width, label='Number Line Neurons', alpha=0.8)
        ax.bar(x + width/2, selective_props, width, label='Number Selective Neurons', alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Proportion of Neurons')
        ax.set_title('Number Neurons Across Layers')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


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


def analyze_single_image_integrated(checkpoint_path, val_csv, data_root, 
                                   save_dir='./integrated_analysis', 
                                   max_samples=500, specific_layers=None,
                                   min_r2=0.5, selectivity_threshold=0.3):
    """集成的单图像CNN模型分析 - 降维+数值神经元"""
    
    print("🖼️ 开始集成的单图像CNN模型分析...")
    
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
            
            print(f"✅ 特征提取完成:")
            print(f"   样本数: {len(true_labels)}")
            print(f"   提取层: {list(features.keys())}")
            
            # 6. 创建可视化引擎
            visualizer = IntegratedVisualizationEngine()
            
            # 7. 降维分析 - 只保留PCA和t-SNE
            print("🎨 生成降维可视化...")
            
            for layer_name, layer_features in features.items():
                if layer_features is None:
                    continue
                    
                print(f"   分析 {layer_name} 层...")
                
                # PCA降维
                try:
                    features_pca = visualizer.reduce_dimensions(layer_features, 'pca')
                    visualizer.plot_scatter(
                        features_pca, true_labels,
                        f'{layer_name} - PCA',
                        save_path=os.path.join(save_dir, f'{layer_name}_pca.png')
                    )
                except Exception as e:
                    print(f"     PCA分析失败: {e}")
                
                # t-SNE降维
                try:
                    features_tsne = visualizer.reduce_dimensions(layer_features, 'tsne')
                    visualizer.plot_scatter(
                        features_tsne, true_labels,
                        f'{layer_name} - t-SNE',
                        save_path=os.path.join(save_dir, f'{layer_name}_tsne.png')
                    )
                except Exception as e:
                    print(f"     t-SNE分析失败: {e}")
            
            # 8. 数值神经元分析
            print("🧠 开始数值神经元分析...")
            
            # 创建数值神经元分析器
            number_analyzer = NumberLineAnalyzer(features, true_labels)
            
            all_number_results = {}
            
            for layer_name in features.keys():
                print(f"\n📊 分析层: {layer_name}")
                
                layer_results = {}
                
                # Number Line分析
                print("🔍 寻找Number Line神经元...")
                number_line_result = number_analyzer.find_number_line_neurons(
                    layer_name, min_r2=min_r2, method='linear'
                )
                layer_results['number_line'] = number_line_result
                
                # 可视化Number Line神经元
                visualizer.plot_number_line_neurons(
                    number_line_result,
                    save_path=os.path.join(save_dir, f'{layer_name}_number_line_neurons.png')
                )
                
                # Number Selective分析
                print("🔍 寻找Number Selective神经元...")
                selective_result = number_analyzer.find_number_selective_neurons(
                    layer_name, selectivity_threshold=selectivity_threshold
                )
                layer_results['selective'] = selective_result
                
                # 可视化Number Selective神经元
                visualizer.plot_number_selective_neurons(
                    selective_result,
                    save_path=os.path.join(save_dir, f'{layer_name}_number_selective_neurons.png')
                )
                
                all_number_results[layer_name] = layer_results
            
            # 9. 跨层比较
            print("\n📈 生成跨层比较...")
            visualizer.plot_layer_comparison(
                all_number_results,
                save_path=os.path.join(save_dir, 'number_neurons_layer_comparison.png')
            )
            
            # 10. 生成综合报告
            print("📝 生成分析报告...")
            generate_integrated_report(
                all_number_results, features, true_labels, predictions, config, save_dir
            )
            
            print(f"🎉 集成分析完成！结果保存在: {save_dir}")
            return {
                'features': features,
                'number_results': all_number_results,
                'labels': true_labels,
                'predictions': predictions
            }
            
        finally:
            feature_extractor.remove_hooks()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_integrated_report(all_number_results, features, true_labels, predictions, config, save_dir):
    """生成集成分析报告"""
    
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
    
    # 数值神经元总结
    number_neuron_summary = {}
    for layer_name, results in all_number_results.items():
        layer_summary = {}
        
        if 'number_line' in results:
            nl_result = results['number_line']
            layer_summary['number_line'] = {
                'total_neurons': nl_result['total_neurons'],
                'number_line_count': len(nl_result['number_line_neurons']),
                'proportion': nl_result['proportion'],
                'best_r2': max([n['r2_score'] for n in nl_result['number_line_neurons']], default=0),
                'avg_r2': np.mean([n['r2_score'] for n in nl_result['number_line_neurons']]) if nl_result['number_line_neurons'] else 0
            }
        
        if 'selective' in results:
            sel_result = results['selective']
            layer_summary['selective'] = {
                'selective_count': len(sel_result['selective_neurons']),
                'proportion': sel_result['proportion'],
                'best_selectivity': max([n['selectivity_index'] for n in sel_result['selective_neurons']], default=0),
                'avg_selectivity': np.mean([n['selectivity_index'] for n in sel_result['selective_neurons']]) if sel_result['selective_neurons'] else 0
            }
        
        number_neuron_summary[layer_name] = layer_summary
    
    # 生成完整报告
    report = {
        'analysis_type': 'Integrated Single Image CNN Analysis',
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
        'number_neuron_summary': number_neuron_summary,
        'recommendations': generate_integrated_recommendations(all_number_results, features, accuracy, config)
    }
    
    # 保存JSON报告
    try:
        with open(os.path.join(save_dir, 'integrated_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ JSON报告已保存")
    except Exception as e:
        print(f"⚠️ JSON报告保存失败: {e}")
    
    # 生成可读报告
    try:
        with open(os.path.join(save_dir, 'integrated_analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== 集成单图像CNN模型分析报告 ===\n\n")
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
                f.write(f"  数值范围: [{complexity['feature_range'][0]:.4f}, {complexity['feature_range'][1]:.4f}]\n\n")
            
            # 数值神经元分析
            f.write("=== 数值神经元分析 ===\n")
            for layer_name, summary in number_neuron_summary.items():
                f.write(f"{layer_name}:\n")
                
                if 'number_line' in summary:
                    nl = summary['number_line']
                    f.write(f"  Number Line神经元:\n")
                    f.write(f"    总神经元数: {nl['total_neurons']}\n")
                    f.write(f"    Number Line数量: {nl['number_line_count']}\n")
                    f.write(f"    比例: {nl['proportion']:.2%}\n")
                    f.write(f"    最佳R²: {nl['best_r2']:.3f}\n")
                    f.write(f"    平均R²: {nl['avg_r2']:.3f}\n")
                
                if 'selective' in summary:
                    sel = summary['selective']
                    f.write(f"  Number Selective神经元:\n")
                    f.write(f"    Selective数量: {sel['selective_count']}\n")
                    f.write(f"    比例: {sel['proportion']:.2%}\n")
                    f.write(f"    最佳选择性: {sel['best_selectivity']:.3f}\n")
                    f.write(f"    平均选择性: {sel['avg_selectivity']:.3f}\n")
                f.write("\n")
            
            # 建议
            f.write("=== 分析建议 ===\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"✅ 文本报告已保存")
        
    except Exception as e:
        print(f"⚠️ 文本报告保存失败: {e}")


def generate_integrated_recommendations(all_number_results, features, accuracy, config):
    """基于集成分析结果生成建议"""
    recommendations = []
    
    # 准确率建议
    if accuracy < 0.6:
        recommendations.append("模型准确率较低，建议检查数据质量、增加训练轮数或调整学习率")
    elif accuracy > 0.95:
        recommendations.append("模型准确率很高，可能存在过拟合，建议在测试集上验证")
    elif accuracy > 0.85:
        recommendations.append("模型准确率良好，可以考虑在更复杂的任务上测试")
    
    # 数值神经元建议
    total_number_line = sum(len(r['number_line']['number_line_neurons']) for r in all_number_results.values() if 'number_line' in r)
    total_selective = sum(len(r['selective']['selective_neurons']) for r in all_number_results.values() if 'selective' in r)
    total_neurons = sum(r['number_line']['total_neurons'] for r in all_number_results.values() if 'number_line' in r)
    
    if total_number_line == 0:
        recommendations.append("未发现Number Line神经元，模型可能没有学到数值的连续表征")
    elif total_number_line / total_neurons < 0.01:
        recommendations.append("Number Line神经元比例很低，建议增加数值相关的训练数据或调整模型架构")
    elif total_number_line / total_neurons > 0.1:
        recommendations.append("发现大量Number Line神经元，表明模型很好地学习了数值的线性表征")
    
    if total_selective == 0:
        recommendations.append("未发现Number Selective神经元，模型可能缺乏对特定数值的专门表征")
    elif total_selective / total_neurons < 0.01:
        recommendations.append("Number Selective神经元比例很低，可能需要更多特定数值的训练样本")
    elif total_selective / total_neurons > 0.05:
        recommendations.append("发现较多Number Selective神经元，表明模型对不同数值有良好的区分能力")
    
    # 层级分析建议
    layer_number_line_props = {name: r['number_line']['proportion'] for name, r in all_number_results.items() if 'number_line' in r}
    if layer_number_line_props:
        best_layer = max(layer_number_line_props, key=layer_number_line_props.get)
        worst_layer = min(layer_number_line_props, key=layer_number_line_props.get)
        
        if layer_number_line_props[best_layer] > 0.05:
            recommendations.append(f"{best_layer}层显示出最强的数值线性编码能力，是数值处理的关键层")
        if layer_number_line_props[worst_layer] < 0.01:
            recommendations.append(f"{worst_layer}层的数值编码能力较弱，可能更专注于其他特征")
    
    # 特征维度建议
    high_dim_layers = [name for name, feats in features.items() 
                      if feats is not None and feats.shape[1] > 1024]
    if high_dim_layers:
        recommendations.append(f"层 {high_dim_layers} 特征维度很高，可考虑降维或正则化防止过拟合")
    
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
    parser = argparse.ArgumentParser(description='集成的单图像CNN模型分析工具')
    
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
    
    # 数值神经元分析参数
    parser.add_argument('--min_r2', type=float, default=0.5,
                       help='Number Line神经元的最小R²阈值')
    parser.add_argument('--selectivity_threshold', type=float, default=0.3,
                       help='Number Selective神经元的选择性阈值')
    
    # 其他选项
    parser.add_argument('--batch_size', type=int, default=32,
                       help='数据加载批次大小')
    
    args = parser.parse_args()
    
    # 设置默认保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./integrated_single_image_analysis_{args.mode}_{timestamp}'
    
    # 检查输入文件
    for path, name in [(args.checkpoint, '检查点文件'), 
                       (args.val_csv, '验证CSV文件'), 
                       (args.data_root, '数据根目录')]:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    print("🖼️ 集成的单图像CNN模型分析工具")
    print("="*50)
    print(f"模式: {args.mode}")
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print(f"分析内容: 降维可视化(PCA+t-SNE) + 数值神经元分析")
    print("="*50)
    
    start_time = time.time()
    
    try:
        if args.mode == 'inspect':
            print("🔬 检查模型结构...")
            inspect_single_image_model_structure(args.checkpoint)
        
        elif args.mode == 'quick':
            print("⚡ 快速分析...")
            results = analyze_single_image_integrated(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, max_samples=min(100, args.max_samples),
                specific_layers=args.layers,
                min_r2=args.min_r2, 
                selectivity_threshold=args.selectivity_threshold
            )
        
        elif args.mode == 'full':
            print("🖼️ 完整分析...")
            results = analyze_single_image_integrated(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, args.max_samples, args.layers,
                min_r2=args.min_r2,
                selectivity_threshold=args.selectivity_threshold
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
        print("🖼️ 集成的单图像CNN模型分析工具")
        print("="*50)
        print("功能: 降维可视化(PCA+t-SNE) + 数值神经元分析")
        print("使用方法:")
        print("python integrated_single_image_analysis.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
        print("\n可用模式:")
        print("  --mode inspect      # 检查模型结构")
        print("  --mode quick        # 快速分析（少量样本）")
        print("  --mode full         # 完整分析（推荐）")
        print("\n基本示例:")
        print("python integrated_single_image_analysis.py \\")
        print("    --checkpoint ./best_single_image_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode full")
        print("\n高级示例:")
        print("python integrated_single_image_analysis.py \\")
        print("    --checkpoint ./best_single_image_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode full \\")
        print("    --layers visual_encoder classifier \\")
        print("    --max_samples 1000 \\")
        print("    --min_r2 0.6 \\")
        print("    --selectivity_threshold 0.4 \\")
        print("    --save_dir ./my_integrated_analysis")
        print("\n📋 推荐工作流:")
        print("1. 首先运行: --mode inspect    # 查看模型结构")
        print("2. 然后运行: --mode quick      # 快速验证")
        print("3. 最后运行: --mode full       # 完整分析")
        print("\n💡 输出内容:")
        print("- 各层PCA和t-SNE降维可视化")
        print("- Number Line神经元分析和可视化")
        print("- Number Selective神经元分析和可视化")
        print("- 跨层数值神经元比例对比")
        print("- 综合分析报告")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷函数，供其他脚本调用
# =============================================================================

def quick_integrated_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=100):
    """快速集成分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./quick_integrated_analysis_{timestamp}'
    
    return analyze_single_image_integrated(
        checkpoint_path, val_csv, data_root, save_dir, max_samples
    )


def full_integrated_analysis(checkpoint_path, val_csv, data_root, save_dir=None, 
                            max_samples=500, layers=None, min_r2=0.5, selectivity_threshold=0.3):
    """完整集成分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./full_integrated_analysis_{timestamp}'
    
    return analyze_single_image_integrated(
        checkpoint_path, val_csv, data_root, save_dir, max_samples, layers,
        min_r2, selectivity_threshold
    )


# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

1. 命令行使用:
   python integrated_single_image_analysis.py \\
       --checkpoint ./best_single_image_model.pth \\
       --val_csv ./val.csv \\
       --data_root ./data \\
       --mode full \\
       --max_samples 500

2. 在Python脚本中使用:
   from integrated_single_image_analysis import quick_integrated_analysis, full_integrated_analysis
   
   # 快速分析
   results = quick_integrated_analysis(
       checkpoint_path='./best_single_image_model.pth',
       val_csv='./val.csv', 
       data_root='./data'
   )
   
   # 完整分析
   results = full_integrated_analysis(
       checkpoint_path='./best_single_image_model.pth',
       val_csv='./val.csv',
       data_root='./data',
       max_samples=1000,
       layers=['visual_encoder', 'classifier'],
       min_r2=0.6,
       selectivity_threshold=0.4
   )

输出内容:
- layer_name_pca.png: PCA降维可视化
- layer_name_tsne.png: t-SNE降维可视化  
- layer_name_number_line_neurons.png: Number Line神经元可视化
- layer_name_number_selective_neurons.png: Number Selective神经元可视化
- number_neurons_layer_comparison.png: 跨层数值神经元比较
- integrated_analysis_report.json: 详细分析报告(JSON)
- integrated_analysis_summary.txt: 分析总结报告(文本)
""" 