"""
寻找Number Line和Number Selective Neurons的分析工具
专门针对计数任务的神经元选择性分析
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
                    'response': neuron_response.copy()
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
            response_matrix[i, :] = np.mean(features[mask, :], axis=0)
        
        selective_neurons = []
        
        print(f"🔍 分析 {layer_name} 层的数值选择性...")
        
        for neuron_idx in tqdm(range(n_neurons), desc="计算选择性"):
            responses = response_matrix[:, neuron_idx]
            
            # 计算选择性指数
            max_response = np.max(responses)
            min_response = np.min(responses)
            
            if max_response != min_response:
                selectivity_index = (max_response - min_response) / (max_response + min_response)
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
    
    def analyze_representational_geometry(self, layer_name):
        """分析表征几何学"""
        features = self.features_dict[layer_name]
        unique_numbers = np.unique(self.labels)
        
        # 计算每个数值的平均表征
        centroids = []
        for num in unique_numbers:
            mask = self.labels == num
            centroid = np.mean(features[mask, :], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)  # [numbers, features]
        
        # 计算数值间的表征距离
        distances = pdist(centroids, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # 计算数值差异
        number_diffs = np.abs(unique_numbers[:, None] - unique_numbers[None, :])
        
        # 表征距离 vs 数值距离的相关性
        triu_indices = np.triu_indices(len(unique_numbers), k=1)
        repr_distances = distance_matrix[triu_indices]
        num_distances = number_diffs[triu_indices]
        
        correlation, p_value = pearsonr(num_distances, repr_distances)
        
        result = {
            'layer_name': layer_name,
            'centroids': centroids,
            'distance_matrix': distance_matrix,
            'number_correlation': correlation,
            'correlation_p_value': p_value,
            'unique_numbers': unique_numbers
        }
        
        return result


class NumberNeuronVisualizer:
    """Number Line和Number Selective Neurons可视化工具"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_number_line_neurons(self, number_line_result, save_path=None, top_n=6):
        """可视化number line神经元"""
        neurons = number_line_result['number_line_neurons'][:top_n]
        layer_name = number_line_result['layer_name']
        
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
            
            # 数值 vs 神经元响应
            unique_numbers = np.unique(number_line_result.get('target_values', range(1, 11)))
            responses = neuron['response']
            
            # 计算每个数值的平均响应
            avg_responses = []
            std_responses = []
            for num in unique_numbers:
                # 这里需要从原始数据中获取，简化处理
                avg_responses.append(np.mean(responses))  # 简化版本
                std_responses.append(np.std(responses))
            
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
    
    def plot_representational_geometry(self, geometry_result, save_path=None):
        """可视化表征几何学"""
        distance_matrix = geometry_result['distance_matrix']
        unique_numbers = geometry_result['unique_numbers']
        correlation = geometry_result['number_correlation']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 表征距离矩阵
        im = axes[0].imshow(distance_matrix, cmap='viridis')
        axes[0].set_xticks(range(len(unique_numbers)))
        axes[0].set_yticks(range(len(unique_numbers)))
        axes[0].set_xticklabels(unique_numbers)
        axes[0].set_yticklabels(unique_numbers)
        axes[0].set_title('Representational Distance Matrix')
        axes[0].set_xlabel('Number')
        axes[0].set_ylabel('Number')
        plt.colorbar(im, ax=axes[0])
        
        # 2. 数值距离 vs 表征距离
        number_diffs = np.abs(unique_numbers[:, None] - unique_numbers[None, :])
        triu_indices = np.triu_indices(len(unique_numbers), k=1)
        
        x = number_diffs[triu_indices]
        y = distance_matrix[triu_indices]
        
        axes[1].scatter(x, y, alpha=0.6, s=50)
        
        # 添加趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[1].plot(x, p(x), "r--", alpha=0.8)
        
        axes[1].set_xlabel('Numerical Distance')
        axes[1].set_ylabel('Representational Distance')
        axes[1].set_title(f'Number-Representation Correlation\nr = {correlation:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_layer_comparison(self, all_results, metric='proportion', save_path=None):
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


def analyze_number_neurons(features_dict, labels, save_dir, 
                          min_r2=0.5, selectivity_threshold=0.3):
    """
    完整的number neurons分析流程
    
    Args:
        features_dict: {layer_name: features_array}
        labels: 真实标签
        save_dir: 保存目录
        min_r2: number line神经元的最小R²
        selectivity_threshold: 选择性神经元阈值
    """
    
    print("🧠 开始Number Neurons分析...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建分析器
    analyzer = NumberLineAnalyzer(features_dict, labels)
    visualizer = NumberNeuronVisualizer()
    
    all_results = {}
    
    for layer_name in features_dict.keys():
        print(f"\n📊 分析层: {layer_name}")
        
        layer_results = {}
        
        # 1. Number Line分析
        print("🔍 寻找Number Line神经元...")
        number_line_result = analyzer.find_number_line_neurons(
            layer_name, min_r2=min_r2, method='linear'
        )
        layer_results['number_line'] = number_line_result
        
        # 可视化Number Line神经元
        visualizer.plot_number_line_neurons(
            number_line_result,
            save_path=os.path.join(save_dir, f'{layer_name}_number_line_neurons.png')
        )
        
        # 2. Number Selective分析
        print("🔍 寻找Number Selective神经元...")
        selective_result = analyzer.find_number_selective_neurons(
            layer_name, selectivity_threshold=selectivity_threshold
        )
        layer_results['selective'] = selective_result
        
        # 可视化Number Selective神经元
        visualizer.plot_number_selective_neurons(
            selective_result,
            save_path=os.path.join(save_dir, f'{layer_name}_number_selective_neurons.png')
        )
        
        # 3. 表征几何学分析
        print("🔍 分析表征几何学...")
        geometry_result = analyzer.analyze_representational_geometry(layer_name)
        layer_results['geometry'] = geometry_result
        
        # 可视化表征几何学
        visualizer.plot_representational_geometry(
            geometry_result,
            save_path=os.path.join(save_dir, f'{layer_name}_representational_geometry.png')
        )
        
        all_results[layer_name] = layer_results
    
    # 4. 跨层比较
    print("\n📈 生成跨层比较...")
    visualizer.plot_layer_comparison(
        all_results,
        save_path=os.path.join(save_dir, 'layer_comparison.png')
    )
    
    # 5. 生成综合报告
    generate_number_neurons_report(all_results, save_dir)
    
    print(f"\n🎉 Number Neurons分析完成！结果保存在: {save_dir}")
    
    return all_results


def generate_number_neurons_report(all_results, save_dir):
    """生成数值神经元分析报告"""
    
    report_content = f"""# Number Neurons Analysis Report

## 分析概述
本报告分析了模型中的数值表征神经元，包括：
1. **Number Line Neurons**: 神经元响应与数值呈线性关系
2. **Number Selective Neurons**: 对特定数值有选择性响应的神经元
3. **Representational Geometry**: 数值表征的几何结构

## 主要发现

"""
    
    for layer_name, results in all_results.items():
        report_content += f"""
### {layer_name} 层

#### Number Line Neurons
- 总神经元数: {results['number_line']['total_neurons']}
- Number Line神经元: {len(results['number_line']['number_line_neurons'])}
- 比例: {results['number_line']['proportion']:.2%}

#### Number Selective Neurons  
- Number Selective神经元: {len(results['selective']['selective_neurons'])}
- 比例: {results['selective']['proportion']:.2%}

#### 表征几何学
- 数值-表征相关性: {results['geometry']['number_correlation']:.3f}
- p值: {results['geometry']['correlation_p_value']:.3e}

"""
    
    # 跨层统计
    total_number_line = sum(len(r['number_line']['number_line_neurons']) for r in all_results.values())
    total_selective = sum(len(r['selective']['selective_neurons']) for r in all_results.values())
    total_neurons = sum(r['number_line']['total_neurons'] for r in all_results.values())
    
    report_content += f"""
## 总体统计
- 总神经元数: {total_neurons}
- Number Line神经元总数: {total_number_line} ({total_number_line/total_neurons:.2%})
- Number Selective神经元总数: {total_selective} ({total_selective/total_neurons:.2%})

## 解释说明

### Number Line Neurons
这些神经元的响应与数值大小呈现线性关系，类似于大脑中发现的"心理数轴"编码。
高R²值表明神经元能够线性地编码数值信息。

### Number Selective Neurons  
这些神经元对特定数值表现出选择性响应，类似于大脑中的数值选择性细胞。
高选择性指数表明神经元专门响应某个特定数值。

### 表征几何学
分析数值表征在高维空间中的几何结构。正相关表明相近的数值在表征空间中也更接近，
体现了模型学习到的数值拓扑结构。
"""
    
    # 保存报告
    with open(os.path.join(save_dir, 'number_neurons_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)


# 使用示例
def example_usage():
    """使用示例"""
    
    # 假设你已经有了特征数据
    # features_dict = {
    #     'visual_encoder': features_visual,      # [samples, neurons]
    #     'classifier': features_classifier,      # [samples, neurons]
    #     'spatial_attention': features_attention # [samples, neurons]
    # }
    # labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]  # 真实数值标签
    
    # 运行分析
    # results = analyze_number_neurons(
    #     features_dict=features_dict,
    #     labels=labels,
    #     save_dir='./number_neurons_analysis',
    #     min_r2=0.5,
    #     selectivity_threshold=0.3
    # )
    
    print("请参考上面的注释代码来运行number neurons分析")


if __name__ == "__main__":
    example_usage()