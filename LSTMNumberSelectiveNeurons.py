"""
LSTM Number Selective Neurons and Tuning Curve Analysis Tool
LSTM数值选择性神经元和调谐曲线分析工具
支持原始Embodiment模型和所有Ablation模型
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, entropy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
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

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LSTMFeatureExtractor:
    """LSTM特征提取器 - 专门提取LSTM隐状态"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.lstm_states_history = []
        self.model_info = self._detect_model_type()
        
        print(f"✅ 检测到模型类型: {self.model_info['type']}")
        print(f"   LSTM层: {self.model_info['has_lstm']}")
        
    def _detect_model_type(self):
        """检测模型类型"""
        model_class_name = self.model.__class__.__name__
        has_lstm = hasattr(self.model, 'lstm')
        
        if hasattr(self.model, 'get_model_info'):
            info = self.model.get_model_info()
            model_type = info.get('model_type', model_class_name)
        else:
            model_type = 'EmbodiedCountingModel'
        
        return {'type': model_type, 'has_lstm': has_lstm}
    
    def _hook_lstm_states(self):
        """钩子函数来捕获LSTM隐状态"""
        def lstm_hook(module, input, output):
            # LSTM输出格式: (output, (h_n, c_n))
            # output: [batch, seq_len, hidden_size]
            # h_n: [num_layers, batch, hidden_size]
            lstm_output, (h_n, c_n) = output
            
            # 保存每个时间步的隐状态
            batch_size, seq_len, hidden_size = lstm_output.shape
            
            # 转换为 [seq_len, batch, hidden_size] 然后保存每个时间步
            lstm_output_transposed = lstm_output.transpose(0, 1)  # [seq_len, batch, hidden_size]
            
            for t in range(seq_len):
                timestep_hidden = lstm_output_transposed[t].detach().cpu()  # [batch, hidden_size]
                self.lstm_states_history.append({
                    'timestep': t,
                    'hidden_states': timestep_hidden,
                    'batch_size': batch_size
                })
        
        # 注册钩子
        if hasattr(self.model, 'lstm'):
            handle = self.model.lstm.register_forward_hook(lstm_hook)
            return handle
        else:
            print("❌ 模型没有LSTM层")
            return None
    
    def extract_lstm_features(self, data_loader, max_samples=500):
        """提取LSTM特征"""
        print("🧠 开始提取LSTM隐状态特征...")
        
        # 注册LSTM钩子
        handle = self._hook_lstm_states()
        if not handle:
            return None
        
        all_labels = []
        all_sample_ids = []
        sample_count = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="提取LSTM隐状态"):
                    if sample_count >= max_samples:
                        break
                    
                    # 清空历史记录
                    self.lstm_states_history = []
                    
                    # 准备数据
                    sequence_data = {
                        'images': batch['sequence_data']['images'].to(self.device),
                        'joints': batch['sequence_data']['joints'].to(self.device),
                        'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                        'labels': batch['sequence_data']['labels'].to(self.device)
                    }
                    
                    # 获取批次信息
                    labels = batch['label'].cpu().numpy()
                    sample_ids = batch['sample_id']
                    
                    # 计算实际处理的样本数
                    remaining_samples = max_samples - sample_count
                    actual_batch_size = min(len(labels), remaining_samples)
                    
                    # 截断批次
                    if actual_batch_size < len(labels):
                        for key in sequence_data:
                            sequence_data[key] = sequence_data[key][:actual_batch_size]
                        labels = labels[:actual_batch_size]
                        sample_ids = sample_ids[:actual_batch_size]
                    
                    # 前向传播 - 根据模型类型
                    if self.model_info['type'] in ['EmbodiedCountingOnly', 'VisualOnlyCountingModel']:
                        outputs = self.model(sequence_data=sequence_data)
                    else:
                        outputs = self.model(
                            sequence_data=sequence_data,
                            use_teacher_forcing=False
                        )
                    
                    # 收集标签
                    all_labels.extend(labels)
                    all_sample_ids.extend(sample_ids)
                    
                    sample_count += actual_batch_size
                    
                    if sample_count >= max_samples:
                        break
            
            # 处理收集到的LSTM状态
            lstm_data = self._process_lstm_states(all_labels, all_sample_ids)
            return lstm_data
            
        finally:
            # 移除钩子
            if handle:
                handle.remove()
    
    def _process_lstm_states(self, all_labels, all_sample_ids):
        """处理收集到的LSTM状态"""
        print("🔄 处理LSTM状态数据...")
        
        if not self.lstm_states_history:
            print("❌ 没有收集到LSTM状态")
            return None
        
        # 按时间步组织数据
        timestep_data = defaultdict(list)
        
        current_sample_idx = 0
        
        for state_info in self.lstm_states_history:
            timestep = state_info['timestep']
            hidden_states = state_info['hidden_states']  # [batch, hidden_size]
            batch_size = state_info['batch_size']
            
            # 为当前批次的每个样本分配标签
            for b in range(batch_size):
                if current_sample_idx < len(all_labels):
                    sample_label = all_labels[current_sample_idx]
                    sample_id = all_sample_ids[current_sample_idx]
                    
                    timestep_data[timestep].append({
                        'sample_id': sample_id,
                        'label': sample_label,
                        'hidden_states': hidden_states[b].numpy(),
                        'sample_idx': current_sample_idx
                    })
                    
                    if timestep == 0:  # 只在第一个时间步增加样本索引
                        current_sample_idx += 1
        
        # 转换为最终格式
        final_data = {}
        for timestep, data_list in timestep_data.items():
            if data_list:
                labels = np.array([d['label'] for d in data_list])
                hidden_states = np.stack([d['hidden_states'] for d in data_list])
                sample_ids = [d['sample_id'] for d in data_list]
                
                final_data[f'timestep_{timestep}'] = {
                    'labels': labels,
                    'hidden_states': hidden_states,  # [samples, hidden_size]
                    'sample_ids': sample_ids
                }
        
        print(f"✅ LSTM特征提取完成:")
        print(f"   时间步数: {len(final_data)}")
        print(f"   样本数: {len(all_labels)}")
        if final_data:
            first_timestep = list(final_data.keys())[0]
            hidden_size = final_data[first_timestep]['hidden_states'].shape[1]
            print(f"   LSTM隐状态维度: {hidden_size}")
        
        return final_data


class LSTMNumberAnalyzer:
    """LSTM数值神经元分析器"""
    
    def __init__(self, lstm_data):
        """
        Args:
            lstm_data: {timestep: {labels, hidden_states, sample_ids}}
        """
        self.lstm_data = lstm_data
        self.timesteps = sorted(lstm_data.keys())
        self.results = {}
        
        # 获取基本信息
        first_timestep = self.timesteps[0]
        self.hidden_size = lstm_data[first_timestep]['hidden_states'].shape[1]
        self.unique_numbers = np.unique(lstm_data[first_timestep]['labels'])
        
        print(f"📊 LSTM数值分析器初始化:")
        print(f"   时间步数: {len(self.timesteps)}")
        print(f"   LSTM隐状态维度: {self.hidden_size}")
        print(f"   数值范围: {self.unique_numbers}")
    
    def find_number_line_neurons(self, timestep_key, min_r2=0.5, method='linear'):
        """寻找具有number line特性的LSTM神经元"""
        if timestep_key not in self.lstm_data:
            print(f"❌ 时间步 {timestep_key} 不存在")
            return None
        
        data = self.lstm_data[timestep_key]
        hidden_states = data['hidden_states']  # [samples, hidden_size]
        labels = data['labels']
        
        if method == 'linear':
            target = labels
        elif method == 'log':
            target = np.log(labels)
        elif method == 'sqrt':
            target = np.sqrt(labels)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        number_line_neurons = []
        
        print(f"🔍 分析 {timestep_key} 的 {self.hidden_size} 个LSTM神经元...")
        
        for neuron_idx in tqdm(range(self.hidden_size), desc="寻找number line神经元"):
            neuron_response = hidden_states[:, neuron_idx]
            
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
                    'target_values': target.copy(),
                    'labels': labels.copy()
                })
        
        # 按R²分数排序
        number_line_neurons.sort(key=lambda x: x['r2_score'], reverse=True)
        
        result = {
            'timestep': timestep_key,
            'method': method,
            'total_neurons': self.hidden_size,
            'number_line_neurons': number_line_neurons,
            'proportion': len(number_line_neurons) / self.hidden_size
        }
        
        print(f"✅ 找到 {len(number_line_neurons)} 个number line神经元 "
              f"({result['proportion']:.2%})")
        
        return result
    
    def find_number_selective_neurons(self, timestep_key, selectivity_threshold=0.3):
        """寻找数值选择性LSTM神经元"""
        if timestep_key not in self.lstm_data:
            print(f"❌ 时间步 {timestep_key} 不存在")
            return None
        
        data = self.lstm_data[timestep_key]
        hidden_states = data['hidden_states']
        labels = data['labels']
        
        # 计算每个神经元对每个数值的平均响应
        response_matrix = np.zeros((len(self.unique_numbers), self.hidden_size))
        response_std_matrix = np.zeros((len(self.unique_numbers), self.hidden_size))
        
        for i, num in enumerate(self.unique_numbers):
            mask = labels == num
            if np.sum(mask) > 0:
                response_matrix[i, :] = np.mean(hidden_states[mask, :], axis=0)
                response_std_matrix[i, :] = np.std(hidden_states[mask, :], axis=0)
        
        selective_neurons = []
        
        print(f"🔍 分析 {timestep_key} 的数值选择性...")
        
        for neuron_idx in tqdm(range(self.hidden_size), desc="计算选择性"):
            responses = response_matrix[:, neuron_idx]
            response_stds = response_std_matrix[:, neuron_idx]
            
            # 计算选择性指数
            max_response = np.max(responses)
            min_response = np.min(responses)
            mean_response = np.mean(responses)
            
            if max_response != min_response:
                selectivity_index = (max_response - min_response) / (max_response + min_response + 1e-8)
            else:
                selectivity_index = 0
            
            # 计算信噪比
            if mean_response != 0:
                signal_to_noise = (max_response - mean_response) / (np.mean(response_stds) + 1e-8)
            else:
                signal_to_noise = 0
            
            # 找到最佳数值
            preferred_number = self.unique_numbers[np.argmax(responses)]
            preferred_idx = np.argmax(responses)
            
            # 计算调谐曲线特性
            tuning_properties = self._analyze_tuning_curve(responses, self.unique_numbers)
            
            # 计算响应的熵（作为选择性的另一种度量）
            response_prob = responses / (np.sum(responses) + 1e-8)
            response_entropy = entropy(response_prob + 1e-8)
            
            if selectivity_index >= selectivity_threshold:
                selective_neurons.append({
                    'neuron_idx': neuron_idx,
                    'selectivity_index': selectivity_index,
                    'preferred_number': preferred_number,
                    'preferred_idx': preferred_idx,
                    'response_profile': responses.copy(),
                    'response_stds': response_stds.copy(),
                    'max_response': max_response,
                    'min_response': min_response,
                    'response_ratio': max_response / (min_response + 1e-8),
                    'signal_to_noise': signal_to_noise,
                    'response_entropy': response_entropy,
                    **tuning_properties
                })
        
        # 按选择性指数排序
        selective_neurons.sort(key=lambda x: x['selectivity_index'], reverse=True)
        
        result = {
            'timestep': timestep_key,
            'total_neurons': self.hidden_size,
            'selective_neurons': selective_neurons,
            'proportion': len(selective_neurons) / self.hidden_size,
            'unique_numbers': self.unique_numbers,
            'response_matrix': response_matrix,
            'response_std_matrix': response_std_matrix
        }
        
        print(f"✅ 找到 {len(selective_neurons)} 个数值选择性神经元 "
              f"({result['proportion']:.2%})")
        
        return result
    
    def _analyze_tuning_curve(self, responses, numbers):
        """分析调谐曲线特性"""
        # 标准化响应
        responses_norm = (responses - np.min(responses)) / (np.max(responses) - np.min(responses) + 1e-8)
        
        # 计算调谐宽度（半高宽度）
        max_idx = np.argmax(responses_norm)
        half_max = responses_norm[max_idx] / 2
        
        # 找到半高点
        left_indices = np.where(responses_norm[:max_idx] <= half_max)[0]
        right_indices = np.where(responses_norm[max_idx:] <= half_max)[0]
        
        if len(left_indices) > 0 and len(right_indices) > 0:
            left_bound = left_indices[-1]
            right_bound = max_idx + right_indices[0]
            tuning_width = numbers[right_bound] - numbers[left_bound]
        else:
            tuning_width = len(numbers)
        
        # 计算调谐曲线的锐度
        peak_sharpness = responses_norm[max_idx] / (np.mean(responses_norm) + 1e-8)
        
        # 检测是否有多个峰
        peaks, _ = find_peaks(responses_norm, height=0.3)
        num_peaks = len(peaks)
        
        # 计算调谐曲线的偏斜度
        mean_pos = np.sum(numbers * responses_norm) / (np.sum(responses_norm) + 1e-8)
        preferred_pos = numbers[max_idx]
        skewness = mean_pos - preferred_pos
        
        return {
            'tuning_width': tuning_width,
            'peak_sharpness': peak_sharpness,
            'num_peaks': num_peaks,
            'skewness': skewness,
            'peak_position': preferred_pos
        }
    
    def analyze_temporal_dynamics(self, neuron_idx, analysis_type='selective'):
        """分析特定神经元在时间维度上的动态变化"""
        temporal_data = []
        
        for timestep_key in self.timesteps:
            data = self.lstm_data[timestep_key]
            hidden_states = data['hidden_states']
            labels = data['labels']
            
            if neuron_idx >= hidden_states.shape[1]:
                continue
            
            neuron_response = hidden_states[:, neuron_idx]
            
            if analysis_type == 'selective':
                # 计算选择性
                response_by_number = []
                for num in self.unique_numbers:
                    mask = labels == num
                    if np.sum(mask) > 0:
                        response_by_number.append(np.mean(neuron_response[mask]))
                    else:
                        response_by_number.append(0)
                
                responses = np.array(response_by_number)
                max_resp = np.max(responses)
                min_resp = np.min(responses)
                selectivity = (max_resp - min_resp) / (max_resp + min_resp + 1e-8)
                preferred_number = self.unique_numbers[np.argmax(responses)]
                
                temporal_data.append({
                    'timestep': int(timestep_key.split('_')[1]),
                    'selectivity': selectivity,
                    'preferred_number': preferred_number,
                    'max_response': max_resp,
                    'response_profile': responses
                })
            
            elif analysis_type == 'number_line':
                # 计算线性拟合质量
                reg = LinearRegression()
                reg.fit(labels.reshape(-1, 1), neuron_response)
                predicted = reg.predict(labels.reshape(-1, 1))
                r2 = r2_score(neuron_response, predicted)
                correlation, _ = pearsonr(labels, neuron_response)
                
                temporal_data.append({
                    'timestep': int(timestep_key.split('_')[1]),
                    'r2_score': r2,
                    'correlation': correlation,
                    'slope': reg.coef_[0],
                    'intercept': reg.intercept_
                })
        
        return temporal_data
    
    def find_best_timesteps_for_analysis(self, analysis_type='selective', top_k=3):
        """找到最适合分析的时间步"""
        timestep_scores = []
        
        for timestep_key in self.timesteps:
            if analysis_type == 'selective':
                result = self.find_number_selective_neurons(timestep_key, selectivity_threshold=0.1)
                if result:
                    score = result['proportion']
                    avg_selectivity = np.mean([n['selectivity_index'] for n in result['selective_neurons']]) if result['selective_neurons'] else 0
                    timestep_scores.append((timestep_key, score, avg_selectivity))
            
            elif analysis_type == 'number_line':
                result = self.find_number_line_neurons(timestep_key, min_r2=0.3)
                if result:
                    score = result['proportion']
                    avg_r2 = np.mean([n['r2_score'] for n in result['number_line_neurons']]) if result['number_line_neurons'] else 0
                    timestep_scores.append((timestep_key, score, avg_r2))
        
        # 按综合分数排序
        timestep_scores.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        return timestep_scores[:top_k]


class LSTMVisualizationEngine:
    """LSTM可视化引擎"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_number_line_neurons(self, number_line_result, save_path=None, top_n=6):
        """可视化LSTM number line神经元"""
        neurons = number_line_result['number_line_neurons'][:top_n]
        timestep = number_line_result['timestep']
        
        if not neurons:
            print(f"⚠️ {timestep} 没有找到number line神经元")
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
            
            # 计算每个数值的平均响应和标准差
            unique_numbers = np.unique(neuron['labels'])
            avg_responses = []
            std_responses = []
            
            for num in unique_numbers:
                mask = neuron['labels'] == num
                if np.sum(mask) > 0:
                    responses = neuron['response'][mask]
                    avg_responses.append(np.mean(responses))
                    std_responses.append(np.std(responses))
                else:
                    avg_responses.append(0)
                    std_responses.append(0)
            
            # 绘制数据点和误差条
            ax.errorbar(unique_numbers, avg_responses, yerr=std_responses, 
                       marker='o', capsize=5, linewidth=2, markersize=8, 
                       color='blue', alpha=0.7, label='Data')
            
            # 添加拟合线
            target_values = neuron['target_values']
            if len(np.unique(target_values)) > 1:
                reg_line = neuron['slope'] * unique_numbers + neuron['intercept']
                ax.plot(unique_numbers, reg_line, '--', color='red', linewidth=2, 
                       label=f'Fit (slope={neuron["slope"]:.3f})')
            
            ax.set_title(f'LSTM Neuron {neuron["neuron_idx"]}\n'
                        f'R² = {neuron["r2_score"]:.3f}, r = {neuron["correlation"]:.3f}')
            ax.set_xlabel('Number')
            ax.set_ylabel('LSTM Hidden State')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(neurons), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'LSTM Number Line Neurons - {timestep}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存Number Line神经元图: {os.path.basename(save_path) if save_path else 'displayed'}")
    
    def plot_number_selective_neurons(self, selective_result, save_path=None, top_n=6):
        """可视化LSTM数值选择性神经元"""
        neurons = selective_result['selective_neurons'][:top_n]
        timestep = selective_result['timestep']
        unique_numbers = selective_result['unique_numbers']
        
        if not neurons:
            print(f"⚠️ {timestep} 没有找到number selective神经元")
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
            response_stds = neuron['response_stds']
            
            # 绘制柱状图和误差条
            bars = ax.bar(unique_numbers, responses, alpha=0.7, 
                         color='skyblue', edgecolor='navy', linewidth=1,
                         yerr=response_stds, capsize=3)
            
            # 标记偏好数值
            preferred_idx = neuron['preferred_idx']
            bars[preferred_idx].set_color('red')
            bars[preferred_idx].set_alpha(0.9)
            
            # 添加调谐曲线信息
            info_text = f"Width: {neuron['tuning_width']:.1f}\n"
            info_text += f"Peaks: {neuron['num_peaks']}\n"
            info_text += f"SNR: {neuron['signal_to_noise']:.2f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=8)
            
            ax.set_title(f'LSTM Neuron {neuron["neuron_idx"]}\n'
                        f'Selectivity = {neuron["selectivity_index"]:.3f}\n'
                        f'Preferred: {neuron["preferred_number"]}')
            ax.set_xlabel('Number')
            ax.set_ylabel('Average LSTM Response')
            ax.grid(True, alpha=0.3)
            
            # 设置x轴
            ax.set_xticks(unique_numbers)
        
        # 隐藏多余的子图
        for i in range(len(neurons), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'LSTM Number Selective Neurons - {timestep}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存Number Selective神经元图: {os.path.basename(save_path) if save_path else 'displayed'}")
    
    def plot_temporal_dynamics(self, temporal_data, neuron_idx, analysis_type, save_path=None):
        """可视化神经元的时间动态"""
        if not temporal_data:
            print("⚠️ 没有时间动态数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        timesteps = [d['timestep'] for d in temporal_data]
        
        if analysis_type == 'selective':
            # 选择性随时间变化
            selectivities = [d['selectivity'] for d in temporal_data]
            preferred_numbers = [d['preferred_number'] for d in temporal_data]
            max_responses = [d['max_response'] for d in temporal_data]
            
            # 1. 选择性指数随时间变化
            axes[0, 0].plot(timesteps, selectivities, 'o-', linewidth=2, markersize=6)
            axes[0, 0].set_title(f'Selectivity Over Time\nNeuron {neuron_idx}')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Selectivity Index')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 偏好数值随时间变化
            axes[0, 1].plot(timesteps, preferred_numbers, 'o-', linewidth=2, markersize=6, color='red')
            axes[0, 1].set_title('Preferred Number Over Time')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Preferred Number')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yticks(range(1, 11))
            
            # 3. 最大响应强度随时间变化
            axes[1, 0].plot(timesteps, max_responses, 'o-', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_title('Maximum Response Over Time')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Max Response')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 响应轮廓的热力图
            response_profiles = np.array([d['response_profile'] for d in temporal_data])
            unique_numbers = np.arange(1, 11)
            
            im = axes[1, 1].imshow(response_profiles.T, aspect='auto', cmap='viridis', 
                                  extent=[min(timesteps), max(timesteps), 
                                         min(unique_numbers), max(unique_numbers)])
            axes[1, 1].set_title('Response Profile Heatmap')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Number')
            axes[1, 1].set_yticks(unique_numbers)
            plt.colorbar(im, ax=axes[1, 1], label='Response')
            
        elif analysis_type == 'number_line':
            # Number line特性随时间变化
            r2_scores = [d['r2_score'] for d in temporal_data]
            correlations = [d['correlation'] for d in temporal_data]
            slopes = [d['slope'] for d in temporal_data]
            intercepts = [d['intercept'] for d in temporal_data]
            
            # 1. R²分数随时间变化
            axes[0, 0].plot(timesteps, r2_scores, 'o-', linewidth=2, markersize=6)
            axes[0, 0].set_title(f'R² Score Over Time\nNeuron {neuron_idx}')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            # 2. 相关系数随时间变化
            axes[0, 1].plot(timesteps, correlations, 'o-', linewidth=2, markersize=6, color='red')
            axes[0, 1].set_title('Correlation Over Time')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(-1, 1)
            
            # 3. 斜率随时间变化
            axes[1, 0].plot(timesteps, slopes, 'o-', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_title('Slope Over Time')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Slope')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 截距随时间变化
            axes[1, 1].plot(timesteps, intercepts, 'o-', linewidth=2, markersize=6, color='purple')
            axes[1, 1].set_title('Intercept Over Time')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Intercept')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存时间动态图: {os.path.basename(save_path) if save_path else 'displayed'}")
    
    def plot_timestep_comparison(self, analyzer, analysis_type='selective', save_path=None):
        """对比不同时间步的神经元特性"""
        timestep_data = []
        
        for timestep_key in analyzer.timesteps:
            if analysis_type == 'selective':
                result = analyzer.find_number_selective_neurons(timestep_key, selectivity_threshold=0.1)
                if result:
                    proportion = result['proportion']
                    avg_selectivity = np.mean([n['selectivity_index'] for n in result['selective_neurons']]) if result['selective_neurons'] else 0
                    max_selectivity = max([n['selectivity_index'] for n in result['selective_neurons']]) if result['selective_neurons'] else 0
                    
                    timestep_data.append({
                        'timestep': int(timestep_key.split('_')[1]),
                        'proportion': proportion,
                        'avg_selectivity': avg_selectivity,
                        'max_selectivity': max_selectivity
                    })
            
            elif analysis_type == 'number_line':
                result = analyzer.find_number_line_neurons(timestep_key, min_r2=0.3)
                if result:
                    proportion = result['proportion']
                    avg_r2 = np.mean([n['r2_score'] for n in result['number_line_neurons']]) if result['number_line_neurons'] else 0
                    max_r2 = max([n['r2_score'] for n in result['number_line_neurons']]) if result['number_line_neurons'] else 0
                    
                    timestep_data.append({
                        'timestep': int(timestep_key.split('_')[1]),
                        'proportion': proportion,
                        'avg_r2': avg_r2,
                        'max_r2': max_r2
                    })
        
        if not timestep_data:
            print("⚠️ 没有时间步数据")
            return
        
        timesteps = [d['timestep'] for d in timestep_data]
        proportions = [d['proportion'] for d in timestep_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 神经元比例随时间变化
        axes[0, 0].plot(timesteps, proportions, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_title(f'{analysis_type.title()} Neuron Proportion Over Time')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Proportion of Neurons')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, max(proportions) * 1.1 if proportions else 0.1)
        
        if analysis_type == 'selective':
            avg_selectivities = [d['avg_selectivity'] for d in timestep_data]
            max_selectivities = [d['max_selectivity'] for d in timestep_data]
            
            # 2. 平均选择性
            axes[0, 1].plot(timesteps, avg_selectivities, 'o-', linewidth=2, markersize=8, color='red')
            axes[0, 1].set_title('Average Selectivity Over Time')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Average Selectivity')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 最大选择性
            axes[1, 0].plot(timesteps, max_selectivities, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 0].set_title('Maximum Selectivity Over Time')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Maximum Selectivity')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 综合分析
            combined_scores = [p * a for p, a in zip(proportions, avg_selectivities)]
            axes[1, 1].plot(timesteps, combined_scores, 'o-', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_title('Combined Score (Proportion × Avg Selectivity)')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Combined Score')
            axes[1, 1].grid(True, alpha=0.3)
            
        elif analysis_type == 'number_line':
            avg_r2s = [d['avg_r2'] for d in timestep_data]
            max_r2s = [d['max_r2'] for d in timestep_data]
            
            # 2. 平均R²
            axes[0, 1].plot(timesteps, avg_r2s, 'o-', linewidth=2, markersize=8, color='red')
            axes[0, 1].set_title('Average R² Over Time')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Average R²')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # 3. 最大R²
            axes[1, 0].plot(timesteps, max_r2s, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 0].set_title('Maximum R² Over Time')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Maximum R²')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1)
            
            # 4. 综合分析
            combined_scores = [p * r for p, r in zip(proportions, avg_r2s)]
            axes[1, 1].plot(timesteps, combined_scores, 'o-', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_title('Combined Score (Proportion × Avg R²)')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Combined Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存时间步对比图: {os.path.basename(save_path) if save_path else 'displayed'}")


def load_model_and_data(checkpoint_path, val_csv, data_root, batch_size=8):
    """通用模型和数据加载函数"""
    print("📥 加载模型和数据...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 确定图像模式
    image_mode = config.get('image_mode', 'rgb')
    
    # 检查模型类型
    model_type = checkpoint.get('model_type', 'embodied')
    
    if model_type in ['counting_only', 'visual_only']:
        # Ablation模型
        from Model_embodiment_ablation import create_ablation_model
        model = create_ablation_model(model_type, config)
        print(f"✅ 加载消融实验模型: {model_type}")
    else:
        # 原始Embodiment模型
        from Model_embodiment import EmbodiedCountingModel
        input_channels = 3 if image_mode == 'rgb' else 1
        model_config = config['model_config'].copy()
        model_config['input_channels'] = input_channels
        model = EmbodiedCountingModel(**model_config)
        print("✅ 加载原始具身计数模型")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"   图像模式: {image_mode}, 设备: {device}")
    
    # 创建数据加载器
    from DataLoader_embodiment import get_ball_counting_data_loaders
    
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


def analyze_lstm_number_neurons(checkpoint_path, val_csv, data_root, 
                               save_dir='./lstm_number_analysis', 
                               max_samples=500, 
                               min_r2=0.5, selectivity_threshold=0.3,
                               analyze_temporal=True):
    """LSTM数值神经元分析主函数"""
    
    print("🧠 开始LSTM数值神经元分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建LSTM特征提取器
        extractor = LSTMFeatureExtractor(model, device)
        
        if not extractor.model_info['has_lstm']:
            print("❌ 模型没有LSTM层，无法分析")
            return None
        
        # 3. 提取LSTM特征
        lstm_data = extractor.extract_lstm_features(val_loader, max_samples)
        
        if not lstm_data:
            print("❌ LSTM特征提取失败")
            return None
        
        # 4. 创建分析器
        analyzer = LSTMNumberAnalyzer(lstm_data)
        
        # 5. 创建可视化引擎
        visualizer = LSTMVisualizationEngine()
        
        # 6. 找到最佳分析时间步
        print("🔍 寻找最佳分析时间步...")
        best_timesteps_selective = analyzer.find_best_timesteps_for_analysis('selective', top_k=3)
        best_timesteps_number_line = analyzer.find_best_timesteps_for_analysis('number_line', top_k=3)
        
        print(f"✅ 最佳选择性时间步: {[t[0] for t in best_timesteps_selective]}")
        print(f"✅ 最佳Number Line时间步: {[t[0] for t in best_timesteps_number_line]}")
        
        analysis_results = {}
        
        # 7. 分析关键时间步
        key_timesteps = set([t[0] for t in best_timesteps_selective] + 
                           [t[0] for t in best_timesteps_number_line])
        
        for timestep_key in sorted(key_timesteps):
            print(f"\n📊 分析时间步: {timestep_key}")
            
            timestep_results = {}
            
            # Number Line分析
            print("🔍 Number Line神经元分析...")
            number_line_result = analyzer.find_number_line_neurons(
                timestep_key, min_r2=min_r2, method='linear'
            )
            
            if number_line_result:
                timestep_results['number_line'] = number_line_result
                
                # 可视化
                visualizer.plot_number_line_neurons(
                    number_line_result,
                    save_path=os.path.join(save_dir, f'{timestep_key}_number_line_neurons.png')
                )
            
            # Number Selective分析
            print("🔍 Number Selective神经元分析...")
            selective_result = analyzer.find_number_selective_neurons(
                timestep_key, selectivity_threshold=selectivity_threshold
            )
            
            if selective_result:
                timestep_results['selective'] = selective_result
                
                # 可视化
                visualizer.plot_number_selective_neurons(
                    selective_result,
                    save_path=os.path.join(save_dir, f'{timestep_key}_number_selective_neurons.png')
                )
            
            analysis_results[timestep_key] = timestep_results
        
        # 8. 时间步对比分析
        print("\n📈 生成时间步对比分析...")
        visualizer.plot_timestep_comparison(
            analyzer, 'selective',
            save_path=os.path.join(save_dir, 'timestep_comparison_selective.png')
        )
        
        visualizer.plot_timestep_comparison(
            analyzer, 'number_line',
            save_path=os.path.join(save_dir, 'timestep_comparison_number_line.png')
        )
        
        # 9. 时间动态分析（可选）
        if analyze_temporal:
            print("\n⏱️ 分析选定神经元的时间动态...")
            
            # 选择几个有代表性的神经元进行时间动态分析
            for timestep_key in list(key_timesteps)[:2]:  # 只分析前两个时间步
                if timestep_key in analysis_results:
                    results = analysis_results[timestep_key]
                    
                    # 选择性神经元时间动态
                    if 'selective' in results and results['selective']['selective_neurons']:
                        best_selective_neuron = results['selective']['selective_neurons'][0]
                        neuron_idx = best_selective_neuron['neuron_idx']
                        
                        temporal_data = analyzer.analyze_temporal_dynamics(neuron_idx, 'selective')
                        if temporal_data:
                            visualizer.plot_temporal_dynamics(
                                temporal_data, neuron_idx, 'selective',
                                save_path=os.path.join(save_dir, f'temporal_dynamics_selective_neuron_{neuron_idx}.png')
                            )
                    
                    # Number line神经元时间动态
                    if 'number_line' in results and results['number_line']['number_line_neurons']:
                        best_nl_neuron = results['number_line']['number_line_neurons'][0]
                        neuron_idx = best_nl_neuron['neuron_idx']
                        
                        temporal_data = analyzer.analyze_temporal_dynamics(neuron_idx, 'number_line')
                        if temporal_data:
                            visualizer.plot_temporal_dynamics(
                                temporal_data, neuron_idx, 'number_line',
                                save_path=os.path.join(save_dir, f'temporal_dynamics_number_line_neuron_{neuron_idx}.png')
                            )
        
        # 10. 生成分析报告
        print("📝 生成分析报告...")
        generate_lstm_analysis_report(analysis_results, analyzer, config, save_dir)
        
        print(f"🎉 LSTM数值神经元分析完成！")
        print(f"📁 结果保存在: {save_dir}")
        
        return {
            'lstm_data': lstm_data,
            'analysis_results': analysis_results,
            'analyzer': analyzer,
            'config': config
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_lstm_analysis_report(analysis_results, analyzer, config, save_dir):
    """生成LSTM分析报告"""
    
    # 统计信息
    total_timesteps = len(analyzer.timesteps)
    analyzed_timesteps = len(analysis_results)
    hidden_size = analyzer.hidden_size
    unique_numbers = analyzer.unique_numbers
    
    # 收集统计数据
    timestep_stats = {}
    overall_stats = {
        'number_line_neurons': [],
        'selective_neurons': [],
        'total_number_line': 0,
        'total_selective': 0
    }
    
    for timestep_key, results in analysis_results.items():
        timestep_stat = {'timestep': timestep_key}
        
        if 'number_line' in results:
            nl_result = results['number_line']
            timestep_stat['number_line'] = {
                'count': len(nl_result['number_line_neurons']),
                'proportion': nl_result['proportion'],
                'best_r2': max([n['r2_score'] for n in nl_result['number_line_neurons']], default=0),
                'avg_r2': np.mean([n['r2_score'] for n in nl_result['number_line_neurons']]) if nl_result['number_line_neurons'] else 0
            }
            overall_stats['total_number_line'] += len(nl_result['number_line_neurons'])
            overall_stats['number_line_neurons'].extend([n['r2_score'] for n in nl_result['number_line_neurons']])
        
        if 'selective' in results:
            sel_result = results['selective']
            timestep_stat['selective'] = {
                'count': len(sel_result['selective_neurons']),
                'proportion': sel_result['proportion'],
                'best_selectivity': max([n['selectivity_index'] for n in sel_result['selective_neurons']], default=0),
                'avg_selectivity': np.mean([n['selectivity_index'] for n in sel_result['selective_neurons']]) if sel_result['selective_neurons'] else 0
            }
            overall_stats['total_selective'] += len(sel_result['selective_neurons'])
            overall_stats['selective_neurons'].extend([n['selectivity_index'] for n in sel_result['selective_neurons']])
        
        timestep_stats[timestep_key] = timestep_stat
    
    # 计算整体统计
    if overall_stats['number_line_neurons']:
        overall_stats['avg_r2_all'] = np.mean(overall_stats['number_line_neurons'])
        overall_stats['max_r2_all'] = max(overall_stats['number_line_neurons'])
    else:
        overall_stats['avg_r2_all'] = 0
        overall_stats['max_r2_all'] = 0
    
    if overall_stats['selective_neurons']:
        overall_stats['avg_selectivity_all'] = np.mean(overall_stats['selective_neurons'])
        overall_stats['max_selectivity_all'] = max(overall_stats['selective_neurons'])
    else:
        overall_stats['avg_selectivity_all'] = 0
        overall_stats['max_selectivity_all'] = 0
    
    # 生成完整报告
    report = {
        'analysis_type': 'LSTM Number Neurons Analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_config': {
            'model_type': config.get('model_type', 'unknown'),
            'image_mode': config.get('image_mode', 'rgb'),
            'sequence_length': config.get('sequence_length', 'unknown'),
            'lstm_hidden_size': hidden_size
        },
        'analysis_summary': {
            'total_timesteps': total_timesteps,
            'analyzed_timesteps': analyzed_timesteps,
            'lstm_hidden_size': hidden_size,
            'number_range': f"{min(unique_numbers)}-{max(unique_numbers)}",
            'total_number_line_neurons': overall_stats['total_number_line'],
            'total_selective_neurons': overall_stats['total_selective'],
            'avg_r2_all_timesteps': overall_stats['avg_r2_all'],
            'max_r2_all_timesteps': overall_stats['max_r2_all'],
            'avg_selectivity_all_timesteps': overall_stats['avg_selectivity_all'],
            'max_selectivity_all_timesteps': overall_stats['max_selectivity_all']
        },
        'timestep_details': timestep_stats
    }
    
    # 保存JSON报告
    try:
        with open(os.path.join(save_dir, 'lstm_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ JSON报告已保存")
    except Exception as e:
        print(f"⚠️ JSON报告保存失败: {e}")
    
    # 生成可读报告
    try:
        with open(os.path.join(save_dir, 'lstm_analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== LSTM数值神经元分析报告 ===\n\n")
            f.write(f"分析时间: {report['timestamp']}\n")
            f.write(f"模型类型: {report['model_config']['model_type']}\n")
            f.write(f"LSTM隐状态维度: {hidden_size}\n")
            f.write(f"数值范围: {report['analysis_summary']['number_range']}\n")
            f.write(f"总时间步数: {total_timesteps}\n")
            f.write(f"分析时间步数: {analyzed_timesteps}\n\n")
            
            f.write("=== 整体统计 ===\n")
            f.write(f"总Number Line神经元: {overall_stats['total_number_line']}\n")
            f.write(f"总Number Selective神经元: {overall_stats['total_selective']}\n")
            f.write(f"平均R²分数: {overall_stats['avg_r2_all']:.3f}\n")
            f.write(f"最大R²分数: {overall_stats['max_r2_all']:.3f}\n")
            f.write(f"平均选择性指数: {overall_stats['avg_selectivity_all']:.3f}\n")
            f.write(f"最大选择性指数: {overall_stats['max_selectivity_all']:.3f}\n\n")
            
            f.write("=== 各时间步详情 ===\n")
            for timestep_key, stats in timestep_stats.items():
                f.write(f"\n{timestep_key}:\n")
                
                if 'number_line' in stats:
                    nl = stats['number_line']
                    f.write(f"  Number Line神经元:\n")
                    f.write(f"    数量: {nl['count']}\n")
                    f.write(f"    比例: {nl['proportion']:.2%}\n")
                    f.write(f"    最佳R²: {nl['best_r2']:.3f}\n")
                    f.write(f"    平均R²: {nl['avg_r2']:.3f}\n")
                
                if 'selective' in stats:
                    sel = stats['selective']
                    f.write(f"  Number Selective神经元:\n")
                    f.write(f"    数量: {sel['count']}\n")
                    f.write(f"    比例: {sel['proportion']:.2%}\n")
                    f.write(f"    最佳选择性: {sel['best_selectivity']:.3f}\n")
                    f.write(f"    平均选择性: {sel['avg_selectivity']:.3f}\n")
            
            f.write("\n=== 分析结论 ===\n")
            if overall_stats['total_number_line'] > 0:
                f.write("• 发现了Number Line神经元，表明LSTM学习了数值的线性表征\n")
            else:
                f.write("• 未发现Number Line神经元，LSTM可能使用其他方式编码数值\n")
            
            if overall_stats['total_selective'] > 0:
                f.write("• 发现了Number Selective神经元，表明LSTM对特定数值有专门化响应\n")
            else:
                f.write("• 未发现Number Selective神经元，LSTM可能使用分布式数值表征\n")
        
        print(f"✅ 可读报告已保存")
    except Exception as e:
        print(f"⚠️ 可读报告保存失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM数值选择性神经元和调谐曲线分析工具')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, 
                        default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 可选参数
    parser.add_argument('--save_dir', type=str, default='./lstm_number_analysis',
                       help='结果保存目录 (默认: ./lstm_number_analysis)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='最大分析样本数 (默认: 500)')
    parser.add_argument('--min_r2', type=float, default=0.3,
                       help='Number Line神经元最小R²阈值 (默认: 0.5)')
    parser.add_argument('--selectivity_threshold', type=float, default=0.1,
                       help='选择性神经元阈值 (默认: 0.3)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小 (默认: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (默认: cuda)')
    parser.add_argument('--no_temporal', action='store_true',
                       help='跳过时间动态分析')
    
    return parser.parse_args()


def print_usage_info():
    """打印使用信息"""
    print("🧠 LSTM数值选择性神经元和调谐曲线分析工具")
    print("="*60)
    print("此工具分析LSTM中的数值编码神经元，包括:")
    print("  • Number Line神经元: 对数值有线性响应的神经元")
    print("  • Number Selective神经元: 对特定数值高度选择性的神经元")
    print("  • 调谐曲线分析: 神经元的数值偏好和调谐特性")
    print("  • 时间动态: 神经元特性在序列中的变化")
    print()
    print("支持的模型类型:")
    print("  • 原始具身计数模型 (EmbodiedCountingModel)")
    print("  • 消融实验模型 (counting_only, visual_only)")
    print()
    print("使用方法:")
    print("python lstm_number_analysis.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
    print()
    print("示例:")
    print("python lstm_number_analysis.py \\")
    print("    --checkpoint ./best_model.pth \\")
    print("    --val_csv ./validation_dataset.csv \\")
    print("    --data_root ./ball_data_collection \\")
    print("    --save_dir ./lstm_analysis_results \\")
    print("    --max_samples 1000 \\")
    print("    --min_r2 0.6 \\")
    print("    --selectivity_threshold 0.4")
    print()
    print("可选参数说明:")
    print("  --save_dir: 结果保存目录")
    print("  --max_samples: 分析的最大样本数 (越多越准确但越慢)")
    print("  --min_r2: Number Line神经元的最小R²阈值 (越高越严格)")
    print("  --selectivity_threshold: 选择性神经元阈值 (越高越严格)")
    print("  --batch_size: 数据加载批次大小")
    print("  --device: 计算设备 (cuda/cpu)")
    print("  --no_temporal: 跳过时间动态分析 (加快速度)")
    print()
    print("输出文件:")
    print("  • timestep_X_number_line_neurons.png: Number Line神经元可视化")
    print("  • timestep_X_number_selective_neurons.png: 选择性神经元可视化")
    print("  • timestep_comparison_*.png: 时间步对比")
    print("  • temporal_dynamics_*.png: 时间动态分析")
    print("  • lstm_analysis_report.json: 详细分析报告")
    print("  • lstm_analysis_summary.txt: 可读分析总结")
    print()
    print("💡 建议:")
    print("  • 首次运行可以用较少样本数 (如200) 快速测试")
    print("  • 对于发表论文的分析，建议使用1000+样本")
    print("  • 如果发现很少神经元，可以降低阈值参数")
    print("  • 使用GPU可以显著加快分析速度")


def main():
    """主函数"""
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        print_usage_info()
        return
    
    # 解析参数
    args = parse_arguments()
    
    print("🧠 LSTM数值选择性神经元和调谐曲线分析")
    print("="*50)
    print(f"模型检查点: {args.checkpoint}")
    print(f"验证数据: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print(f"最大样本数: {args.max_samples}")
    print(f"Number Line R²阈值: {args.min_r2}")
    print(f"选择性阈值: {args.selectivity_threshold}")
    print(f"设备: {args.device}")
    print(f"时间动态分析: {not args.no_temporal}")
    print("="*50)
    
    # 验证输入文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.val_csv):
        print(f"❌ 验证CSV文件不存在: {args.val_csv}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"❌ 数据根目录不存在: {args.data_root}")
        return
    
    # 运行分析
    try:
        start_time = time.time()
        
        results = analyze_lstm_number_neurons(
            checkpoint_path=args.checkpoint,
            val_csv=args.val_csv,
            data_root=args.data_root,
            save_dir=args.save_dir,
            max_samples=args.max_samples,
            min_r2=args.min_r2,
            selectivity_threshold=args.selectivity_threshold,
            analyze_temporal=not args.no_temporal
        )
        
        end_time = time.time()
        
        if results:
            print(f"\n🎉 分析成功完成!")
            print(f"⏱️ 总耗时: {end_time - start_time:.1f} 秒")
            print(f"📁 结果保存在: {args.save_dir}")
            
            # 打印关键结果摘要
            analysis_results = results['analysis_results']
            if analysis_results:
                print(f"\n📊 关键发现:")
                
                total_nl_neurons = sum(
                    len(res.get('number_line', {}).get('number_line_neurons', []))
                    for res in analysis_results.values()
                )
                
                total_sel_neurons = sum(
                    len(res.get('selective', {}).get('selective_neurons', []))
                    for res in analysis_results.values()
                )
                
                print(f"  • 发现 {total_nl_neurons} 个Number Line神经元")
                print(f"  • 发现 {total_sel_neurons} 个Number Selective神经元")
                print(f"  • 分析了 {len(analysis_results)} 个关键时间步")
                
                # 最佳时间步
                best_timestep = max(analysis_results.keys()) if analysis_results else None
                if best_timestep and 'selective' in analysis_results[best_timestep]:
                    sel_result = analysis_results[best_timestep]['selective']
                    best_proportion = sel_result['proportion']
                    print(f"  • 最佳时间步 {best_timestep}: {best_proportion:.1%} 的神经元有选择性")
            
            print(f"\n📋 生成的文件:")
            generated_files = []
            for root, dirs, files in os.walk(args.save_dir):
                for file in files:
                    if file.endswith(('.png', '.json', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, file), args.save_dir)
                        generated_files.append(rel_path)
            
            for file in sorted(generated_files):
                print(f"  • {file}")
            
        else:
            print(f"\n❌ 分析失败")
            
    except KeyboardInterrupt:
        print(f"\n⏸️ 分析被用户中断")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试函数"""
    print("🧪 运行快速测试...")
    
    # 这里可以添加一些快速测试代码
    # 比如测试模型加载、数据处理等基本功能
    
    print("✅ 快速测试完成")


if __name__ == "__main__":
    main()