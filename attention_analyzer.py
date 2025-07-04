"""
修复后的注意力机制深度分析工具
支持多头注意力机制
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
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


class AttentionAnalyzer:
    """专门的注意力机制分析器 - 支持多头注意力"""
    
    def __init__(self, figsize=(20, 15)):
        self.figsize = figsize
    
    def extract_attention_data(self, model, data_loader, max_samples=300, device='cuda'):
        """提取完整的注意力数据"""
        print("🎯 提取注意力机制数据...")
        
        all_attention_sequences = []  # [sample, seq_len, heads, spatial_size]
        all_labels = []
        all_predictions = []
        all_sample_ids = []
        all_counting_sequences = []  # [sample, seq_len] - 每个时刻的计数预测
        all_true_sequences = []     # [sample, seq_len] - 每个时刻的真实标签
        
        model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取注意力数据"):
                if sample_count >= max_samples:
                    break
                
                # 准备数据
                sequence_data = {
                    'images': batch['sequence_data']['images'].to(device),
                    'joints': batch['sequence_data']['joints'].to(device),
                    'timestamps': batch['sequence_data']['timestamps'].to(device),
                    'labels': batch['sequence_data']['labels'].to(device)
                }
                
                labels = batch['label'].cpu().numpy()  # CSV中的最终标签
                sample_ids = batch['sample_id']
                true_sequence_labels = sequence_data['labels'].cpu().numpy()  # 每个时刻的真实标签
                
                # 限制批次大小
                remaining_samples = max_samples - sample_count
                actual_batch_size = min(len(labels), remaining_samples)
                
                if actual_batch_size < len(labels):
                    for key in sequence_data:
                        sequence_data[key] = sequence_data[key][:actual_batch_size]
                    labels = labels[:actual_batch_size]
                    sample_ids = sample_ids[:actual_batch_size]
                    true_sequence_labels = true_sequence_labels[:actual_batch_size]
                
                # 前向传播，获取注意力权重
                try:
                    outputs = model(
                        sequence_data=sequence_data,
                        use_teacher_forcing=False,
                        return_attention=True
                    )
                except Exception as e:
                    print(f"⚠️ 模型前向传播失败，可能不支持return_attention: {e}")
                    # 尝试不返回注意力权重
                    outputs = model(
                        sequence_data=sequence_data,
                        use_teacher_forcing=False
                    )
                
                # 提取计数预测
                count_logits = outputs['counts']  # [batch, seq_len, 11]
                pred_sequence = torch.argmax(count_logits, dim=-1).cpu().numpy()  # [batch, seq_len]
                final_pred = pred_sequence[:, -1]  # 最终预测
                
                all_predictions.extend(final_pred)
                all_labels.extend(labels)
                all_sample_ids.extend(sample_ids)
                all_counting_sequences.extend(pred_sequence)
                all_true_sequences.extend(true_sequence_labels)
                
                # 提取注意力权重（如果可用）
                if 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights'].cpu().numpy()  # [batch, seq_len, heads, spatial]
                    all_attention_sequences.extend(attention_weights)
                else:
                    print("⚠️ 模型输出中没有注意力权重")
                
                sample_count += actual_batch_size
                
                if sample_count >= max_samples:
                    break
        
        # 构建结果
        result = {
            'attention_sequences': np.array(all_attention_sequences) if all_attention_sequences else None,
            'counting_sequences': np.array(all_counting_sequences),
            'true_sequences': np.array(all_true_sequences),
            'final_predictions': np.array(all_predictions),
            'true_labels': np.array(all_labels),
            'sample_ids': all_sample_ids
        }
        
        print(f"✅ 数据提取完成:")
        print(f"   样本数: {len(result['true_labels'])}")
        print(f"   注意力数据: {'有' if result['attention_sequences'] is not None else '无'}")
        if result['attention_sequences'] is not None:
            print(f"   注意力序列形状: {result['attention_sequences'].shape}")
            if len(result['attention_sequences'].shape) == 4:
                n_samples, seq_len, n_heads, spatial_size = result['attention_sequences'].shape
                print(f"   检测到多头注意力: {n_heads} 个头，{spatial_size} 个空间位置")
        print(f"   计数序列形状: {result['counting_sequences'].shape}")
        
        return result
    
    def _process_multi_head_attention(self, attention_seq):
        """处理多头注意力，返回平均注意力和每个头的注意力"""
        if len(attention_seq.shape) == 3:
            # 形状: (seq_len, heads, spatial) -> 平均所有头
            avg_attention = np.mean(attention_seq, axis=1)  # (seq_len, spatial)
            individual_heads = attention_seq  # (seq_len, heads, spatial)
            return avg_attention, individual_heads
        elif len(attention_seq.shape) == 2:
            # 形状: (seq_len, spatial) -> 单头注意力
            return attention_seq, attention_seq.unsqueeze(1)
        else:
            raise ValueError(f"不支持的注意力形状: {attention_seq.shape}")
    
    def plot_attention_evolution_detailed(self, attention_data, save_path=None):
        """详细的注意力演化分析 - 支持多头"""
        if attention_data['attention_sequences'] is None:
            print("⚠️ 没有注意力数据，跳过演化分析")
            return
        
        attention_sequences = attention_data['attention_sequences']
        true_labels = attention_data['true_labels']
        counting_sequences = attention_data['counting_sequences']
        
        # 检查是否为多头注意力
        is_multi_head = len(attention_sequences.shape) == 4
        
        if is_multi_head:
            n_samples, seq_len, n_heads, spatial_size = attention_sequences.shape
            print(f"🔍 检测到多头注意力: {n_heads} 个头")
        else:
            n_samples, seq_len, spatial_size = attention_sequences.shape
            n_heads = 1
        
        # 选择不同类别的代表性样本
        unique_labels = np.unique(true_labels)
        n_classes = min(len(unique_labels), 10)  # 最多显示3个类别
        samples_per_class = 2
        
        # 计算子图数量
        cols_per_sample = 4  # 每个样本4列: 平均注意力, 注意力集中度, 计数vs注意力, 多头对比
        total_cols = samples_per_class * cols_per_sample
        
        fig, axes = plt.subplots(n_classes, total_cols, 
                                figsize=(total_cols * 3, n_classes * 4))
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        elif total_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, label in enumerate(unique_labels[:n_classes]):
            class_mask = true_labels == label
            class_indices = np.where(class_mask)[0]
            
            # 选择该类别的样本
            if len(class_indices) >= samples_per_class:
                selected_samples = np.random.choice(class_indices, samples_per_class, replace=False)
            else:
                selected_samples = class_indices
            
            for sample_idx, sample_id in enumerate(selected_samples):
                if sample_idx >= samples_per_class:
                    break
                    
                attention_seq = attention_sequences[sample_id]  # (seq_len, heads, spatial) 或 (seq_len, spatial)
                counting_seq = counting_sequences[sample_id]    # (seq_len,)
                
                # 处理多头注意力
                if is_multi_head:
                    avg_attention, individual_heads = self._process_multi_head_attention(attention_seq)
                else:
                    avg_attention = attention_seq
                    individual_heads = attention_seq
                
                seq_len, spatial_size = avg_attention.shape
                
                # 计算列的基础索引
                col_base = sample_idx * cols_per_sample
                
                # 1. 平均注意力时序演化热力图
                ax1 = axes[class_idx, col_base] if n_classes > 1 else axes[col_base]
                
                im1 = ax1.imshow(avg_attention.T, cmap='viridis', aspect='auto')
                ax1.set_title(f'Class {label} Sample {sample_idx+1}\nAvg Attention Evolution')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Spatial Location')
                plt.colorbar(im1, ax=ax1, shrink=0.8)
                
                # 2. 注意力集中度随时间变化
                ax2 = axes[class_idx, col_base + 1] if n_classes > 1 else axes[col_base + 1]
                
                # 计算每个时刻的注意力集中度（熵的倒数）
                concentration = []
                for t in range(seq_len):
                    attention_t = avg_attention[t] + 1e-8
                    entropy = -np.sum(attention_t * np.log(attention_t))
                    concentration.append(1.0 / (entropy + 1e-8))
                
                time_steps = range(seq_len)
                ax2.plot(time_steps, concentration, 'b-', linewidth=2, marker='o')
                ax2.set_title('Attention Concentration')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Concentration (1/Entropy)')
                ax2.grid(True, alpha=0.3)
                
                # 3. 计数预测与注意力关联
                ax3 = axes[class_idx, col_base + 2] if n_classes > 1 else axes[col_base + 2]
                
                # 双轴图：计数预测和注意力集中度
                ax3_twin = ax3.twinx()
                
                line1 = ax3.plot(time_steps, counting_seq, 'r-', linewidth=2, marker='s', 
                               label='Count Prediction')
                line2 = ax3_twin.plot(time_steps, concentration, 'b--', linewidth=2, marker='o', 
                                    label='Attention Concentration')
                
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Predicted Count', color='red')
                ax3_twin.set_ylabel('Attention Concentration', color='blue')
                ax3.set_title('Count vs Attention')
                ax3.grid(True, alpha=0.3)
                
                # 添加真实标签线
                true_label = true_labels[sample_id]
                ax3.axhline(y=true_label, color='green', linestyle=':', 
                          linewidth=2, label=f'True Count ({true_label})')
                
                # 合并图例
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines + [ax3.lines[-1]], labels + ['True Count'], loc='upper left')
                
                # 4. 多头注意力对比 (如果是多头)
                ax4 = axes[class_idx, col_base + 3] if n_classes > 1 else axes[col_base + 3]
                
                if is_multi_head and n_heads > 1:
                    # 显示每个头的注意力集中度
                    colors = plt.cm.tab10(np.linspace(0, 1, n_heads))
                    
                    for head_idx in range(n_heads):
                        head_attention = individual_heads[:, head_idx, :]  # (seq_len, spatial)
                        head_concentration = []
                        
                        for t in range(seq_len):
                            attention_t = head_attention[t] + 1e-8
                            entropy = -np.sum(attention_t * np.log(attention_t))
                            head_concentration.append(1.0 / (entropy + 1e-8))
                        
                        ax4.plot(time_steps, head_concentration, 
                               color=colors[head_idx], linewidth=2, 
                               marker='o', label=f'Head {head_idx+1}')
                    
                    ax4.plot(time_steps, concentration, 'k--', linewidth=3, 
                           label='Average', alpha=0.8)
                    ax4.set_title('Multi-Head Attention Comparison')
                    ax4.set_xlabel('Time Step')
                    ax4.set_ylabel('Concentration')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    # 单头或平均注意力的空间分布
                    final_attention = avg_attention[-1]  # 最后时刻的注意力
                    ax4.bar(range(min(20, len(final_attention))), final_attention[:20])
                    ax4.set_title('Final Attention Distribution\n(Top 20 locations)')
                    ax4.set_xlabel('Spatial Location')
                    ax4.set_ylabel('Attention Weight')
                    ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_accuracy_analysis(self, attention_data, save_path=None):
        """注意力与准确性的详细分析 - 支持多头"""
        if attention_data['attention_sequences'] is None:
            print("⚠️ 没有注意力数据，跳过准确性分析")
            return
        
        attention_sequences = attention_data['attention_sequences']
        true_labels = attention_data['true_labels']
        final_predictions = attention_data['final_predictions']
        counting_sequences = attention_data['counting_sequences']
        
        # 处理多头注意力 - 取平均
        if len(attention_sequences.shape) == 4:
            # (n_samples, seq_len, n_heads, spatial) -> (n_samples, seq_len, spatial)
            attention_sequences = np.mean(attention_sequences, axis=2)
            print("🔍 多头注意力已平均处理")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 计算注意力集中度
        attention_concentration = []
        for seq in attention_sequences:
            concentration = []
            for t in range(seq.shape[0]):
                attention_t = seq[t] + 1e-8
                entropy = -np.sum(attention_t * np.log(attention_t))
                concentration.append(1.0 / (entropy + 1e-8))
            attention_concentration.append(concentration)
        
        attention_concentration = np.array(attention_concentration)
        
        # 1. 整体注意力集中度演化
        mean_concentration = np.mean(attention_concentration, axis=0)
        std_concentration = np.std(attention_concentration, axis=0)
        time_steps = range(len(mean_concentration))
        
        axes[0, 0].plot(time_steps, mean_concentration, 'b-', linewidth=2, label='Mean')
        axes[0, 0].fill_between(time_steps, 
                               mean_concentration - std_concentration,
                               mean_concentration + std_concentration,
                               alpha=0.3, color='blue')
        axes[0, 0].set_title('Overall Attention Concentration Evolution')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Concentration')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 正确vs错误预测的注意力对比
        correct_mask = final_predictions == true_labels
        error_mask = ~correct_mask
        
        if np.any(correct_mask) and np.any(error_mask):
            correct_concentration = np.mean(attention_concentration[correct_mask], axis=0)
            error_concentration = np.mean(attention_concentration[error_mask], axis=0)
            
            axes[0, 1].plot(time_steps, correct_concentration, 'g-', linewidth=2, 
                           label=f'Correct ({np.sum(correct_mask)} samples)')
            axes[0, 1].plot(time_steps, error_concentration, 'r-', linewidth=2, 
                           label=f'Error ({np.sum(error_mask)} samples)')
            axes[0, 1].set_title('Attention: Correct vs Error Predictions')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Mean Concentration')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No comparison available\n(all correct or all error)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Attention: Correct vs Error Predictions')
        
        # 3. 不同类别的注意力模式
        unique_labels = np.unique(true_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            class_mask = true_labels == label
            if np.any(class_mask):
                class_concentration = np.mean(attention_concentration[class_mask], axis=0)
                axes[0, 2].plot(time_steps, class_concentration, 
                               color=colors[i], linewidth=2, 
                               label=f'Class {label} ({np.sum(class_mask)} samples)')
        
        axes[0, 2].set_title('Attention Patterns by True Class')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('Mean Concentration')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # 4. 最终注意力集中度分布
        final_concentration = attention_concentration[:, -1]
        
        if np.any(correct_mask):
            axes[1, 0].hist(final_concentration[correct_mask], bins=20, alpha=0.7, 
                           color='green', label='Correct Predictions', density=True)
        if np.any(error_mask):
            axes[1, 0].hist(final_concentration[error_mask], bins=20, alpha=0.7, 
                           color='red', label='Error Predictions', density=True)
        axes[1, 0].set_title('Final Attention Concentration Distribution')
        axes[1, 0].set_xlabel('Final Concentration')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 注意力集中度与准确性的关系
        # 将最终集中度分桶，计算每桶的准确率
        n_bins = 8
        concentration_bins = np.linspace(np.min(final_concentration), 
                                       np.max(final_concentration), n_bins + 1)
        
        bin_accuracies = []
        bin_centers = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = (final_concentration >= concentration_bins[i]) & \
                      (final_concentration < concentration_bins[i + 1])
            if i == n_bins - 1:  # 最后一个bin包含右边界
                bin_mask = (final_concentration >= concentration_bins[i]) & \
                          (final_concentration <= concentration_bins[i + 1])
            
            if np.any(bin_mask):
                bin_accuracy = np.mean(final_predictions[bin_mask] == true_labels[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_centers.append((concentration_bins[i] + concentration_bins[i + 1]) / 2)
                bin_counts.append(np.sum(bin_mask))
            
        if bin_centers:
            axes[1, 1].bar(bin_centers, bin_accuracies, 
                          width=np.diff(concentration_bins)[0] * 0.8, alpha=0.7,
                          color='purple')
            axes[1, 1].set_title('Accuracy vs Final Attention Concentration')
            axes[1, 1].set_xlabel('Final Attention Concentration')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加样本数量标注
            for center, accuracy, count in zip(bin_centers, bin_accuracies, bin_counts):
                axes[1, 1].text(center, accuracy + 0.02, f'n={count}', 
                               ha='center', va='bottom', fontsize=8)
        
        # 6. 计数准确性随时间演化
        step_accuracies = []
        for t in range(counting_sequences.shape[1]):
            step_accuracy = np.mean(counting_sequences[:, t] == true_labels)
            step_accuracies.append(step_accuracy)
        
        # 双轴：计数准确性和注意力集中度
        ax_acc = axes[1, 2]
        ax_att = ax_acc.twinx()
        
        line1 = ax_acc.plot(time_steps, step_accuracies, 'g-', linewidth=2, 
                           marker='o', label='Count Accuracy')
        line2 = ax_att.plot(time_steps, mean_concentration, 'b--', linewidth=2, 
                           marker='s', label='Attention Concentration')
        
        ax_acc.set_xlabel('Time Step')
        ax_acc.set_ylabel('Count Accuracy', color='green')
        ax_att.set_ylabel('Attention Concentration', color='blue')
        ax_acc.set_title('Count Accuracy vs Attention Over Time')
        ax_acc.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_acc.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spatial_attention_analysis(self, attention_data, save_path=None):
        """空间注意力分析 - 支持多头"""
        if attention_data['attention_sequences'] is None:
            print("⚠️ 没有注意力数据，跳过空间分析")
            return
        
        attention_sequences = attention_data['attention_sequences']
        true_labels = attention_data['true_labels']
        
        # 处理多头注意力
        if len(attention_sequences.shape) == 4:
            n_samples, seq_len, n_heads, spatial_size = attention_sequences.shape
            # 取所有头的平均
            attention_sequences_avg = np.mean(attention_sequences, axis=2)
            print(f"🔍 多头注意力 ({n_heads} 个头) 已平均处理")
        else:
            n_samples, seq_len, spatial_size = attention_sequences.shape
            attention_sequences_avg = attention_sequences
            n_heads = 1
        
        grid_size = int(np.sqrt(spatial_size))
        
        if grid_size * grid_size != spatial_size:
            print(f"⚠️ 空间大小 {spatial_size} 不是完全平方数，使用1D可视化")
            self._plot_1d_spatial_analysis(attention_data, save_path)
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 整体平均空间注意力
        overall_avg = np.mean(attention_sequences_avg, axis=(0, 1))  # 平均所有样本和时刻
        overall_heatmap = overall_avg.reshape(grid_size, grid_size)
        
        im1 = axes[0, 0].imshow(overall_heatmap, cmap='viridis')
        axes[0, 0].set_title('Overall Average Spatial Attention')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 初始vs最终时刻对比
        initial_avg = np.mean(attention_sequences_avg[:, 0, :], axis=0).reshape(grid_size, grid_size)
        final_avg = np.mean(attention_sequences_avg[:, -1, :], axis=0).reshape(grid_size, grid_size)
        
        im2 = axes[0, 1].imshow(initial_avg, cmap='viridis')
        axes[0, 1].set_title('Initial Time Step Attention')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(final_avg, cmap='viridis')
        axes[0, 2].set_title('Final Time Step Attention')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 3. 不同类别的空间注意力对比
        unique_labels = np.unique(true_labels)
        if len(unique_labels) >= 2:
            label1, label2 = unique_labels[0], unique_labels[-1]
            
            class1_attention = np.mean(attention_sequences_avg[true_labels == label1], axis=(0, 1))
            class2_attention = np.mean(attention_sequences_avg[true_labels == label2], axis=(0, 1))
            
            class1_heatmap = class1_attention.reshape(grid_size, grid_size)
            
            im4 = axes[1, 0].imshow(class1_heatmap, cmap='viridis')
            axes[1, 0].set_title(f'Class {label1} Average Attention')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0])
            
            # 计算并显示类别差异
            diff_heatmap = class2_attention.reshape(grid_size, grid_size) - class1_heatmap
            
            im5 = axes[1, 1].imshow(diff_heatmap, cmap='RdBu_r')
            axes[1, 1].set_title(f'Attention Difference\n(Class {label2} - Class {label1})')
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1])
        
        # 4. 空间注意力的时序变化
        temporal_change = np.std(attention_sequences_avg, axis=1)  # [samples, spatial]
        mean_temporal_change = np.mean(temporal_change, axis=0).reshape(grid_size, grid_size)
        
        im6 = axes[1, 2].imshow(mean_temporal_change, cmap='plasma')
        axes[1, 2].set_title('Temporal Variability in Spatial Attention')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_1d_spatial_analysis(self, attention_data, save_path=None):
        """1D空间注意力分析（当空间维度不是完全平方数时） - 支持多头"""
        attention_sequences = attention_data['attention_sequences']
        true_labels = attention_data['true_labels']
        
        # 处理多头注意力
        if len(attention_sequences.shape) == 4:
            # 取平均
            attention_sequences = np.mean(attention_sequences, axis=2)
            print("🔍 1D分析: 多头注意力已平均处理")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. 整体空间注意力分布
        overall_avg = np.mean(attention_sequences, axis=(0, 1))
        
        axes[0, 0].bar(range(len(overall_avg)), overall_avg, alpha=0.7)
        axes[0, 0].set_title('Overall Average Spatial Attention')
        axes[0, 0].set_xlabel('Spatial Dimension')
        axes[0, 0].set_ylabel('Attention Weight')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 时序演化热力图
        mean_attention_over_time = np.mean(attention_sequences, axis=0)  # [seq_len, spatial]
        
        im = axes[0, 1].imshow(mean_attention_over_time.T, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Spatial Attention Evolution Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Spatial Dimension')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. 不同类别的空间注意力
        unique_labels = np.unique(true_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            class_mask = true_labels == label
            if np.any(class_mask):
                class_attention = np.mean(attention_sequences[class_mask], axis=(0, 1))
                axes[1, 0].plot(class_attention, color=colors[i], linewidth=2, 
                               label=f'Class {label}')
        
        axes[1, 0].set_title('Spatial Attention by Class')
        axes[1, 0].set_xlabel('Spatial Dimension')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 空间注意力的方差
        spatial_variance = np.var(attention_sequences, axis=(0, 1))
        
        axes[1, 1].bar(range(len(spatial_variance)), spatial_variance, 
                      alpha=0.7, color='orange')
        axes[1, 1].set_title('Spatial Attention Variance')
        axes[1, 1].set_xlabel('Spatial Dimension')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multi_head_analysis(self, attention_data, save_path=None):
        """专门的多头注意力分析"""
        if attention_data['attention_sequences'] is None:
            print("⚠️ 没有注意力数据，跳过多头分析")
            return
        
        attention_sequences = attention_data['attention_sequences']
        
        if len(attention_sequences.shape) != 4:
            print("⚠️ 不是多头注意力，跳过多头分析")
            return
        
        n_samples, seq_len, n_heads, spatial_size = attention_sequences.shape
        true_labels = attention_data['true_labels']
        
        print(f"🔍 多头注意力分析: {n_heads} 个头")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 每个头的平均注意力集中度
        head_concentrations = []
        for head_idx in range(n_heads):
            head_attention = attention_sequences[:, :, head_idx, :]  # [samples, seq_len, spatial]
            
            head_conc = []
            for sample_idx in range(n_samples):
                sample_conc = []
                for t in range(seq_len):
                    attention_t = head_attention[sample_idx, t] + 1e-8
                    entropy = -np.sum(attention_t * np.log(attention_t))
                    sample_conc.append(1.0 / (entropy + 1e-8))
                head_conc.append(sample_conc)
            head_concentrations.append(np.array(head_conc))
        
        # 绘制每个头的平均集中度
        time_steps = range(seq_len)
        colors = plt.cm.tab10(np.linspace(0, 1, n_heads))
        
        for head_idx in range(n_heads):
            mean_conc = np.mean(head_concentrations[head_idx], axis=0)
            axes[0, 0].plot(time_steps, mean_conc, color=colors[head_idx], 
                           linewidth=2, marker='o', label=f'Head {head_idx+1}')
        
        # 平均所有头
        overall_mean = np.mean([np.mean(hc, axis=0) for hc in head_concentrations], axis=0)
        axes[0, 0].plot(time_steps, overall_mean, 'k--', linewidth=3, 
                       label='Average All Heads', alpha=0.8)
        
        axes[0, 0].set_title('Multi-Head Attention Concentration')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Concentration')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 头之间的相关性矩阵
        head_correlations = np.zeros((n_heads, n_heads))
        
        for i in range(n_heads):
            for j in range(n_heads):
                # 计算两个头在所有样本和时刻的相关性
                head_i_flat = attention_sequences[:, :, i, :].flatten()
                head_j_flat = attention_sequences[:, :, j, :].flatten()
                correlation = np.corrcoef(head_i_flat, head_j_flat)[0, 1]
                head_correlations[i, j] = correlation
        
        im_corr = axes[0, 1].imshow(head_correlations, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title('Head-to-Head Correlation Matrix')
        axes[0, 1].set_xlabel('Head Index')
        axes[0, 1].set_ylabel('Head Index')
        
        # 添加数值标注
        for i in range(n_heads):
            for j in range(n_heads):
                text = axes[0, 1].text(j, i, f'{head_correlations[i, j]:.2f}',
                                     ha="center", va="center", color="black" if abs(head_correlations[i, j]) < 0.5 else "white")
        
        plt.colorbar(im_corr, ax=axes[0, 1])
        
        # 3. 每个头的空间注意力分布方差
        head_spatial_vars = []
        for head_idx in range(n_heads):
            head_attention = attention_sequences[:, :, head_idx, :]
            # 计算每个空间位置的方差
            spatial_var = np.var(head_attention, axis=(0, 1))  # [spatial]
            head_spatial_vars.append(np.mean(spatial_var))  # 平均方差
        
        axes[0, 2].bar(range(n_heads), head_spatial_vars, color=colors, alpha=0.7)
        axes[0, 2].set_title('Spatial Attention Variance by Head')
        axes[0, 2].set_xlabel('Head Index')
        axes[0, 2].set_ylabel('Mean Spatial Variance')
        axes[0, 2].set_xticks(range(n_heads))
        axes[0, 2].set_xticklabels([f'H{i+1}' for i in range(n_heads)])
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 不同类别下各头的表现
        unique_labels = np.unique(true_labels)
        if len(unique_labels) >= 2:
            label1, label2 = unique_labels[0], unique_labels[-1]
            
            class1_mask = true_labels == label1
            class2_mask = true_labels == label2
            
            class1_head_conc = []
            class2_head_conc = []
            
            for head_idx in range(n_heads):
                class1_conc = np.mean(head_concentrations[head_idx][class1_mask])
                class2_conc = np.mean(head_concentrations[head_idx][class2_mask])
                class1_head_conc.append(class1_conc)
                class2_head_conc.append(class2_conc)
            
            x = np.arange(n_heads)
            width = 0.35
            
            axes[1, 0].bar(x - width/2, class1_head_conc, width, label=f'Class {label1}', alpha=0.7)
            axes[1, 0].bar(x + width/2, class2_head_conc, width, label=f'Class {label2}', alpha=0.7)
            
            axes[1, 0].set_title('Head Performance by Class')
            axes[1, 0].set_xlabel('Head Index')
            axes[1, 0].set_ylabel('Mean Concentration')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([f'H{i+1}' for i in range(n_heads)])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 头的多样性分析
        # 计算每个时刻各头注意力的标准差（多样性指标）
        diversity_over_time = []
        for t in range(seq_len):
            time_diversity = []
            for sample_idx in range(n_samples):
                # 当前样本当前时刻所有头的注意力
                heads_attention = attention_sequences[sample_idx, t, :, :]  # [n_heads, spatial]
                # 计算头之间的标准差
                head_std = np.std(heads_attention, axis=0)  # [spatial]
                time_diversity.append(np.mean(head_std))
            diversity_over_time.append(np.mean(time_diversity))
        
        axes[1, 1].plot(time_steps, diversity_over_time, 'purple', linewidth=2, marker='o')
        axes[1, 1].set_title('Head Diversity Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Mean Head Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 最有效的头识别
        # 基于与最终预测准确性的相关性
        final_predictions = attention_data['final_predictions']
        correct_mask = final_predictions == true_labels
        
        head_accuracy_correlation = []
        for head_idx in range(n_heads):
            head_final_conc = np.mean(head_concentrations[head_idx], axis=1)  # 每个样本的平均集中度
            
            # 计算与准确性的相关性
            if len(np.unique(correct_mask)) > 1:  # 确保有正确和错误的样本
                correlation = np.corrcoef(head_final_conc, correct_mask.astype(float))[0, 1]
            else:
                correlation = 0.0
            head_accuracy_correlation.append(correlation)
        
        axes[1, 2].bar(range(n_heads), head_accuracy_correlation, 
                      color=['green' if c > 0 else 'red' for c in head_accuracy_correlation], 
                      alpha=0.7)
        axes[1, 2].set_title('Head-Accuracy Correlation')
        axes[1, 2].set_xlabel('Head Index')
        axes[1, 2].set_ylabel('Correlation with Accuracy')
        axes[1, 2].set_xticks(range(n_heads))
        axes[1, 2].set_xticklabels([f'H{i+1}' for i in range(n_heads)])
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_attention_report(self, attention_data, save_path=None):
        """生成详细的注意力分析报告 - 支持多头"""
        
        if attention_data['attention_sequences'] is None:
            print("⚠️ 没有注意力数据，无法生成报告")
            return None
        
        attention_sequences = attention_data['attention_sequences']
        true_labels = attention_data['true_labels']
        final_predictions = attention_data['final_predictions']
        counting_sequences = attention_data['counting_sequences']
        
        # 检查多头注意力
        is_multi_head = len(attention_sequences.shape) == 4
        if is_multi_head:
            n_samples, seq_len, n_heads, spatial_size = attention_sequences.shape
            # 为报告使用平均注意力
            attention_sequences_avg = np.mean(attention_sequences, axis=2)
        else:
            n_samples, seq_len, spatial_size = attention_sequences.shape
            attention_sequences_avg = attention_sequences
            n_heads = 1
        
        # 计算各种统计指标
        overall_accuracy = np.mean(final_predictions == true_labels)
        
        # 注意力集中度分析
        attention_concentration = []
        for seq in attention_sequences_avg:
            concentration = []
            for t in range(seq.shape[0]):
                attention_t = seq[t] + 1e-8
                entropy = -np.sum(attention_t * np.log(attention_t))
                concentration.append(1.0 / (entropy + 1e-8))
            attention_concentration.append(concentration)
        
        attention_concentration = np.array(attention_concentration)
        
        # 构建报告
        report = {
            'analysis_summary': {
                'total_samples': int(n_samples),
                'sequence_length': int(seq_len),
                'spatial_dimension': int(spatial_size),
                'is_multi_head': is_multi_head,
                'num_heads': int(n_heads),
                'overall_accuracy': float(overall_accuracy),
                'unique_classes': len(np.unique(true_labels))
            },
            'attention_concentration_stats': {
                'initial_mean': float(np.mean(attention_concentration[:, 0])),
                'initial_std': float(np.std(attention_concentration[:, 0])),
                'final_mean': float(np.mean(attention_concentration[:, -1])),
                'final_std': float(np.std(attention_concentration[:, -1])),
                'mean_change': float(np.mean(attention_concentration[:, -1] - attention_concentration[:, 0])),
                'max_concentration': float(np.max(attention_concentration)),
                'min_concentration': float(np.min(attention_concentration))
            }
        }
        
        # 多头注意力特有分析
        if is_multi_head:
            # 计算头之间的相关性
            head_correlations = []
            for i in range(n_heads):
                for j in range(i+1, n_heads):
                    head_i_flat = attention_sequences[:, :, i, :].flatten()
                    head_j_flat = attention_sequences[:, :, j, :].flatten()
                    correlation = np.corrcoef(head_i_flat, head_j_flat)[0, 1]
                    head_correlations.append(correlation)
            
            report['multi_head_analysis'] = {
                'mean_inter_head_correlation': float(np.mean(head_correlations)),
                'std_inter_head_correlation': float(np.std(head_correlations)),
                'max_correlation': float(np.max(head_correlations)),
                'min_correlation': float(np.min(head_correlations))
            }
        
        # 准确性相关分析
        correct_mask = final_predictions == true_labels
        if np.any(correct_mask) and np.any(~correct_mask):
            correct_final_conc = attention_concentration[correct_mask, -1]
            error_final_conc = attention_concentration[~correct_mask, -1]
            
            report['accuracy_correlation'] = {
                'correct_samples': int(np.sum(correct_mask)),
                'error_samples': int(np.sum(~correct_mask)),
                'correct_mean_concentration': float(np.mean(correct_final_conc)),
                'error_mean_concentration': float(np.mean(error_final_conc)),
                'concentration_difference': float(np.mean(correct_final_conc) - np.mean(error_final_conc)),
                'statistical_significance': self._compute_significance(correct_final_conc, error_final_conc)
            }
        
        # 类别特异性分析
        unique_labels = np.unique(true_labels)
        class_stats = {}
        
        for label in unique_labels:
            class_mask = true_labels == label
            if np.any(class_mask):
                class_concentration = attention_concentration[class_mask]
                class_accuracy = np.mean(final_predictions[class_mask] == label)
                
                class_stats[f'class_{label}'] = {
                    'sample_count': int(np.sum(class_mask)),
                    'accuracy': float(class_accuracy),
                    'initial_concentration_mean': float(np.mean(class_concentration[:, 0])),
                    'final_concentration_mean': float(np.mean(class_concentration[:, -1])),
                    'concentration_change_mean': float(np.mean(class_concentration[:, -1] - class_concentration[:, 0])),
                    'concentration_stability': float(np.std(np.std(class_concentration, axis=1)))
                }
        
        report['class_specific_attention'] = class_stats
        
        # 时序分析
        step_accuracies = []
        for t in range(seq_len):
            step_accuracy = np.mean(counting_sequences[:, t] == true_labels)
            step_accuracies.append(step_accuracy)
        
        report['temporal_analysis'] = {
            'initial_accuracy': float(step_accuracies[0]),
            'final_accuracy': float(step_accuracies[-1]),
            'max_accuracy': float(np.max(step_accuracies)),
            'accuracy_improvement': float(step_accuracies[-1] - step_accuracies[0]),
            'convergence_step': int(np.argmax(step_accuracies))
        }
        
        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # JSON格式
            json_path = save_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # 文本格式
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== 深度注意力机制分析报告 ===\n\n")
                
                # 基础统计
                summary = report['analysis_summary']
                f.write("基础统计信息:\n")
                f.write(f"  总样本数: {summary['total_samples']}\n")
                f.write(f"  序列长度: {summary['sequence_length']}\n")
                f.write(f"  空间维度: {summary['spatial_dimension']}\n")
                f.write(f"  多头注意力: {'是' if summary['is_multi_head'] else '否'}\n")
                if summary['is_multi_head']:
                    f.write(f"  注意力头数: {summary['num_heads']}\n")
                f.write(f"  整体准确率: {summary['overall_accuracy']:.4f}\n")
                f.write(f"  类别数量: {summary['unique_classes']}\n\n")
                
                # 多头注意力分析
                if 'multi_head_analysis' in report:
                    mha = report['multi_head_analysis']
                    f.write("多头注意力分析:\n")
                    f.write(f"  头间平均相关性: {mha['mean_inter_head_correlation']:.4f} ± {mha['std_inter_head_correlation']:.4f}\n")
                    f.write(f"  最大头间相关性: {mha['max_correlation']:.4f}\n")
                    f.write(f"  最小头间相关性: {mha['min_correlation']:.4f}\n\n")
                
                # 注意力集中度分析
                conc_stats = report['attention_concentration_stats']
                f.write("注意力集中度分析:\n")
                f.write(f"  初始集中度: {conc_stats['initial_mean']:.4f} ± {conc_stats['initial_std']:.4f}\n")
                f.write(f"  最终集中度: {conc_stats['final_mean']:.4f} ± {conc_stats['final_std']:.4f}\n")
                f.write(f"  平均变化量: {conc_stats['mean_change']:.4f}\n")
                f.write(f"  集中度范围: [{conc_stats['min_concentration']:.4f}, {conc_stats['max_concentration']:.4f}]\n\n")
                
                # 准确性关联分析
                if 'accuracy_correlation' in report:
                    acc_corr = report['accuracy_correlation']
                    f.write("准确性关联分析:\n")
                    f.write(f"  正确预测样本: {acc_corr['correct_samples']}\n")
                    f.write(f"  错误预测样本: {acc_corr['error_samples']}\n")
                    f.write(f"  正确预测的平均注意力集中度: {acc_corr['correct_mean_concentration']:.4f}\n")
                    f.write(f"  错误预测的平均注意力集中度: {acc_corr['error_mean_concentration']:.4f}\n")
                    f.write(f"  集中度差异: {acc_corr['concentration_difference']:.4f}\n")
                    f.write(f"  统计显著性: {acc_corr['statistical_significance']}\n\n")
                
                # 类别特异性分析
                f.write("类别特异性注意力分析:\n")
                for class_name, stats in report['class_specific_attention'].items():
                    f.write(f"  {class_name}:\n")
                    f.write(f"    样本数: {stats['sample_count']}\n")
                    f.write(f"    准确率: {stats['accuracy']:.4f}\n")
                    f.write(f"    初始集中度: {stats['initial_concentration_mean']:.4f}\n")
                    f.write(f"    最终集中度: {stats['final_concentration_mean']:.4f}\n")
                    f.write(f"    集中度变化: {stats['concentration_change_mean']:.4f}\n")
                    f.write(f"    注意力稳定性: {stats['concentration_stability']:.4f}\n")
                f.write("\n")
                
                # 时序分析
                temporal = report['temporal_analysis']
                f.write("时序演化分析:\n")
                f.write(f"  初始准确率: {temporal['initial_accuracy']:.4f}\n")
                f.write(f"  最终准确率: {temporal['final_accuracy']:.4f}\n")
                f.write(f"  最高准确率: {temporal['max_accuracy']:.4f}\n")
                f.write(f"  准确率提升: {temporal['accuracy_improvement']:.4f}\n")
                f.write(f"  收敛步骤: {temporal['convergence_step']}\n\n")
                
                # 分析建议
                f.write("分析建议:\n")
                recommendations = self._generate_recommendations(report)
                for rec in recommendations:
                    f.write(f"  • {rec}\n")
            
            print(f"✅ 报告已保存: {save_path}")
            print(f"✅ JSON数据已保存: {json_path}")
        
        return report
    
    def _compute_significance(self, correct_data, error_data):
        """计算统计显著性"""
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(correct_data, error_data)
            if p_value < 0.001:
                return "高度显著 (p < 0.001)"
            elif p_value < 0.01:
                return "显著 (p < 0.01)"
            elif p_value < 0.05:
                return "边际显著 (p < 0.05)"
            else:
                return "不显著 (p >= 0.05)"
        except:
            return "无法计算"
    
    def _generate_recommendations(self, report):
        """基于分析结果生成建议"""
        recommendations = []
        
        # 基于准确率的建议
        accuracy = report['analysis_summary']['overall_accuracy']
        if accuracy < 0.7:
            recommendations.append("整体准确率较低，注意力机制可能需要改进")
        elif accuracy > 0.9:
            recommendations.append("准确率很高，注意力机制运作良好")
        
        # 多头注意力建议
        if report['analysis_summary']['is_multi_head']:
            if 'multi_head_analysis' in report:
                mha = report['multi_head_analysis']
                if mha['mean_inter_head_correlation'] > 0.8:
                    recommendations.append("注意力头之间相关性过高，可能存在冗余，建议减少头数或增加多样性")
                elif mha['mean_inter_head_correlation'] < 0.3:
                    recommendations.append("注意力头之间相关性较低，显示了良好的多样性")
        
        # 基于注意力集中度的建议
        conc_stats = report['attention_concentration_stats']
        if conc_stats['mean_change'] > 0:
            recommendations.append("注意力集中度随时间增加，显示了良好的学习过程")
        else:
            recommendations.append("注意力集中度随时间减少，可能存在注意力分散问题")
        
        # 基于准确性关联的建议
        if 'accuracy_correlation' in report:
            acc_corr = report['accuracy_correlation']
            if acc_corr['concentration_difference'] > 0:
                recommendations.append("正确预测具有更高的注意力集中度，注意力质量与准确性正相关")
            else:
                recommendations.append("错误预测的注意力集中度更高，可能存在过度关注错误特征的问题")
        
        # 基于时序分析的建议
        temporal = report['temporal_analysis']
        if temporal['accuracy_improvement'] > 0.1:
            recommendations.append("随时间准确率显著提升，时序建模效果良好")
        elif temporal['accuracy_improvement'] < 0:
            recommendations.append("准确率随时间下降，可能存在梯度消失或过拟合问题")
        
        return recommendations


def comprehensive_attention_analysis(checkpoint_path, val_csv, data_root, 
                                   save_dir='./attention_analysis', 
                                   max_samples=300):
    """完整的注意力机制分析流水线 - 支持多头注意力"""
    
    print("🔍 开始深度注意力机制分析...")
    print("="*60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建注意力分析器
        analyzer = AttentionAnalyzer()
        
        # 3. 提取注意力数据
        print("\n📊 第1步: 提取注意力机制数据")
        attention_data = analyzer.extract_attention_data(
            model, val_loader, max_samples, device
        )
        
        if attention_data['attention_sequences'] is None:
            print("❌ 模型不支持注意力权重提取，分析终止")
            print("💡 请确保模型的forward方法支持return_attention=True参数")
            return None
        
        # 检查是否为多头注意力
        is_multi_head = len(attention_data['attention_sequences'].shape) == 4
        if is_multi_head:
            n_heads = attention_data['attention_sequences'].shape[2]
            print(f"🎯 检测到多头注意力机制: {n_heads} 个注意力头")
        
        # 4. 注意力演化分析
        print("\n📈 第2步: 注意力演化分析")
        analyzer.plot_attention_evolution_detailed(
            attention_data,
            save_path=os.path.join(save_dir, 'attention_evolution_detailed.png')
        )
        
        # 5. 注意力与准确性关联分析
        print("\n🎯 第3步: 注意力与准确性关联分析")
        analyzer.plot_attention_accuracy_analysis(
            attention_data,
            save_path=os.path.join(save_dir, 'attention_accuracy_analysis.png')
        )
        
        # 6. 空间注意力分析
        print("\n🗺️ 第4步: 空间注意力分析")
        analyzer.plot_spatial_attention_analysis(
            attention_data,
            save_path=os.path.join(save_dir, 'spatial_attention_analysis.png')
        )
        
        # 7. 多头注意力专门分析（如果适用）
        if is_multi_head:
            print("\n🧠 第5步: 多头注意力专门分析")
            analyzer.plot_multi_head_analysis(
                attention_data,
                save_path=os.path.join(save_dir, 'multi_head_attention_analysis.png')
            )
        
        # 8. 生成详细报告
        print(f"\n📝 第{'6' if is_multi_head else '5'}步: 生成分析报告")
        report = analyzer.generate_attention_report(
            attention_data,
            save_path=os.path.join(save_dir, 'attention_analysis_report.txt')
        )
        
        # 9. 输出结果总结
        print("\n🎉 注意力分析完成！")
        print("="*60)
        print(f"📁 结果保存在: {save_dir}")
        print("\n生成的文件:")
        print("  📈 attention_evolution_detailed.png     - 详细的注意力时序演化分析")
        print("  🎯 attention_accuracy_analysis.png      - 注意力与准确性关联分析")
        print("  🗺️ spatial_attention_analysis.png       - 空间注意力分布分析")
        if is_multi_head:
            print("  🧠 multi_head_attention_analysis.png    - 多头注意力专门分析")
        print("  📝 attention_analysis_report.txt        - 详细文字报告")
        print("  📊 attention_analysis_report.json       - 结构化数据报告")
        
        if report:
            print(f"\n📊 分析摘要:")
            summary = report['analysis_summary']
            print(f"   总样本数: {summary['total_samples']}")
            print(f"   序列长度: {summary['sequence_length']}")
            print(f"   空间维度: {summary['spatial_dimension']}")
            if summary['is_multi_head']:
                print(f"   注意力头数: {summary['num_heads']}")
            print(f"   整体准确率: {summary['overall_accuracy']:.4f}")
            
            if 'accuracy_correlation' in report:
                acc_corr = report['accuracy_correlation']
                print(f"   正确预测注意力集中度: {acc_corr['correct_mean_concentration']:.4f}")
                print(f"   错误预测注意力集中度: {acc_corr['error_mean_concentration']:.4f}")
            
            if 'multi_head_analysis' in report:
                mha = report['multi_head_analysis']
                print(f"   头间平均相关性: {mha['mean_inter_head_correlation']:.4f}")
        
        return attention_data, report
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='独立的注意力机制深度分析工具')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 可选参数
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大分析样本数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='数据加载批次大小')
    
    args = parser.parse_args()
    
    # 设置默认保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./attention_analysis_{timestamp}'
    
    # 检查输入文件
    for path, name in [(args.checkpoint, '检查点文件'), 
                       (args.val_csv, '验证CSV文件'), 
                       (args.data_root, '数据根目录')]:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    print("🔍 独立注意力机制分析工具")
    print("="*60)
    print(f"检查点: {args.checkpoint}")
    print(f"验证集: {args.val_csv}")
    print(f"数据根目录: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    print(f"最大样本数: {args.max_samples}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 执行完整的注意力分析
        results = comprehensive_attention_analysis(
            args.checkpoint, args.val_csv, args.data_root, 
            args.save_dir, args.max_samples
        )
        
        elapsed_time = time.time() - start_time
        
        if results:
            print(f"\n🎉 分析成功完成！")
            print(f"⏱️ 总耗时: {elapsed_time:.2f} 秒")
            print(f"📁 所有结果已保存到: {args.save_dir}")
        else:
            print(f"\n❌ 分析失败")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    if len(sys.argv) == 1:
        print("🔍 独立注意力机制深度分析工具")
        print("="*60)
        print("专门分析具身计数模型的注意力机制")
        print("支持单头和多头注意力机制")
        print("提供多维度、多时刻的深度注意力分析")
        print()
        print("使用方法:")
        print("python attention_analyzer.py \\")
        print("    --checkpoint MODEL.pth \\")
        print("    --val_csv VAL.csv \\")
        print("    --data_root DATA_DIR")
        print()
        print("可选参数:")
        print("  --save_dir DIR        # 结果保存目录")
        print("  --max_samples N       # 最大分析样本数 (默认500)")
        print("  --batch_size N        # 批次大小 (默认8)")
        print()
        print("示例:")
        print("python attention_analyzer.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --max_samples 500 \\")
        print("    --save_dir ./my_attention_analysis")
        print()
        print("生成的分析包括:")
        print("  📈 注意力时序演化分析")
        print("  🎯 注意力与预测准确性关联")
        print("  🗺️ 空间注意力分布分析")
        print("  🧠 多头注意力专门分析 (如适用)")
        print("  📝 详细量化分析报告")
        print()
        print("💡 注意: 模型必须支持return_attention=True参数")
        print("🔧 支持形状: (batch, seq_len, spatial) 或 (batch, seq_len, heads, spatial)")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷调用函数
# =============================================================================

def quick_attention_analysis(checkpoint_path, val_csv, data_root, save_dir=None):
    """快速注意力分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./quick_attention_{timestamp}'
    
    return comprehensive_attention_analysis(
        checkpoint_path, val_csv, data_root, save_dir, max_samples=100
    )


def detailed_attention_analysis(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=500):
    """详细注意力分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./detailed_attention_{timestamp}'
    
    return comprehensive_attention_analysis(
        checkpoint_path, val_csv, data_root, save_dir, max_samples
    )


"""
=============================================================================
使用说明
=============================================================================

这是一个修复后的独立注意力机制分析工具，现在支持多头注意力机制。

主要改进:
1. 支持多头注意力: 自动检测形状 (batch, seq_len, heads, spatial)
2. 多头专门分析: 头间相关性、多样性、效果对比等
3. 智能形状处理: 自动适配单头和多头注意力
4. 增强的可视化: 包含多头对比和分析

注意力数据形状支持:
- 单头: (n_samples, seq_len, spatial_size) 
- 多头: (n_samples, seq_len, n_heads, spatial_size)

命令行使用:
python attention_analyzer.py \\
    --checkpoint your_model.pth \\
    --val_csv your_val.csv \\
    --data_root your_data \\
    --max_samples 300

Python脚本使用:
from attention_analyzer import quick_attention_analysis, detailed_attention_analysis

# 快速分析
results = quick_attention_analysis(
    'model.pth', 'val.csv', 'data_dir'
)

# 详细分析
results = detailed_attention_analysis(
    'model.pth', 'val.csv', 'data_dir', max_samples=500
)

注意事项:
- 模型必须支持return_attention=True参数
- 自动检测并处理多头注意力
- 生成的分析包含多头特有的分析图表
- 报告包含多头注意力的详细统计
=============================================================================
"""