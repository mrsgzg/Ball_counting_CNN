"""
通用动态模型可视化工具
支持原始Embodiment模型和Ablation模型的可视化
可视化每个样本每一帧的:
1. Attention热力图 (如果模型支持)
2. Softmax输出分布
3. LSTM隐状态变化
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from datetime import datetime
from PIL import Image
import cv2

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class UniversalModelVisualizer:
    """通用模型可视化器 - 支持所有模型类型"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 检测模型类型和能力
        self.model_info = self._detect_model_capabilities()
        print(f"✅ 检测到模型类型: {self.model_info['model_type']}")
        print(f"   支持的功能: {', '.join(self.model_info['capabilities'])}")
        
    def _detect_model_capabilities(self):
        """检测模型类型和支持的功能"""
        model_info = {
            'model_type': 'unknown',
            'capabilities': [],
            'has_attention': False,
            'has_embodiment': False,
            'has_motion_decoder': False
        }
        
        # 检查模型类型
        model_class_name = self.model.__class__.__name__
        
        if hasattr(self.model, 'get_model_info'):
            # 新的ablation模型有get_model_info方法
            info = self.model.get_model_info()
            model_info.update(info)
            model_info['model_type'] = info.get('model_type', model_class_name)
            model_info['has_embodiment'] = info.get('has_embodiment', False)
            model_info['has_motion_decoder'] = info.get('has_motion_decoder', False)
        else:
            # 原始Embodiment模型
            model_info['model_type'] = 'EmbodiedCountingModel'
            model_info['has_embodiment'] = True
            model_info['has_motion_decoder'] = True
        
        # 检查是否有attention机制
        if hasattr(self.model, 'fusion') or hasattr(self.model, 'attention_weights_history'):
            model_info['has_attention'] = True
        
        # 确定支持的功能
        capabilities = ['counting', 'lstm_states']
        if model_info['has_attention']:
            capabilities.append('attention')
        if model_info['has_embodiment']:
            capabilities.append('embodiment')
        if model_info['has_motion_decoder']:
            capabilities.append('motion_prediction')
            
        model_info['capabilities'] = capabilities
        
        return model_info
    
    def _prepare_model_for_visualization(self):
        """为可视化准备模型"""
        # 清空历史记录
        if hasattr(self.model, 'lstm_hidden_states'):
            self.model.lstm_hidden_states = []
        if hasattr(self.model, 'attention_weights_history'):
            self.model.attention_weights_history = []
    
    def _get_model_outputs(self, sequence_data):
        """获取模型输出"""
        self._prepare_model_for_visualization()
        
        with torch.no_grad():
            # 根据模型类型调用forward
            if self.model_info['has_attention']:
                outputs = self.model(
                    sequence_data=sequence_data,
                    use_teacher_forcing=False,
                    return_attention=True
                )
            else:
                outputs = self.model(
                    sequence_data=sequence_data,
                    use_teacher_forcing=False
                )
        
        return outputs
    
    def _extract_visualization_data(self, outputs):
        """提取可视化数据"""
        viz_data = {}
        
        # 计数相关数据
        count_logits = outputs['counts'][0]  # [seq_len, 11]
        
        viz_data['softmax_probs'] = F.softmax(count_logits, dim=-1)  # [seq_len, 11]
        viz_data['predictions'] = torch.argmax(count_logits, dim=-1)  # [seq_len]
        
        # LSTM状态
        if hasattr(self.model, 'lstm_hidden_states') and self.model.lstm_hidden_states:
            viz_data['lstm_states'] = self.model.lstm_hidden_states
        else:
            viz_data['lstm_states'] = []
        
        # Attention权重
        if (self.model_info['has_attention'] and 
            hasattr(self.model, 'attention_weights_history') and 
            self.model.attention_weights_history):
            viz_data['attention_weights'] = self.model.attention_weights_history
        else:
            viz_data['attention_weights'] = []
        
        # 关节预测（如果有）
        if 'joints' in outputs:
            viz_data['joint_predictions'] = outputs['joints'][0]
        else:
            viz_data['joint_predictions'] = None
            
        return viz_data
    
    def visualize_sample_sequence(self, sample_data, sample_id, save_dir):
        """可视化单个样本的完整序列"""
        
        # 创建样本保存目录
        sample_dir = os.path.join(save_dir, f'sample_{sample_id}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 准备数据
        sequence_data = {
            'images': sample_data['sequence_data']['images'].unsqueeze(0).to(self.device),
            'joints': sample_data['sequence_data']['joints'].unsqueeze(0).to(self.device),
            'timestamps': sample_data['sequence_data']['timestamps'].unsqueeze(0).to(self.device),
            'labels': sample_data['sequence_data']['labels'].unsqueeze(0).to(self.device)
        }
        
        seq_len = sequence_data['images'].shape[1]
        true_label = sample_data['label'].item()
        
        print(f"\n处理样本 {sample_id} (真实标签: {true_label}, 序列长度: {seq_len})")
        
        # 获取模型输出和可视化数据
        outputs = self._get_model_outputs(sequence_data)
        viz_data = self._extract_visualization_data(outputs)
        
        # 生成每一帧的可视化
        self._generate_frame_visualizations(
            sequence_data, viz_data, true_label, sample_id, sample_dir, seq_len
        )
        
        # 生成汇总图
        self._generate_summary_plot(
            viz_data, sequence_data, true_label, sample_id, sample_dir
        )
        
        # 保存数值数据
        self._save_numerical_data(
            viz_data, true_label, sample_id, sample_dir
        )
        
        print(f"✅ 样本 {sample_id} 可视化完成")
    
    def _generate_frame_visualizations(self, sequence_data, viz_data, true_label, 
                                     sample_id, sample_dir, seq_len):
        """生成每一帧的可视化"""
        
        softmax_probs = viz_data['softmax_probs']
        predictions = viz_data['predictions']
        lstm_states = viz_data['lstm_states']
        attention_weights = viz_data['attention_weights']
        
        for t in range(seq_len):
            fig = self._create_frame_figure(
                sequence_data, viz_data, t, true_label, sample_id, seq_len
            )
            
            # 保存当前帧
            frame_path = os.path.join(sample_dir, f'frame_{t:03d}.png')
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _create_frame_figure(self, sequence_data, viz_data, t, true_label, sample_id, seq_len):
        """创建单帧可视化图"""
        
        # 根据模型能力确定子图布局
        if self.model_info['has_attention']:
            fig = plt.figure(figsize=(20, 12))
            subplot_layout = (2, 3)
        else:
            fig = plt.figure(figsize=(16, 10))
            subplot_layout = (2, 2)
        
        subplot_idx = 1
        
        # 1. 原始图像
        ax1 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
        self._plot_original_image(ax1, sequence_data['images'][0, t], t, seq_len)
        subplot_idx += 1
        
        # 2. Attention热力图 (如果支持)
        if self.model_info['has_attention']:
            ax2 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
            self._plot_attention_map(ax2, viz_data['attention_weights'], 
                                   sequence_data['images'][0, t], t)
            subplot_idx += 1
        
        # 3. Softmax输出分布
        ax3 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
        self._plot_softmax_distribution(ax3, viz_data['softmax_probs'], 
                                      viz_data['predictions'], t)
        subplot_idx += 1
        
        # 4. Softmax分布时序变化
        ax4 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
        self._plot_softmax_evolution(ax4, viz_data['softmax_probs'], 
                                   viz_data['predictions'], t, seq_len)
        subplot_idx += 1
        
        # 5. LSTM隐状态变化
        if subplot_idx <= subplot_layout[0] * subplot_layout[1]:
            ax5 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
            self._plot_lstm_states(ax5, viz_data['lstm_states'], t)
            subplot_idx += 1
        
        # 6. 预测序列和信息
        if subplot_idx <= subplot_layout[0] * subplot_layout[1]:
            ax6 = plt.subplot(subplot_layout[0], subplot_layout[1], subplot_idx)
            self._plot_prediction_sequence(ax6, viz_data['predictions'], 
                                         sequence_data['labels'][0], t, 
                                         true_label, sample_id)
        
        plt.tight_layout()
        return fig
    
    def _plot_original_image(self, ax, img_tensor, t, seq_len):
        """绘制原始图像"""
        img = img_tensor.cpu()
        
        if img.shape[0] == 1:  # 灰度图
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img = img.permute(1, 2, 0)
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            ax.imshow(img)
        
        ax.set_title(f'Frame {t+1}/{seq_len}')
        ax.axis('off')
    
    def _plot_attention_map(self, ax, attention_weights, img_tensor, t):
        """绘制attention热力图"""
        if t < len(attention_weights):
            att_weights = attention_weights[t].cpu().numpy().squeeze()
            spatial_size = len(att_weights)
            grid_size = int(np.sqrt(spatial_size))
            
            if grid_size * grid_size == spatial_size:
                att_map = att_weights.reshape(grid_size, grid_size)
                att_map_resized = cv2.resize(att_map, (224, 224))
                
                # 显示图像背景
                img = img_tensor.cpu()
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    ax.imshow(img, cmap='gray', alpha=0.5)
                else:
                    img = img.permute(1, 2, 0)
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)
                    ax.imshow(img, alpha=0.5)
                
                # 叠加attention
                im = ax.imshow(att_map_resized, cmap='viridis', alpha=0.5)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.bar(range(len(att_weights)), att_weights)
                ax.set_xlabel('Spatial Position')
                ax.set_ylabel('Attention Weight')
            
            ax.set_title(f'Attention Map (t={t+1})')
        else:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center')
            ax.set_title(f'Attention Map (t={t+1})')
    
    def _plot_softmax_distribution(self, ax, softmax_probs, predictions, t):
        """绘制Softmax分布"""
        probs_t = softmax_probs[t].cpu().numpy()
        pred_t = predictions[t].item()
        
        bars = ax.bar(range(11), probs_t, color='skyblue', alpha=0.7)
        bars[pred_t].set_color('red')
        
        # 添加概率值标签
        for i, prob in enumerate(probs_t):
            if prob > 0.01:
                ax.text(i, prob + 0.01, f'{prob:.2f}', ha='center', fontsize=8)
        
        ax.set_xlabel('Count Class')
        ax.set_ylabel('Probability')
        ax.set_title(f'Softmax Output (Pred: {pred_t})')
        ax.set_xticks(range(11))
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_softmax_evolution(self, ax, softmax_probs, predictions, t, seq_len):
        """绘制Softmax演化"""
        pred_t = predictions[t].item()
        
        for class_idx in range(11):
            probs_history = softmax_probs[:t+1, class_idx].cpu().numpy()
            if np.max(probs_history) > 0.1:
                ax.plot(range(t+1), probs_history, label=f'Class {class_idx}', 
                       linewidth=2 if class_idx == pred_t else 1,
                       alpha=1.0 if class_idx == pred_t else 0.5)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Probability')
        ax.set_title('Softmax Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, seq_len-0.5)
    
    def _plot_lstm_states(self, ax, lstm_states, t):
        """绘制LSTM状态 - 支持2D和3D PCA可视化"""
        if t < len(lstm_states):
            states_so_far = torch.stack(lstm_states[:t+1]).cpu().numpy()
            if len(states_so_far.shape) > 2:
                states_so_far = states_so_far.squeeze(1)
            
            if states_so_far.shape[0] > 3:  # 至少需要4个时间步才能做有意义的PCA
                try:
                    from sklearn.decomposition import PCA
                    
                    # 选择使用2D还是3D PCA
                    if states_so_far.shape[1] >= 3 and t >= 5:  # 如果维度够高且时间步够多，使用3D
                        try:
                            from mpl_toolkits.mplot3d import Axes3D
                            
                            pca = PCA(n_components=3)
                            states_pca = pca.fit_transform(states_so_far)
                            
                            # 清除当前axes的内容
                            ax.clear()
                            
                            # 将当前axes转换为3D
                            ax.remove()
                            fig = plt.gcf()  # 获取当前figure
                            
                            # 根据模型是否支持attention确定subplot位置
                            if self.model_info['has_attention']:
                                ax = fig.add_subplot(2, 3, 5, projection='3d')
                            else:
                                ax = fig.add_subplot(2, 2, 4, projection='3d')
                            
                            # 绘制3D轨迹
                            ax.plot(states_pca[:, 0], states_pca[:, 1], states_pca[:, 2], 
                                   'b-', alpha=0.6, linewidth=2)
                            
                            # 绘制时间步点
                            scatter = ax.scatter(states_pca[:, 0], states_pca[:, 1], states_pca[:, 2],
                                               c=range(t+1), cmap='viridis', s=60, alpha=0.8)
                            
                            # 突出显示当前点
                            ax.scatter(states_pca[-1, 0], states_pca[-1, 1], states_pca[-1, 2],
                                      color='red', s=120, marker='*', 
                                      edgecolor='black', linewidth=1, label='Current')
                            
                            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
                            ax.set_title('LSTM State Trajectory (3D PCA)')
                            
                            # 设置视角
                            ax.view_init(elev=20, azim=45)
                            
                            total_var = pca.explained_variance_ratio_.sum()
                            ax.text2D(0.02, 0.98, f'Total Var: {total_var:.1%}', 
                                     transform=ax.transAxes, fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            
                        except Exception as e3d:
                            print(f"3D PCA失败，回退到2D: {e3d}")
                            # 回退到2D PCA
                            self._plot_2d_pca(ax, states_so_far, t)
                    
                    else:
                        # 使用2D PCA
                        self._plot_2d_pca(ax, states_so_far, t)
                        
                except ImportError:
                    # 如果没有sklearn，显示前两个维度的原始值
                    self._plot_raw_states(ax, states_so_far)
                    
            else:
                # 时间步不够，显示原始维度
                self._plot_raw_states(ax, states_so_far)
        else:
            ax.text(0.5, 0.5, 'No LSTM state data', ha='center', va='center')
            ax.set_title('LSTM States (Not Available)')
        
        # 只有2D图才有grid方法
        if hasattr(ax, 'grid') and not hasattr(ax, 'zaxis'):
            ax.grid(True, alpha=0.3)
    
    def _plot_2d_pca(self, ax, states_so_far, t):
        """绘制2D PCA"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states_so_far)
        
        # 绘制2D轨迹
        ax.plot(states_2d[:, 0], states_2d[:, 1], 'b-', alpha=0.6, linewidth=2)
        ax.scatter(states_2d[:, 0], states_2d[:, 1], 
                  c=range(t+1), cmap='viridis', s=60, alpha=0.8)
        ax.scatter(states_2d[-1, 0], states_2d[-1, 1], 
                  color='red', s=100, marker='*', 
                  edgecolor='black', linewidth=1, label='Current')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('LSTM State Trajectory (2D PCA)')
        ax.legend()
        
        total_var = pca.explained_variance_ratio_.sum()
        ax.text(0.02, 0.98, f'Total Var: {total_var:.1%}', 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_raw_states(self, ax, states_so_far):
        """绘制原始状态值"""
        ax.plot(states_so_far[:, 0], 'b-', label='Dim 0', linewidth=2)
        if states_so_far.shape[1] > 1:
            ax.plot(states_so_far[:, 1], 'r-', label='Dim 1', linewidth=2)
        if states_so_far.shape[1] > 2:
            ax.plot(states_so_far[:, 2], 'g-', label='Dim 2', linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('State Value')
        ax.set_title('LSTM State Evolution (Raw)')
        ax.legend()
        ax.text(0.02, 0.98, 'Raw dimensions', 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    def _plot_prediction_sequence(self, ax, predictions, true_labels, t, 
                                true_label, sample_id):
        """绘制预测序列"""
        pred_seq = predictions[:t+1].cpu().numpy()
        true_seq = true_labels[:t+1].cpu().numpy()
        
        ax.plot(range(t+1), pred_seq, 'ro-', label='Predictions', markersize=8)
        ax.plot(range(t+1), true_seq, 'bs--', label='True Labels', markersize=6)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Sequence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 10.5)
        ax.set_yticks(range(11))
        
        # 添加信息文本
        pred_t = predictions[t].item()
        
        # 安全地计算置信度
        try:
            # 对当前时刻的logits计算softmax
            current_logits = predictions[t].float()
            if current_logits.dim() == 0:  # 如果是标量
                confidence = 1.0  # 或者设为合理的默认值
            else:
                current_probs = F.softmax(current_logits.unsqueeze(0), dim=-1)
                confidence = current_probs[pred_t].item()
        except:
            confidence = 0.0  # 如果计算失败，设为0
        
        info_text = f"Sample ID: {sample_id}\n"
        info_text += f"True Label: {true_label}\n"
        info_text += f"Current Pred: {pred_t}\n"
        info_text += f"Final Pred: {predictions[-1].item()}\n"
        info_text += f"Model: {self.model_info['model_type']}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
               verticalalignment='top', fontsize=10)
    
    def _generate_summary_plot(self, viz_data, sequence_data, true_label, 
                              sample_id, sample_dir):
        """生成汇总可视化图"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Softmax热力图
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(viz_data['softmax_probs'].cpu().numpy().T, 
                   cmap='YlOrRd', cbar=True, 
                   xticklabels=range(len(viz_data['softmax_probs'])),
                   yticklabels=range(11))
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Count Class')
        ax1.set_title('Softmax Probability Heatmap')
        
        # 2. 预测准确度时序图
        ax2 = plt.subplot(2, 2, 2)
        pred_np = viz_data['predictions'].cpu().numpy()
        true_np = sequence_data['labels'][0].cpu().numpy()
        correct = pred_np == true_np
        
        ax2.plot(range(len(pred_np)), pred_np, 'ro-', label='Predictions')
        ax2.plot(range(len(true_np)), true_np, 'bs--', label='True Labels')
        
        for i, is_correct in enumerate(correct):
            color = 'green' if is_correct else 'red'
            ax2.axvspan(i-0.3, i+0.3, alpha=0.2, color=color)
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. LSTM状态热力图
        ax3 = plt.subplot(2, 2, 3)
        if viz_data['lstm_states']:
            lstm_array = torch.stack(viz_data['lstm_states']).cpu().numpy()
            if len(lstm_array.shape) > 2:
                lstm_array = lstm_array.squeeze(1)
            
            n_dims_to_show = min(50, lstm_array.shape[1])
            sns.heatmap(lstm_array[:, :n_dims_to_show].T,
                       cmap='coolwarm', center=0,
                       xticklabels=range(len(viz_data['lstm_states'])),
                       yticklabels=False)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel(f'LSTM Hidden Dims (first {n_dims_to_show})')
            ax3.set_title('LSTM Hidden State Evolution')
        else:
            ax3.text(0.5, 0.5, 'No LSTM state data', ha='center', va='center')
            ax3.set_title('LSTM States (Not Available)')
        
        # 4. 置信度变化
        ax4 = plt.subplot(2, 2, 4)
        
        max_probs = torch.max(viz_data['softmax_probs'], dim=1)[0].cpu().numpy()
        
        correct_class_probs = []
        for t in range(len(viz_data['predictions'])):
            true_class = int(true_np[t])
            prob = viz_data['softmax_probs'][t, true_class].item()
            correct_class_probs.append(prob)
        
        ax4.plot(range(len(max_probs)), max_probs, 'b-', 
                label='Max Confidence', linewidth=2)
        ax4.plot(range(len(correct_class_probs)), correct_class_probs, 'g--', 
                label='True Class Prob', linewidth=2)
        
        ax4.fill_between(range(len(max_probs)), 0, max_probs, alpha=0.3)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Probability')
        ax4.set_title('Model Confidence Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        # 添加模型信息
        model_info_text = f"Model: {self.model_info['model_type']}\n"
        model_info_text += f"Capabilities: {', '.join(self.model_info['capabilities'])}"
        
        fig.suptitle(f'Sample {sample_id} Summary - True Label: {true_label}, '
                    f'Final Prediction: {viz_data["predictions"][-1].item()}\n'
                    f'{model_info_text}', fontsize=14)
        
        plt.tight_layout()
        
        summary_path = os.path.join(sample_dir, 'summary.png')
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _save_numerical_data(self, viz_data, true_label, sample_id, sample_dir):
        """保存数值数据"""
        
        data = {
            'sample_id': sample_id,
            'true_label': int(true_label),
            'predictions': viz_data['predictions'].cpu().tolist(),
            'softmax_probs': viz_data['softmax_probs'].cpu().tolist(),
            'final_prediction': int(viz_data['predictions'][-1].item()),
            'sequence_length': len(viz_data['predictions']),
            'accuracy': float((viz_data['predictions'].cpu().numpy() == true_label).mean()),
            'model_info': self.model_info
        }
        
        # LSTM状态统计
        if viz_data['lstm_states']:
            lstm_array = torch.stack(viz_data['lstm_states']).cpu().numpy()
            if len(lstm_array.shape) > 2:
                lstm_array = lstm_array.squeeze(1)
            
            data['lstm_stats'] = {
                'mean': float(lstm_array.mean()),
                'std': float(lstm_array.std()),
                'shape': list(lstm_array.shape)
            }
        
        # Attention统计
        if viz_data['attention_weights']:
            att_array = torch.cat(viz_data['attention_weights']).cpu().numpy()
            data['attention_stats'] = {
                'mean': float(att_array.mean()),
                'std': float(att_array.std()),
                'max': float(att_array.max()),
                'min': float(att_array.min())
            }
        
        json_path = os.path.join(sample_dir, 'data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)


def load_model_and_data(checkpoint_path, val_csv, data_root, device='cuda'):
    """通用模型和数据加载函数"""
    print("📥 加载模型和数据...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 导入模块
    from DataLoader_embodiment import BallCountingDataset
    
    # 确定图像模式
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
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
        model_config = config['model_config'].copy()
        model_config['input_channels'] = input_channels
        model = EmbodiedCountingModel(**model_config)
        print("✅ 加载原始具身计数模型")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"   图像模式: {image_mode}, 设备: {device}")
    
    # 创建数据集
    dataset = BallCountingDataset(
        csv_path=val_csv,
        data_root=data_root,
        sequence_length=config['sequence_length'],
        normalize=config['normalize'],
        image_mode=image_mode,
        normalize_images=True
    )
    
    print(f"✅ 数据集加载完成，样本数: {len(dataset)}")
    
    return model, dataset, device, config


def main():
    parser = argparse.ArgumentParser(description='通用动态模型可视化工具')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str, 
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val_single_per_label_v1.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    parser.add_argument('--save_dir', type=str, default='./universal_visualizations',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='要可视化的样本索引（默认全部）')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='最大可视化样本数')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'viz_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎬 通用动态模型可视化工具")
    print("="*50)
    print(f"检查点: {args.checkpoint}")
    print(f"数据集: {args.val_csv}")
    print(f"保存目录: {save_dir}")
    print("="*50)
    
    try:
        # 加载模型和数据
        model, dataset, device, config = load_model_and_data(
            args.checkpoint, args.val_csv, args.data_root, args.device
        )
        
        # 创建可视化器
        visualizer = UniversalModelVisualizer(model, device)
        
        # 确定要可视化的样本
        if args.sample_indices is not None:
            sample_indices = args.sample_indices
        else:
            sample_indices = list(range(min(len(dataset), args.max_samples)))
        
        print(f"\n准备可视化 {len(sample_indices)} 个样本: {sample_indices}")
        
        # 可视化每个样本
        for idx in sample_indices:
            if idx >= len(dataset):
                print(f"⚠️ 样本索引 {idx} 超出范围，跳过")
                continue
            
            try:
                sample_data = dataset[idx]
                visualizer.visualize_sample_sequence(sample_data, idx, save_dir)
            except Exception as e:
                print(f"❌ 样本 {idx} 可视化失败: {e}")
                continue
        
        print(f"\n🎉 可视化完成！")
        print(f"📁 结果保存在: {save_dir}")
        print(f"📊 每个样本的帧序列和汇总图已生成")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def generate_index_html(sample_indices, save_dir, model_info):
    """生成HTML索引页面 - 已移除，不再使用"""
    pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("🎬 通用动态模型可视化工具")
        print("="*50)
        print("支持所有模型类型的可视化:")
        print("  • 原始Embodiment模型 (完整功能)")
        print("  • Counting-Only模型 (无关节预测)")
        print("  • Visual-Only模型 (无具身信息)")
        print()
        print("自动检测模型能力并调整可视化内容:")
        print("  • Attention热力图 (如果支持)")
        print("  • Softmax输出分布")
        print("  • LSTM隐状态演化")
        print("  • 预测序列对比")
        print()
        print("使用方法:")
        print("python Universal_LSTM_Viz.py --checkpoint MODEL.pth --val_csv VAL.csv")
        print()
        print("示例:")
        print("# 可视化原始Embodiment模型")
        print("python Universal_LSTM_Viz.py \\")
        print("    --checkpoint ./best_embodied_model.pth \\")
        print("    --val_csv ./small_val_subset.csv")
        print()
        print("# 可视化Counting-Only消融模型")
        print("python Universal_LSTM_Viz.py \\")
        print("    --checkpoint ./best_counting_only_model.pth \\")
        print("    --val_csv ./small_val_subset.csv")
        print()
        print("# 可视化Visual-Only消融模型")
        print("python Universal_LSTM_Viz.py \\")
        print("    --checkpoint ./best_visual_only_model.pth \\")
        print("    --val_csv ./small_val_subset.csv")
        print()
        print("可选参数:")
        print("  --save_dir DIR          保存目录")
        print("  --device DEVICE         设备 (cuda/cpu)")
        print("  --sample_indices 0 1 2  指定样本索引")
        print("  --max_samples N         最大样本数")
        print()
        print("💡 新特性:")
        print("  • 自动检测模型类型和能力")
        print("  • 根据模型功能调整可视化布局")
        print("  • 统一的接口支持所有模型")
        print("  • 3D PCA可视化LSTM状态轨迹")
        print("  • 优化的代码结构和错误处理")
        print("  • 自动选择2D/3D PCA显示")
        sys.exit(0)
    
    main()