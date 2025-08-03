"""
动态具身计数模型可视化工具
可视化每个样本每一帧的:
1. Attention热力图
2. Softmax输出分布
3. LSTM隐状态变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from PIL import Image
import cv2

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DynamicModelVisualizer:
    """动态模型可视化器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 存储中间结果
        self.attention_weights_history = []
        self.lstm_states_history = []
        self.softmax_outputs_history = []
        
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
        
        # 清空历史记录
        self.model.lstm_hidden_states = []
        self.model.attention_weights_history = []
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                sequence_data=sequence_data,
                use_teacher_forcing=False,
                return_attention=True
            )
        
        # 提取结果
        count_logits = outputs['counts'][0]  # [seq_len, 11]
        softmax_probs = F.softmax(count_logits, dim=-1)  # [seq_len, 11]
        predictions = torch.argmax(count_logits, dim=-1)  # [seq_len]
        
        # 获取attention权重和LSTM状态
        attention_weights = self.model.attention_weights_history  # list of [1, H*W]
        lstm_states = self.model.lstm_hidden_states  # list of [lstm_hidden_size]
        
        # 1. 创建综合可视化图（每一帧）
        for t in range(seq_len):
            fig = plt.figure(figsize=(20, 12))
            
            # 1.1 原始图像
            ax1 = plt.subplot(2, 3, 1)
            img = sequence_data['images'][0, t].cpu()
            
            # 处理不同通道数的图像
            if img.shape[0] == 1:  # 灰度图
                img = img.squeeze(0)
                ax1.imshow(img, cmap='gray')
            else:  # RGB
                img = img.permute(1, 2, 0)
                # 反归一化
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                ax1.imshow(img)
            
            ax1.set_title(f'Frame {t+1}/{seq_len}')
            ax1.axis('off')
            
            # 1.2 Attention热力图
            ax2 = plt.subplot(2, 3, 2)
            if t < len(attention_weights):
                att_weights = attention_weights[t].cpu().numpy().squeeze()
                #att_weights = att_weights.mean(axis=0)  # 变成 (784,)
                #print(f"Frame {t}: Attention weights shape: {att_weights.shape}")
                # 尝试重塑为2D（假设是正方形）
                spatial_size = len(att_weights)
                #print(f"Spatial size: {spatial_size}")
                grid_size = int(np.sqrt(spatial_size))
                #print(f"Grid size: {grid_size}")
                if grid_size * grid_size == spatial_size:
                    att_map = att_weights.reshape(grid_size, grid_size)
                    
                    # 上采样到原始图像大小
                    att_map_resized = cv2.resize(att_map, (224, 224))
                    
                    # 叠加显示
                    if img.dim() == 2:  # 灰度图
                        ax2.imshow(img, cmap='gray', alpha=0.5)
                    else:  # RGB
                        ax2.imshow(img, alpha=0.5)
                    
                    im = ax2.imshow(att_map_resized, cmap='viridis', alpha=0.5)
                    plt.colorbar(im, ax=ax2, fraction=0.046)
                else:
                    # 如果不能重塑，显示1D attention
                    ax2.bar(range(len(att_weights)), att_weights)
                    ax2.set_xlabel('Spatial Position')
                    ax2.set_ylabel('Attention Weight')
                
                ax2.set_title(f'Attention Map (t={t+1})')
            else:
                ax2.text(0.5, 0.5, 'No attention data', ha='center', va='center')
                ax2.set_title(f'Attention Map (t={t+1})')
            
            # 1.3 Softmax输出分布
            ax3 = plt.subplot(2, 3, 3)
            probs_t = softmax_probs[t].cpu().numpy()
            pred_t = predictions[t].item()
            
            bars = ax3.bar(range(11), probs_t, color='skyblue', alpha=0.7)
            bars[pred_t].set_color('red')  # 高亮预测类别
            
            # 添加概率值标签
            for i, prob in enumerate(probs_t):
                if prob > 0.01:  # 只显示大于1%的概率
                    ax3.text(i, prob + 0.01, f'{prob:.2f}', ha='center', fontsize=8)
            
            ax3.set_xlabel('Count Class')
            ax3.set_ylabel('Probability')
            ax3.set_title(f'Softmax Output (Pred: {pred_t})')
            ax3.set_xticks(range(11))
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
            
            # 1.4 Softmax分布时序变化
            ax4 = plt.subplot(2, 3, 4)
            
            # 绘制前t+1帧的softmax变化
            for class_idx in range(11):
                probs_history = softmax_probs[:t+1, class_idx].cpu().numpy()
                if np.max(probs_history) > 0.1:  # 只显示重要的类别
                    ax4.plot(range(t+1), probs_history, label=f'Class {class_idx}', 
                            linewidth=2 if class_idx == pred_t else 1,
                            alpha=1.0 if class_idx == pred_t else 0.5)
            
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Probability')
            ax4.set_title('Softmax Evolution')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(-0.5, seq_len-0.5)
            
            # 1.5 LSTM隐状态变化
            ax5 = plt.subplot(2, 3, 5)
            
            if t < len(lstm_states):
                print(f"Frame {t}: LSTM states length: {len(lstm_states)}")
                print(f"Frame {t}: LSTM states shape: {lstm_states[t].shape}")
                # 显示LSTM状态的PCA投影（2D）
                states_so_far = torch.stack(lstm_states[:t+1]).cpu().numpy()
                states_so_far = states_so_far.squeeze(1)  # [t+1, lstm_hidden_size]
                print(f"Frame {t}: LSTM states shape: {states_so_far.shape}")
                if states_so_far.shape[0] > 2:
                    # 使用PCA降维到2D
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    states_2d = pca.fit_transform(states_so_far)
                    
                    # 绘制轨迹
                    ax5.plot(states_2d[:, 0], states_2d[:, 1], 'b-', alpha=0.5)
                    ax5.scatter(states_2d[:, 0], states_2d[:, 1], 
                              c=range(t+1), cmap='viridis', s=50)
                    ax5.scatter(states_2d[-1, 0], states_2d[-1, 1], 
                              color='red', s=100, marker='*', label='Current')
                    
                    ax5.set_xlabel('PC1')
                    ax5.set_ylabel('PC2')
                    ax5.set_title(f'LSTM State Trajectory (PCA)')
                    ax5.legend()
                else:
                    # 如果维度已经很低，直接显示
                    ax5.plot(states_so_far[:, 0], 'b-', label='Dim 0')
                    if states_so_far.shape[1] > 1:
                        ax5.plot(states_so_far[:, 1], 'r-', label='Dim 1')
                    ax5.set_xlabel('Frame')
                    ax5.set_ylabel('State Value')
                    ax5.set_title('LSTM State Evolution')
                    ax5.legend()
            
            ax5.grid(True, alpha=0.3)
            
            # 1.6 预测序列和信息
            ax6 = plt.subplot(2, 3, 6)
            
            # 显示预测序列
            pred_seq = predictions[:t+1].cpu().numpy()
            true_seq = sequence_data['labels'][0, :t+1].cpu().numpy()
            
            ax6.plot(range(t+1), pred_seq, 'ro-', label='Predictions', markersize=8)
            ax6.plot(range(t+1), true_seq, 'bs--', label='True Labels', markersize=6)
            
            ax6.set_xlabel('Frame')
            ax6.set_ylabel('Count')
            ax6.set_title('Prediction Sequence')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(-0.5, 10.5)
            ax6.set_yticks(range(11))
            
            # 添加文本信息
            info_text = f"Sample ID: {sample_id}\n"
            info_text += f"True Label: {true_label}\n"
            info_text += f"Current Pred: {pred_t}\n"
            info_text += f"Final Pred: {predictions[-1].item()}\n"
            info_text += f"Confidence: {probs_t[pred_t]:.3f}"
            
            ax6.text(0.02, 0.98, info_text, transform=ax6.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                    verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            
            # 保存当前帧
            frame_path = os.path.join(sample_dir, f'frame_{t:03d}.png')
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. 生成汇总图
        self._generate_summary_plot(
            softmax_probs, predictions, lstm_states, attention_weights,
            sequence_data, true_label, sample_id, sample_dir
        )
        
        # 3. 保存数值数据
        self._save_numerical_data(
            softmax_probs, predictions, lstm_states, attention_weights,
            true_label, sample_id, sample_dir
        )
        
        print(f"✅ 样本 {sample_id} 可视化完成")
        
    def _generate_summary_plot(self, softmax_probs, predictions, lstm_states, 
                              attention_weights, sequence_data, true_label, 
                              sample_id, sample_dir):
        """生成汇总可视化图"""
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Softmax热力图
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(softmax_probs.cpu().numpy().T, 
                   cmap='YlOrRd', cbar=True, 
                   xticklabels=range(len(softmax_probs)),
                   yticklabels=range(11))
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Count Class')
        ax1.set_title('Softmax Probability Heatmap')
        
        # 2. 预测准确度时序图
        ax2 = plt.subplot(2, 2, 2)
        pred_np = predictions.cpu().numpy()
        true_np = sequence_data['labels'][0].cpu().numpy()
        correct = pred_np == true_np
        
        ax2.plot(range(len(pred_np)), pred_np, 'ro-', label='Predictions')
        ax2.plot(range(len(true_np)), true_np, 'bs--', label='True Labels')
        
        # 标记正确和错误的预测
        for i, is_correct in enumerate(correct):
            color = 'green' if is_correct else 'red'
            ax2.axvspan(i-0.3, i+0.3, alpha=0.2, color=color)
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. LSTM状态热力图（显示部分维度）
        ax3 = plt.subplot(2, 2, 3)
        if lstm_states:
            lstm_array = torch.stack(lstm_states).cpu().numpy()
            #print(f"LSTM states shape: {lstm_array.shape}")
            # 只显示前50个维度
            n_dims_to_show = min(50, lstm_array.shape[1])
            sns.heatmap(lstm_array[:, :n_dims_to_show].T.squeeze(1),
                       cmap='coolwarm', center=0,
                       xticklabels=range(len(lstm_states)),
                       yticklabels=False)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel(f'LSTM Hidden Dims (first {n_dims_to_show})')
            ax3.set_title('LSTM Hidden State Evolution')
        
        # 4. 置信度变化
        ax4 = plt.subplot(2, 2, 4)
        
        # 最高概率值（置信度）
        max_probs = torch.max(softmax_probs, dim=1)[0].cpu().numpy()
        
        # 正确预测的置信度
        correct_class_probs = []
        for t in range(len(predictions)):
            true_class = int(true_np[t])  # 确保索引为整数
            #print(f"Frame {t}: True class {true_class}, Predicted {predictions[t].item()}")
            prob = softmax_probs[t, true_class].item()
            correct_class_probs.append(prob)
        
        ax4.plot(range(len(max_probs)), max_probs, 'b-', label='Max Confidence', linewidth=2)
        ax4.plot(range(len(correct_class_probs)), correct_class_probs, 'g--', 
                label='True Class Prob', linewidth=2)
        
        ax4.fill_between(range(len(max_probs)), 0, max_probs, alpha=0.3)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Probability')
        ax4.set_title('Model Confidence Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        # 添加总体信息
        fig.suptitle(f'Sample {sample_id} Summary - True Label: {true_label}, '
                    f'Final Prediction: {predictions[-1].item()}', fontsize=16)
        
        plt.tight_layout()
        
        # 保存汇总图
        summary_path = os.path.join(sample_dir, 'summary.png')
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    def _save_numerical_data(self, softmax_probs, predictions, lstm_states, 
                           attention_weights, true_label, sample_id, sample_dir):
        """保存数值数据为JSON格式"""
        
        data = {
            'sample_id': sample_id,
            'true_label': int(true_label),
            'predictions': predictions.cpu().tolist(),
            'softmax_probs': softmax_probs.cpu().tolist(),
            'final_prediction': int(predictions[-1].item()),
            'sequence_length': len(predictions),
            'accuracy': float((predictions.cpu().numpy() == true_label).mean())
        }
        
        # 保存LSTM状态统计信息
        if lstm_states:
            lstm_array = torch.stack(lstm_states).cpu().numpy()
            data['lstm_stats'] = {
                'mean': float(lstm_array.mean()),
                'std': float(lstm_array.std()),
                'shape': list(lstm_array.shape)
            }
        
        # 保存attention统计信息
        if attention_weights:
            att_array = torch.cat(attention_weights).cpu().numpy()
            data['attention_stats'] = {
                'mean': float(att_array.mean()),
                'std': float(att_array.std()),
                'max': float(att_array.max()),
                'min': float(att_array.min())
            }
        
        # 保存JSON文件
        json_path = os.path.join(sample_dir, 'data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)


def load_model_and_data(checkpoint_path, val_csv, data_root, device='cuda'):
    """加载模型和数据"""
    print("📥 加载模型和数据...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 导入必要的模块
    from Model_embodiment import EmbodiedCountingModel
    from DataLoader_embodiment import BallCountingDataset
    
    # 确定图像模式
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    # 重建模型
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    model = EmbodiedCountingModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成 (图像模式: {image_mode}, 设备: {device})")
    
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
    parser = argparse.ArgumentParser(description='动态具身计数模型可视化')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val_single_per_label_v1.csv',
                       help='验证集CSV文件路径（建议使用小子集）')
    parser.add_argument('--data_root', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    
    # 可选参数
    parser.add_argument('--save_dir', type=str, default='./dynamic_visualizations',
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
    
    print("🎬 动态具身计数模型可视化工具")
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
        visualizer = DynamicModelVisualizer(model, device)
        
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
            
            sample_data = dataset[idx]
            visualizer.visualize_sample_sequence(sample_data, idx, save_dir)
        
        # 生成索引页面
        generate_index_html(sample_indices, save_dir)
        
        print(f"\n🎉 可视化完成！")
        print(f"📁 结果保存在: {save_dir}")
        print(f"🌐 打开 {os.path.join(save_dir, 'index.html')} 查看结果")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def generate_index_html(sample_indices, save_dir):
    """生成HTML索引页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Model Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .sample-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .sample-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .sample-card img {
                max-width: 100%;
                border-radius: 4px;
            }
            .sample-card h3 {
                margin: 10px 0;
                color: #555;
            }
            .sample-card a {
                display: inline-block;
                margin: 5px;
                padding: 8px 15px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
            }
            .sample-card a:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Dynamic Embodied Counting Model Visualization</h1>
        <div class="sample-grid">
    """
    
    for idx in sample_indices:
        sample_dir = f'sample_{idx}'
        summary_path = f'{sample_dir}/summary.png'
        
        html_content += f"""
            <div class="sample-card">
                <h3>Sample {idx}</h3>
                <img src="{summary_path}" alt="Sample {idx} Summary">
                <br>
                <a href="{sample_dir}/" target="_blank">View Frames</a>
                <a href="{sample_dir}/data.json" target="_blank">View Data</a>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    index_path = os.path.join(save_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ 索引页面已生成: {index_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("🎬 动态具身计数模型可视化工具")
        print("="*50)
        print("此工具可视化模型对每个样本序列的动态处理过程")
        print("\n功能包括:")
        print("  • 每一帧的attention热力图")
        print("  • Softmax输出分布的动态变化")
        print("  • LSTM隐状态的演化轨迹")
        print("  • 预测序列与真实标签的对比")
        print("\n使用方法:")
        print("python dynamic_viz.py --checkpoint MODEL.pth --val_csv VAL.csv --data_root DATA_DIR")
        print("\n示例:")
        print("python dynamic_viz.py \\")
        print("    --checkpoint ./best_model.pth \\")
        print("    --val_csv ./small_val_subset.csv \\")
        print("    --data_root ./data \\")
        print("    --max_samples 5")
        print("\n可选参数:")
        print("  --save_dir DIR          保存目录 (默认: ./dynamic_visualizations)")
        print("  --device DEVICE         设备 (默认: cuda)")
        print("  --sample_indices 0 1 2  指定样本索引")
        print("  --max_samples N         最大样本数 (默认: 10)")
        print("\n💡 建议:")
        print("  • 准备一个小的验证子集（10-20个样本）")
        print("  • 每个样本会生成序列长度个图片，注意存储空间")
        print("  • 生成的index.html可以方便地浏览所有结果")
        sys.exit(0)
    
    main()