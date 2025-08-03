"""
CNN视觉模型可视化脚本 - 服务器版本
支持多层特征可视化、注意力热力图分析
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 导入你的模型和数据加载器
from Model_single_image import create_single_image_model
from DataLoader_single_image import get_single_image_data_loaders


class SingleImageModelVisualizer:
    """CNN视觉模型可视化器 - 服务器版本"""
    
    def __init__(self, model_path, config_path=None, data_root=None, output_dir="visualization_results"):
        """
        初始化可视化器
        
        Args:
            model_path: 训练好的模型检查点路径
            config_path: 配置文件路径 (可选)
            data_root: 数据根目录
            output_dir: 结果保存目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置matplotlib为非交互式后端
        plt.switch_backend('Agg')
        
        # 加载模型
        self.model, self.config = self._load_model(model_path, config_path)
        
        # 注册hooks用于特征提取
        self.feature_maps = {}
        self.attention_weights = None
        self._register_hooks()
        
        print(f"SingleImageModelVisualizer 初始化完成")
        print(f"模型设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        print(f"图像模式: {self.config.get('image_mode', 'rgb')}")
    
    def _load_model(self, model_path, config_path=None):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # 如果提供了额外的配置文件，则合并
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                extra_config = json.load(f)
                config.update(extra_config)
        
        # 创建模型
        model = create_single_image_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"模型加载成功")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, config
    
    def _register_hooks(self):
        """注册forward hooks来提取中间特征"""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # 如果返回多个值（比如attention有权重）
                    self.feature_maps[name] = output[0].detach().cpu()
                    if len(output) > 1:
                        self.attention_weights = output[1].detach().cpu()
                else:
                    self.feature_maps[name] = output.detach().cpu()
            return hook
        
        # 注册CNN层的hooks - 修复hook名称以匹配新的模型结构
        if hasattr(self.model, 'visual_encoder') and hasattr(self.model.visual_encoder, 'cnn'):
            cnn_layers = self.model.visual_encoder.cnn
            layer_count = 0
            for i, layer in enumerate(cnn_layers):
                if isinstance(layer, torch.nn.ReLU):
                    layer.register_forward_hook(make_hook(f'layer_{layer_count}_relu'))
                    layer_count += 1
        
        # 注册注意力层的hook（如果存在）
        if hasattr(self.model, 'spatial_attention') and self.model.use_attention:
            self.model.spatial_attention.register_forward_hook(make_hook('spatial_attention'))
    
    def _load_test_data(self, csv_path, num_samples=10):
        """加载测试数据"""
        if not self.data_root or not csv_path:
            print("警告: 未提供数据路径，将使用模拟数据")
            return None, None
            
        try:
            # 根据实际的DataLoader_single_image.py调用
            _, test_loader = get_single_image_data_loaders(
                train_csv_path=csv_path,
                val_csv_path=csv_path,
                data_root=self.data_root,
                batch_size=1,
                num_workers=0,
                image_mode=self.config.get('image_mode', 'rgb'),
                normalize_images=True,
                custom_image_norm_stats=None
            )
            
            # 获取指定数量的样本
            samples = []
            for i, batch in enumerate(test_loader):
                if i >= num_samples:
                    break
                samples.append(batch)
            
            print(f"成功加载 {len(samples)} 个测试样本")
            return samples, test_loader.dataset
            
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _denormalize_image(self, tensor):
        """反归一化图像用于显示"""
        # 确保在CPU上操作
        if tensor.is_cuda:
            tensor = tensor.cpu()
            
        if self.config.get('image_mode', 'rgb') == 'rgb':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            mean = torch.tensor([0.5]).view(1, 1, 1)
            std = torch.tensor([0.5]).view(1, 1, 1)
        
        denorm = tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        return denorm
    
    def _tensor_to_image(self, tensor):
        """将张量转换为可显示的图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.shape[0] == 1:
            # 灰度图像
            return tensor.squeeze(0).numpy()
        elif tensor.shape[0] == 3:
            # RGB图像
            return tensor.permute(1, 2, 0).numpy()
        else:
            return tensor.numpy()
    
    def _create_attention_heatmap(self, attention_weights, image_shape):
        """创建注意力热力图"""
        if attention_weights is None:
            return None
        
        # 确保在CPU上操作
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
        
        # attention_weights: [1, heads, HW, HW] or [heads, HW, HW]
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(0)
        
        # 对于自注意力，我们取平均或第一个查询的注意力
        if attention_weights.shape[-1] == attention_weights.shape[-2]:
            # 取对角线或平均
            attention_map = attention_weights.mean(0).mean(0)  # [HW]
        else:
            attention_map = attention_weights.mean(0)[0]  # 取第一个查询
        
        # 重塑为空间形状
        spatial_size = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map.view(spatial_size, spatial_size)
        
        # 上采样到图像大小
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=image_shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return attention_map.detach().numpy()
    
    def visualize_single_sample(self, sample_data, sample_idx, save_prefix="sample"):
        """可视化单个样本的详细分析"""
        print(f"可视化样本 {sample_idx}...")
        
        # 数据准备
        if sample_data is None:
            # 使用模拟数据
            channels = 3 if self.config.get('image_mode', 'rgb') == 'rgb' else 1
            image = torch.randn(1, channels, 224, 224).to(self.device)
            true_label = np.random.randint(1, 11)  # 1-10
            print(f"使用模拟数据进行测试")
        else:
            # 修复维度问题：从DataLoader来的数据可能是 [batch=1, channels, H, W]
            raw_image = sample_data['image']
            print(f"原始图像形状: {raw_image.shape}")
            
            # 处理可能的维度问题
            if raw_image.dim() == 5:  # [1, 1, channels, H, W]
                image = raw_image.squeeze(1).to(self.device)  # [1, channels, H, W]
            elif raw_image.dim() == 4:  # [1, channels, H, W] 或 [channels, H, W] 需要unsqueeze
                if raw_image.shape[0] == 1:  # 已经有batch维度
                    image = raw_image.to(self.device)
                else:  # 没有batch维度，需要添加
                    image = raw_image.unsqueeze(0).to(self.device)
            elif raw_image.dim() == 3:  # [channels, H, W]
                image = raw_image.unsqueeze(0).to(self.device)  # [1, channels, H, W]
            else:
                print(f"警告: 意外的图像维度: {raw_image.shape}")
                image = raw_image.unsqueeze(0).to(self.device)
            
            true_label = sample_data['label'].item() if hasattr(sample_data['label'], 'item') else sample_data['label']
            print(f"修正后图像形状: {image.shape}")
            print(f"样本信息: 真实标签={true_label}, 样本ID={sample_data.get('sample_id', 'N/A')}")
        
        # 清空之前的特征图
        self.feature_maps.clear()
        self.attention_weights = None
        
        # 前向传播
        with torch.no_grad():
            try:
                if self.model.use_attention:
                    logits, features, attention_weights = self.model(
                        image, return_features=True, return_attention=True
                    )
                    self.attention_weights = attention_weights
                else:
                    logits = self.model(image)
                    attention_weights = None
                
                # 预测结果
                probs = F.softmax(logits, dim=-1)
                pred_class_idx = torch.argmax(logits, dim=-1).item()  # 0-9
                pred_label = pred_class_idx + 1  # 转换为1-10标签
                confidence = probs.max().item()
                
            except Exception as e:
                print(f"模型前向传播失败: {e}")
                print(f"图像形状: {image.shape}")
                raise e
        
        print(f"预测结果: 预测={pred_label}, 置信度={confidence:.3f}")
        
        # 创建可视化布局
        fig = plt.figure(figsize=(20, 12))
        
        # 准备图像数据 - 修复图像显示问题
        original_image = image.squeeze(0).cpu()  # 先移到CPU并去掉batch维度
        
        # 检查图像是否已经归一化，如果是则反归一化
        if original_image.min() < -0.5 or original_image.max() > 1.5:
            # 图像可能已经标准化，需要反归一化
            denorm_image = self._denormalize_image(original_image)
        else:
            # 图像在合理范围内，直接使用
            denorm_image = torch.clamp(original_image, 0, 1)
        
        # 转换为显示格式
        display_image = self._tensor_to_image(denorm_image)
        
        # 打印调试信息
        print(f"处理后图像形状: {original_image.shape}")
        print(f"图像值范围: [{original_image.min():.3f}, {original_image.max():.3f}]")
        print(f"反归一化后范围: [{denorm_image.min():.3f}, {denorm_image.max():.3f}]")
        
        # 1. 原始图像 + 基本信息
        ax1 = plt.subplot(3, 4, 1)
        if len(display_image.shape) == 3:
            ax1.imshow(display_image)
        else:
            ax1.imshow(display_image, cmap='gray')
        ax1.set_title(f'Original Image\nTrue: {true_label}, Pred: {pred_label}\nConfidence: {confidence:.3f}')
        ax1.axis('off')
        
        # 2. 注意力热力图
        ax2 = plt.subplot(3, 4, 2)
        if self.model.use_attention and attention_weights is not None:
            attention_heatmap = self._create_attention_heatmap(attention_weights, image.shape)
            if attention_heatmap is not None:
                im = ax2.imshow(attention_heatmap, cmap='jet', alpha=0.6)
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                ax2.set_title('Attention Heatmap')
            else:
                ax2.text(0.5, 0.5, 'No Attention\nWeights', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Attention (N/A)')
        else:
            ax2.text(0.5, 0.5, 'Attention\nNot Used', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Attention (Disabled)')
        ax2.axis('off')
        
        # 3. 叠加注意力的原图
        ax3 = plt.subplot(3, 4, 3)
        if len(display_image.shape) == 3:
            ax3.imshow(display_image)
        else:
            ax3.imshow(display_image, cmap='gray')
        
        if self.model.use_attention and attention_weights is not None:
            attention_heatmap = self._create_attention_heatmap(attention_weights, image.shape)
            if attention_heatmap is not None:
                ax3.imshow(attention_heatmap, cmap='jet', alpha=0.4)
        ax3.set_title('Image + Attention Overlay')
        ax3.axis('off')
        
        # 4. 预测概率分布 (修复标签显示)
        ax4 = plt.subplot(3, 4, 4)
        probs_np = probs.squeeze().cpu().numpy()
        
        # 创建1-10的标签
        ball_counts = list(range(1, 11))
        bars = ax4.bar(range(10), probs_np, alpha=0.7)
        
        # 高亮预测标签
        pred_idx = pred_class_idx  # 0-9索引
        true_idx = true_label - 1  # 转换为0-9索引
        
        if pred_idx < len(bars):
            bars[pred_idx].set_color('red')
        if 0 <= true_idx < len(bars) and true_idx != pred_idx:
            bars[true_idx].set_edgecolor('green')
            bars[true_idx].set_linewidth(3)
            
        ax4.set_title('Class Probabilities')
        ax4.set_xlabel('Ball Count')
        ax4.set_ylabel('Probability')
        ax4.set_xticks(range(10))
        ax4.set_xticklabels([str(i) for i in ball_counts])
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Predicted: {pred_label}'),
            Patch(facecolor='white', edgecolor='green', linewidth=3, label=f'True: {true_label}')
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 5-8. CNN特征图（不同层）
        cnn_feature_names = [name for name in self.feature_maps.keys() if 'cnn_layer' in name or 'layer_' in name]
        for i, feature_name in enumerate(cnn_feature_names[:4]):
            ax = plt.subplot(3, 4, 5 + i)
            
            features = self.feature_maps[feature_name]
            if features.dim() == 4:
                features = features.squeeze(0)
            
            # 确保在CPU上
            if features.is_cuda:
                features = features.cpu()
            
            # 取前几个通道的平均或选择性显示
            if features.shape[0] > 1:
                # 显示通道平均
                feature_avg = features.mean(0)
            else:
                feature_avg = features.squeeze(0)
            
            im = ax.imshow(feature_avg.detach().numpy(), cmap='viridis')
            ax.set_title(f'{feature_name}\nShape: {list(features.shape)}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 9-12. 多头注意力可视化（如果有）
        if self.model.use_attention and attention_weights is not None and attention_weights.shape[1] > 1:
            for head_idx in range(min(4, attention_weights.shape[1])):
                ax = plt.subplot(3, 4, 9 + head_idx)
                
                head_attention = attention_weights[0, head_idx]  # [HW, HW]
                
                # 确保在CPU上
                if head_attention.is_cuda:
                    head_attention = head_attention.cpu()
                    
                if head_attention.shape[-1] == head_attention.shape[-2]:
                    head_map = head_attention.mean(0)  # 平均查询
                else:
                    head_map = head_attention[0]  # 第一个查询
                
                spatial_size = int(np.sqrt(head_map.shape[0]))
                head_map = head_map.view(spatial_size, spatial_size)
                
                im = ax.imshow(head_map.detach().numpy(), cmap='jet')
                ax.set_title(f'Attention Head {head_idx}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.output_dir, f"{save_prefix}_{sample_idx}_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"样本 {sample_idx} 分析结果保存到: {save_path}")
        return save_path
    
    def visualize_feature_evolution(self, sample_data, sample_idx, save_prefix="features"):
        """可视化特征在不同层的演化"""
        print(f"可视化样本 {sample_idx} 的特征演化...")
        
        # 数据准备 - 修复维度问题
        if sample_data is None:
            channels = 3 if self.config.get('image_mode', 'rgb') == 'rgb' else 1
            image = torch.randn(1, channels, 224, 224).to(self.device)
        else:
            # 修复维度问题
            raw_image = sample_data['image']
            
            if raw_image.dim() == 5:  # [1, 1, channels, H, W]
                image = raw_image.squeeze(1).to(self.device)  # [1, channels, H, W]
            elif raw_image.dim() == 4:  # [1, channels, H, W] 或需要添加batch维度
                if raw_image.shape[0] == 1:
                    image = raw_image.to(self.device)
                else:
                    image = raw_image.unsqueeze(0).to(self.device)
            elif raw_image.dim() == 3:  # [channels, H, W]
                image = raw_image.unsqueeze(0).to(self.device)  # [1, channels, H, W]
            else:
                image = raw_image.unsqueeze(0).to(self.device)
        
        # 清空并前向传播
        self.feature_maps.clear()
        with torch.no_grad():
            _ = self.model(image)
        
        # 获取CNN特征图
        cnn_features = [(name, feat) for name, feat in self.feature_maps.items() 
                    if 'cnn_layer' in name or 'layer_' in name]
        
        if not cnn_features:
            print("未找到CNN特征图")
            return None
        
        # 创建特征演化可视化
        num_layers = len(cnn_features)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, features) in enumerate(cnn_features):
            if features.dim() == 4:
                features = features.squeeze(0)
            
            # 确保在CPU上
            if features.is_cuda:
                features = features.cpu()
            
            # 上排：通道平均特征图
            if features.shape[0] > 1:
                avg_features = features.mean(0)
            else:
                avg_features = features.squeeze(0)
            
            im1 = axes[0, i].imshow(avg_features.detach().numpy(), cmap='viridis')
            axes[0, i].set_title(f'{name}\nAvg Features\n{list(features.shape)}')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 下排：最大激活通道
            if features.shape[0] > 1:
                max_channel_idx = features.view(features.shape[0], -1).mean(-1).argmax()
                max_features = features[max_channel_idx]
            else:
                max_features = features.squeeze(0)
            
            im2 = axes[1, i].imshow(max_features.detach().numpy(), cmap='plasma')
            axes[1, i].set_title(f'Max Activation Channel\nChannel {max_channel_idx.item() if features.shape[0] > 1 else 0}')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, f"{save_prefix}_{sample_idx}_evolution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征演化图保存到: {save_path}")
        return save_path
    
    def batch_analysis(self, csv_path=None, num_samples=10):
        """批量分析多个样本"""
        print(f"开始批量分析 {num_samples} 个样本...")
        
        # 加载测试数据
        samples, dataset = self._load_test_data(csv_path, num_samples=num_samples)
        
        # 分析每个样本
        results = []
        for i in range(num_samples):
            sample_data = samples[i] if samples else None
            
            try:
                # 单样本详细分析
                analysis_path = self.visualize_single_sample(sample_data, i, "batch_sample")
                
                # 特征演化分析
                evolution_path = self.visualize_feature_evolution(sample_data, i, "batch_features")
                
                results.append({
                    'sample_idx': i,
                    'analysis_path': analysis_path,
                    'evolution_path': evolution_path,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"分析样本 {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'sample_idx': i,
                    'analysis_path': None,
                    'evolution_path': None,
                    'status': f'error: {str(e)}'
                })
        
        # 生成汇总报告
        self._generate_batch_report(results, num_samples)
        
        print(f"批量分析完成，结果保存在: {self.output_dir}")
        return results
    
    def _generate_batch_report(self, results, num_samples):
        """生成批量分析报告"""
        report_content = f"""# CNN视觉模型批量分析报告

## 分析概况
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 模型路径: {self.model_path}
- 分析样本数: {num_samples}
- 成功分析: {sum(1 for r in results if r['status'] == 'success')}
- 失败样本: {sum(1 for r in results if r['status'] != 'success')}

## 模型配置
- 图像模式: {self.config.get('image_mode', 'N/A')}
- 使用注意力: {self.config.get('use_attention', 'N/A')}
- CNN层数: {self.config.get('model_config', {}).get('cnn_layers', 'N/A')}
- 特征维度: {self.config.get('model_config', {}).get('feature_dim', 'N/A')}

## 生成文件列表
"""
        
        for result in results:
            report_content += f"\n### 样本 {result['sample_idx']}\n"
            report_content += f"- 状态: {result['status']}\n"
            if result['analysis_path']:
                report_content += f"- 详细分析: {os.path.basename(result['analysis_path'])}\n"
            if result['evolution_path']:
                report_content += f"- 特征演化: {os.path.basename(result['evolution_path'])}\n"
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "batch_analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"批量分析报告保存到: {report_path}")


def main():
    """主函数 - 示例使用"""
    
    # 配置路径 - 请根据你的实际路径修改
    model_path = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/Result_data/single_CNN/checkpoints/best_single_image_model.pth"
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    test_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val2.csv"
    
    print("=== CNN视觉模型可视化分析 ===")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请确保已经训练了single image模型")
        return
    
    try:
        # 创建可视化器
        visualizer = SingleImageModelVisualizer(
            model_path=model_path,
            data_root=data_root,
            output_dir="single_image_visualization_results"
        )
        
        # 批量分析1-10个样本
        results = visualizer.batch_analysis(
            csv_path=test_csv,
            num_samples=10
        )
        
        print("\n=== 分析完成 ===")
        print(f"成功分析: {sum(1 for r in results if r['status'] == 'success')}/10 个样本")
        print(f"结果保存在: {visualizer.output_dir}")
        
        # 可以单独分析特定样本
        # visualizer.visualize_single_sample(sample_data=None, sample_idx=0, save_prefix="detailed")
        
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()