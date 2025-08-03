"""
通用模型特征分析工具
支持原始Embodiment模型和所有Ablation模型
专注于降维可视化：PCA和t-SNE的2D/3D展示
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys
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


class UniversalFeatureExtractor:
    """通用特征提取器 - 支持所有模型类型"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.features = {}
        self.hooks = []
        
        # 检测模型类型
        self.model_info = self._detect_model_type()
        print(f"✅ 检测到模型类型: {self.model_info['type']}")
        print(f"   支持的组件: {', '.join(self.model_info['available_components'])}")
        
    def _detect_model_type(self):
        """检测模型类型和可用组件"""
        model_class_name = self.model.__class__.__name__
        
        # 检查模型类型
        if hasattr(self.model, 'get_model_info'):
            # 新的ablation模型
            info = self.model.get_model_info()
            model_type = info.get('model_type', model_class_name)
        else:
            # 原始Embodiment模型
            model_type = 'EmbodiedCountingModel'
        
        # 检测可用组件
        available_components = []
        
        # 通用组件（所有模型都有）
        if hasattr(self.model, 'counting_decoder'):
            available_components.append('counting_decoder')
        if hasattr(self.model, 'lstm'):
            available_components.append('lstm')
        if hasattr(self.model, 'visual_encoder'):
            available_components.append('visual_encoder')
        
        # 特定组件
        if hasattr(self.model, 'embodiment_encoder'):
            available_components.append('embodiment_encoder')
        if hasattr(self.model, 'fusion'):
            available_components.append('fusion')
        if hasattr(self.model, 'motion_decoder'):
            available_components.append('motion_decoder')
        
        return {
            'type': model_type,
            'available_components': available_components
        }
    
    def get_recommended_components(self):
        """获取推荐分析的组件"""
        all_components = self.model_info['available_components']
        
        # 按重要性排序
        priority_order = [
            'fusion',              # 多模态融合（最重要）
            'lstm',                # 时序处理
            'counting_decoder',    # 计数解码
            'visual_encoder',      # 视觉编码
            'embodiment_encoder',  # 具身编码
            'motion_decoder'       # 动作解码
        ]
        
        recommended = []
        for component in priority_order:
            if component in all_components:
                recommended.append(component)
        
        return recommended
        
    def register_hooks(self, components_to_extract=None):
        """注册钩子函数来提取组件特征"""
        if components_to_extract is None:
            components_to_extract = self.get_recommended_components()
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0].detach().cpu()
                else:
                    self.features[name] = output.detach().cpu()
            return hook
        
        successful_hooks = []
        failed_hooks = []
        
        for component_name in components_to_extract:
            try:
                # 直接获取组件模块
                if hasattr(self.model, component_name):
                    module = getattr(self.model, component_name)
                    handle = module.register_forward_hook(get_activation(component_name))
                    self.hooks.append(handle)
                    successful_hooks.append(component_name)
                    print(f"✅ 成功注册钩子: {component_name}")
                else:
                    failed_hooks.append(component_name)
                    print(f"❌ 组件不存在: {component_name}")
                
            except Exception as e:
                failed_hooks.append(component_name)
                print(f"❌ 注册钩子失败: {component_name} - {e}")
        
        print(f"\n钩子注册结果: 成功 {len(successful_hooks)}, 失败 {len(failed_hooks)}")
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
    
    def extract_features(self, data_loader, max_samples=500):
        """提取特征 - 通用于所有模型类型"""
        all_features = defaultdict(list)
        all_labels = []
        all_sample_ids = []
        
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
                labels = batch['label'].cpu().numpy()
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
                
                # 前向传播 - 根据模型类型调用
                if self.model_info['type'] in ['EmbodiedCountingOnly', 'VisualOnlyCountingModel']:
                    # Ablation模型
                    outputs = self.model(sequence_data=sequence_data)
                else:
                    # 原始Embodiment模型
                    outputs = self.model(
                        sequence_data=sequence_data,
                        use_teacher_forcing=False
                    )
                
                # 收集标签和ID
                all_labels.extend(labels)
                all_sample_ids.extend(sample_ids)
                
                # 收集中间层特征
                for component_name, feature_tensor in self.features.items():
                    processed_features = self._process_feature_tensor(feature_tensor)
                    all_features[component_name].append(processed_features)
                
                sample_count += actual_batch_size
                
                if sample_count >= max_samples:
                    break
        
        # 合并所有特征
        final_features = {}
        for component_name, feature_list in all_features.items():
            if feature_list:
                final_features[component_name] = np.vstack(feature_list)
        
        result = {
            'features': final_features,
            'labels': np.array(all_labels),
            'sample_ids': all_sample_ids,
            'model_type': self.model_info['type']
        }
        
        print(f"\n特征提取完成:")
        print(f"  实际样本数: {len(result['labels'])}")
        print(f"  标签范围: {result['labels'].min()} - {result['labels'].max()}")
        print(f"  提取的组件: {list(final_features.keys())}")
        for name, features in final_features.items():
            print(f"    {name}: {features.shape}")
        
        return result


class VisualizationEngine:
    """可视化引擎 - 专注于降维可视化"""
    
    def __init__(self, figsize_2d=(12, 8), figsize_3d=(10, 8)):
        self.figsize_2d = figsize_2d
        self.figsize_3d = figsize_3d
        
    def reduce_dimensions(self, features, method='tsne', n_components=2):
        """降维"""
        print(f"  执行{method.upper()} {n_components}D降维...")
        
        if method == 'tsne':
            perplexity = min(30, len(features)//4, 50)
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=perplexity, max_iter=1000)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        reduced_features = reducer.fit_transform(features)
        
        # 返回降维结果和解释方差比例（如果是PCA）
        info = {'method': method, 'n_components': n_components}
        if method == 'pca':
            info['explained_variance_ratio'] = reducer.explained_variance_ratio_
            info['total_variance'] = reducer.explained_variance_ratio_.sum()
        
        return reduced_features, info
    
    def plot_2d_scatter(self, features_2d, labels, title, info, save_path):
        """绘制2D散点图"""
        plt.figure(figsize=self.figsize_2d)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Count {label}', alpha=0.7, s=50)
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel(f'Component 1', fontsize=12)
        plt.ylabel(f'Component 2', fontsize=12)
        
        # 添加方差解释比例（如果是PCA）
        if 'explained_variance_ratio' in info:
            plt.xlabel(f'PC1 ({info["explained_variance_ratio"][0]:.1%})', fontsize=12)
            plt.ylabel(f'PC2 ({info["explained_variance_ratio"][1]:.1%})', fontsize=12)
            plt.text(0.02, 0.98, f'Total Variance: {info["total_variance"]:.1%}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ 保存2D图: {os.path.basename(save_path)}")
    
    def plot_3d_scatter(self, features_3d, labels, title, info, save_path):
        """绘制3D散点图"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=self.figsize_3d)
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                      c=[colors[i]], label=f'Count {label}', alpha=0.7, s=50)
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # 设置坐标轴标签
        if 'explained_variance_ratio' in info:
            ax.set_xlabel(f'PC1 ({info["explained_variance_ratio"][0]:.1%})')
            ax.set_ylabel(f'PC2 ({info["explained_variance_ratio"][1]:.1%})')
            ax.set_zlabel(f'PC3 ({info["explained_variance_ratio"][2]:.1%})')
            
            # 添加总方差解释
            ax.text2D(0.02, 0.98, f'Total Variance: {info["total_variance"]:.1%}', 
                     transform=ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ 保存3D图: {os.path.basename(save_path)}")
    
    def create_component_visualizations(self, features_dict, labels, model_type, save_dir):
        """为每个组件创建可视化"""
        print(f"\n🎨 创建可视化图表...")
        
        for component_name, features in features_dict.items():
            print(f"\n📊 处理组件: {component_name}")
            print(f"   特征形状: {features.shape}")
            
            component_dir = os.path.join(save_dir, component_name)
            
            # PCA 2D
            features_2d, info_2d = self.reduce_dimensions(features, 'pca', 2)
            title_2d = f'{model_type} - {component_name} (PCA 2D)'
            save_path_2d = os.path.join(component_dir, f'{component_name}_pca_2d.png')
            self.plot_2d_scatter(features_2d, labels, title_2d, info_2d, save_path_2d)
            
            # PCA 3D
            if features.shape[1] >= 3:  # 确保特征维度足够
                features_3d, info_3d = self.reduce_dimensions(features, 'pca', 3)
                title_3d = f'{model_type} - {component_name} (PCA 3D)'
                save_path_3d = os.path.join(component_dir, f'{component_name}_pca_3d.png')
                self.plot_3d_scatter(features_3d, labels, title_3d, info_3d, save_path_3d)
            
            # t-SNE 2D
            if len(features) > 50:  # t-SNE需要足够的样本
                features_2d_tsne, info_2d_tsne = self.reduce_dimensions(features, 'tsne', 2)
                title_2d_tsne = f'{model_type} - {component_name} (t-SNE 2D)'
                save_path_2d_tsne = os.path.join(component_dir, f'{component_name}_tsne_2d.png')
                self.plot_2d_scatter(features_2d_tsne, labels, title_2d_tsne, info_2d_tsne, save_path_2d_tsne)
                
                # t-SNE 3D
                if features.shape[1] >= 3:
                    features_3d_tsne, info_3d_tsne = self.reduce_dimensions(features, 'tsne', 3)
                    title_3d_tsne = f'{model_type} - {component_name} (t-SNE 3D)'
                    save_path_3d_tsne = os.path.join(component_dir, f'{component_name}_tsne_3d.png')
                    self.plot_3d_scatter(features_3d_tsne, labels, title_3d_tsne, info_3d_tsne, save_path_3d_tsne)
            else:
                print(f"    ⚠️ 样本数不足，跳过t-SNE分析")


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


def analyze_model(checkpoint_path, val_csv, data_root, save_dir, 
                 max_samples=500, components=None):
    """主分析函数"""
    
    print("🔬 开始模型特征分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. 加载模型和数据
        model, val_loader, device, config = load_model_and_data(
            checkpoint_path, val_csv, data_root
        )
        
        # 2. 创建特征提取器
        extractor = UniversalFeatureExtractor(model, device)
        
        # 3. 确定要分析的组件
        if components is None:
            components = extractor.get_recommended_components()
        
        print(f"📋 准备分析的组件: {components}")
        
        # 4. 注册钩子并提取特征
        successful_components = extractor.register_hooks(components)
        
        if not successful_components:
            print("❌ 没有成功注册任何钩子！")
            return None
        
        try:
            # 5. 提取特征
            print("🎯 提取组件特征...")
            data = extractor.extract_features(val_loader, max_samples)
            
            features = data['features']
            labels = data['labels']
            model_type = data['model_type']
            
            if not features:
                print("❌ 没有提取到任何特征！")
                return None
            
            # 6. 创建可视化
            visualizer = VisualizationEngine()
            visualizer.create_component_visualizations(
                features, labels, model_type, save_dir
            )
            
            print(f"\n🎉 分析完成！")
            print(f"📁 结果保存在: {save_dir}")
            print(f"📊 生成的可视化:")
            for component in features.keys():
                print(f"   • {component}: PCA/t-SNE 2D/3D")
            
            return {
                'features': features,
                'labels': labels,
                'model_type': model_type,
                'components_analyzed': list(features.keys())
            }
            
        finally:
            extractor.remove_hooks()
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_model(checkpoint_path):
    """检查模型结构和可用组件"""
    print("🔍 检查模型结构...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        model_type = checkpoint.get('model_type', 'embodied')
        
        # 加载模型
        if model_type in ['counting_only', 'visual_only']:
            from Model_embodiment_ablation import create_ablation_model
            model = create_ablation_model(model_type, config)
        else:
            from Model_embodiment import EmbodiedCountingModel
            image_mode = config.get('image_mode', 'rgb')
            input_channels = 3 if image_mode == 'rgb' else 1
            model_config = config['model_config'].copy()
            model_config['input_channels'] = input_channels
            model = EmbodiedCountingModel(**model_config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 创建特征提取器来检查组件
        extractor = UniversalFeatureExtractor(model, device)
        recommended = extractor.get_recommended_components()
        
        print(f"\n📋 模型信息:")
        print(f"   类型: {extractor.model_info['type']}")
        print(f"   可用组件: {extractor.model_info['available_components']}")
        print(f"   推荐分析: {recommended}")
        
        return recommended
        
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        return None


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='通用模型特征分析工具')
    
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
    parser.add_argument('--mode', type=str, default='analyze',
                       choices=['inspect', 'analyze'],
                       help='运行模式')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='最大分析样本数')
    parser.add_argument('--components', nargs='+', default=None,
                       help='指定要分析的组件名称')
    
    # 其他选项
    parser.add_argument('--batch_size', type=int, default=16,
                       help='数据加载批次大小')
    
    args = parser.parse_args()
    
    # 设置默认保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(args.checkpoint).replace('.pth', '')
        args.save_dir = f'./analysis_{model_name}_{timestamp}'
    
    # 检查输入文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    print("🔬 通用模型特征分析工具")
    print("="*50)
    print(f"模式: {args.mode}")
    print(f"检查点: {args.checkpoint}")
    if args.mode == 'analyze':
        print(f"验证集: {args.val_csv}")
        print(f"数据根目录: {args.data_root}")
        print(f"保存目录: {args.save_dir}")
        print(f"最大样本数: {args.max_samples}")
    print("="*50)
    
    start_time = time.time()
    
    try:
        if args.mode == 'inspect':
            recommended = inspect_model(args.checkpoint)
            if recommended:
                print(f"\n💡 使用建议:")
                print(f"python {sys.argv[0]} \\")
                print(f"    --checkpoint {args.checkpoint} \\")
                print(f"    --mode analyze \\")
                print(f"    --components {' '.join(recommended[:3])}")  # 只显示前3个
        
        elif args.mode == 'analyze':
            # 检查其他必需文件
            for path, name in [(args.val_csv, '验证CSV文件'), 
                              (args.data_root, '数据根目录')]:
                if not os.path.exists(path):
                    print(f"❌ {name}不存在: {path}")
                    return
            
            results = analyze_model(
                args.checkpoint, args.val_csv, args.data_root, 
                args.save_dir, args.max_samples, args.components
            )
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 完成！")
        print(f"⏱️ 总耗时: {elapsed_time:.2f} 秒")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    if len(sys.argv) == 1:
        print("🔬 通用模型特征分析工具")
        print("="*50)
        print("支持所有模型类型的特征降维可视化")
        print("  • 原始Embodiment模型")
        print("  • Counting-Only消融模型")
        print("  • Visual-Only消融模型")
        print()
        print("功能特色:")
        print("  • 自动检测模型类型和可用组件")
        print("  • PCA和t-SNE降维可视化")
        print("  • 2D和3D可视化")
        print("  • 按组件分类展示")
        print()
        print("使用方法:")
        print("1. 检查模型结构:")
        print("   python Universal_Model_Analysis.py --checkpoint MODEL.pth --mode inspect")
        print()
        print("2. 完整分析:")
        print("   python Universal_Model_Analysis.py \\")
        print("       --checkpoint MODEL.pth \\")
        print("       --val_csv VAL.csv \\")
        print("       --data_root DATA_DIR \\")
        print("       --mode analyze")
        print()
        print("示例:")
        print("# 分析原始Embodiment模型")
        print("python Universal_Model_Analysis.py \\")
        print("    --checkpoint ./best_embodied_model.pth \\")
        print("    --val_csv ./val_subset.csv \\")
        print("    --data_root ./data \\")
        print("    --mode analyze \\")
        print("    --max_samples 300")
        print()
        print("# 分析Counting-Only消融模型")
        print("python Universal_Model_Analysis.py \\")
        print("    --checkpoint ./best_counting_only_model.pth \\")
        print("    --val_csv ./val_subset.csv \\")
        print("    --data_root ./data \\")
        print("    --mode analyze")
        print()
        print("# 分析特定组件")
        print("python Universal_Model_Analysis.py \\")
        print("    --checkpoint ./model.pth \\")
        print("    --val_csv ./val.csv \\")
        print("    --data_root ./data \\")
        print("    --mode analyze \\")
        print("    --components fusion lstm counting_decoder")
        print()
        print("可选参数:")
        print("  --save_dir DIR          保存目录")
        print("  --max_samples N         最大样本数 (默认500)")
        print("  --components LIST       指定组件 (默认自动选择)")
        print("  --batch_size N          批次大小 (默认8)")
        print()
        print("💡 推荐工作流:")
        print("1. 先运行 --mode inspect 查看模型结构")
        print("2. 再运行 --mode analyze 进行完整分析")
        print("3. 每个组件会生成4张图: PCA-2D, PCA-3D, t-SNE-2D, t-SNE-3D")
        sys.exit(0)
    
    main()


# =============================================================================
# 便捷函数，供其他脚本调用
# =============================================================================

def quick_analyze(checkpoint_path, val_csv, data_root, save_dir=None, max_samples=200):
    """快速分析接口"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        save_dir = f'./quick_analysis_{model_name}_{timestamp}'
    
    return analyze_model(checkpoint_path, val_csv, data_root, save_dir, max_samples)


def compare_models(checkpoint_paths, val_csv, data_root, base_save_dir='./model_comparison'):
    """对比多个模型的组件特征"""
    print("🔄 开始多模型对比分析...")
    
    results = {}
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"分析模型 {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        print(f"{'='*60}")
        
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        save_dir = os.path.join(base_save_dir, f'model_{i+1}_{model_name}')
        
        try:
            result = analyze_model(checkpoint_path, val_csv, data_root, save_dir, max_samples=300)
            results[model_name] = result
            print(f"✅ 模型 {model_name} 分析完成")
        except Exception as e:
            print(f"❌ 模型 {model_name} 分析失败: {e}")
            results[model_name] = None
    
    # 生成对比总结
    print(f"\n📊 多模型对比总结:")
    print("-" * 60)
    for model_name, result in results.items():
        if result:
            print(f"{model_name}:")
            print(f"  模型类型: {result['model_type']}")
            print(f"  分析组件: {', '.join(result['components_analyzed'])}")
            print(f"  样本数: {len(result['labels'])}")
        else:
            print(f"{model_name}: 分析失败")
    
    print(f"\n📁 所有结果保存在: {base_save_dir}")
    
    return results


def analyze_specific_components(checkpoint_path, val_csv, data_root, 
                               components, save_dir=None, max_samples=500):
    """分析特定组件"""
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        components_str = '_'.join(components)
        save_dir = f'./component_analysis_{model_name}_{components_str}_{timestamp}'
    
    return analyze_model(checkpoint_path, val_csv, data_root, save_dir, max_samples, components)


# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

1. 命令行使用:
   # 检查模型
   python Universal_Model_Analysis.py --checkpoint model.pth --mode inspect
   
   # 完整分析
   python Universal_Model_Analysis.py \\
       --checkpoint model.pth \\
       --val_csv val.csv \\
       --data_root ./data \\
       --mode analyze \\
       --max_samples 500

2. 在Python脚本中使用:
   from Universal_Model_Analysis import quick_analyze, compare_models
   
   # 快速分析单个模型
   result = quick_analyze(
       checkpoint_path='./best_model.pth',
       val_csv='./val.csv', 
       data_root='./data',
       max_samples=300
   )
   
   # 对比多个模型
   model_paths = [
       './embodied_model.pth',
       './counting_only_model.pth',
       './visual_only_model.pth'
   ]
   
   comparison_results = compare_models(
       checkpoint_paths=model_paths,
       val_csv='./val.csv',
       data_root='./data'
   )

3. 分析特定组件:
   from Universal_Model_Analysis import analyze_specific_components
   
   result = analyze_specific_components(
       checkpoint_path='./model.pth',
       val_csv='./val.csv',
       data_root='./data',
       components=['fusion', 'lstm'],
       max_samples=400
   )

输出结构:
analysis_results/
├── fusion/
│   ├── fusion_pca_2d.png      # PCA 2D可视化
│   ├── fusion_pca_3d.png      # PCA 3D可视化
│   ├── fusion_tsne_2d.png     # t-SNE 2D可视化
│   └── fusion_tsne_3d.png     # t-SNE 3D可视化
├── lstm/
│   ├── lstm_pca_2d.png
│   ├── lstm_pca_3d.png
│   ├── lstm_tsne_2d.png
│   └── lstm_tsne_3d.png
└── counting_decoder/
    ├── counting_decoder_pca_2d.png
    ├── counting_decoder_pca_3d.png
    ├── counting_decoder_tsne_2d.png
    └── counting_decoder_tsne_3d.png
"""
print("分析完成")