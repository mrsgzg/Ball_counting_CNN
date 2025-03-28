import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入您的模型和数据集
from src.model import SimplerBallCounterCNN
from src.dataset import get_data_loaders

class ConvLayerActivationRecorder:
    def __init__(self, model, device, save_dir='conv_activations'):
        """
        初始化激活记录器，只记录卷积层的激活值
        
        Args:
            model: 加载的模型
            device: 运行设备 ('cuda' 或 'cpu')
            save_dir: 保存激活值的目录
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义要捕获的卷积层
        self.conv_layers = {
            'conv1': model.conv1,  # 第一个卷积层
            'conv2': model.conv2,  # 第二个卷积层
            'conv3': model.conv3,  # 第三个卷积层
        }
        
        # 初始化存储激活值的字典
        self.activations = {}
        for layer_name in self.conv_layers:
            self.activations[layer_name] = {}
            for label in range(10):
                self.activations[layer_name][label] = []
        
        # 存储当前处理的标签
        self.current_labels = None
        
        # 注册钩子函数
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向传播钩子函数来捕获卷积层的激活值"""
        
        for layer_name, layer in self.conv_layers.items():
            # 为每一层创建钩子函数
            def get_hook(name):
                def hook(module, input, output):
                    # 如果当前没有处理样本，则返回
                    if self.current_labels is None:
                        return
                    
                    # 对于每个样本
                    for i, label in enumerate(self.current_labels.cpu().numpy()):
                        # 获取ReLU激活后的值(特征图)
                        # 注意：这里可能需要根据您的模型调整，确保捕获的是ReLU激活后的值
                        feature_maps = output[i]  # 形状为 [num_filters, height, width]
                        
                        # 存储每个特征图的原始激活值（不进行平均）
                        # 将张量转换为NumPy数组
                        act = feature_maps.cpu().numpy()
                        
                        # 将激活值添加到相应标签的列表中
                        self.activations[name][label].append(act)
                
                return hook
            
            # 注册钩子
            self.hooks.append(layer.register_forward_hook(get_hook(layer_name)))
    
    def record_activations(self, dataloader, max_batches=None):
        """
        记录一个数据加载器中所有样本的卷积层激活值
        
        Args:
            dataloader: 包含图像和标签的数据加载器
            max_batches: 最大处理的批次数(用于调试)
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Recording activations")):
                # 如果指定了最大批次数，则在达到后停止
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                self.current_labels = labels
                
                # 前向传播
                _ = self.model(images)
                
                # 清除当前标签
                self.current_labels = None
    
    def save_to_csv(self):
        """将记录的卷积层激活值保存为CSV文件，每个特征图（神经元）一个文件"""
        print("正在保存卷积层激活值到CSV文件...")
        
        # 遍历每一个卷积层
        for layer_name in tqdm(self.conv_layers.keys(), desc="处理卷积层"):
            # 为该层创建目录
            layer_dir = os.path.join(self.save_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            # 检查是否有数据
            if not any(self.activations[layer_name].values()):
                print(f"警告：层 {layer_name} 没有记录到激活值")
                continue
            
            # 获取该层特征图数量
            sample_data = next(iter(v for v in self.activations[layer_name].values() if v))
            if not sample_data:
                continue
                
            num_filters = sample_data[0].shape[0]  # 特征图数量
            
            # 为每个特征图（神经元）创建CSV，记录其全局平均激活值
            for filter_idx in range(num_filters):
                data = []
                for label in range(10):
                    activations_for_label = self.activations[layer_name][label]
                    if not activations_for_label:
                        continue
                        
                    for activation in activations_for_label:
                        # 计算该特征图的全局平均激活值
                        filter_activation = activation[filter_idx]
                        mean_activation = np.mean(filter_activation)
                        
                        data.append({
                            'label': label,
                            'activation': mean_activation
                        })
                
                if data:  # 确保有数据
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(layer_dir, f'filter_{filter_idx}.csv'), index=False)
        
        print(f"卷积层激活值已保存到 {self.save_dir} 目录")
    
    def save_layer_summary(self):
        """保存每一个卷积层的统计摘要"""
        summary_data = []
        
        for layer_name in self.conv_layers.keys():
            # 检查是否有数据
            if not any(self.activations[layer_name].values()):
                continue
                
            # 获取特征图数量
            sample_data = next(iter(v for v in self.activations[layer_name].values() if v))
            if not sample_data:
                continue
                
            num_filters = sample_data[0].shape[0]
            filter_height = sample_data[0].shape[1]
            filter_width = sample_data[0].shape[2]
            
            # 获取样本数量
            total_samples = sum(len(samples) for samples in self.activations[layer_name].values())
            
            summary_data.append({
                'layer_name': layer_name,
                'num_filters': num_filters,
                'filter_height': filter_height,
                'filter_width': filter_width,
                'total_samples': total_samples
            })
        
        # 保存摘要
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.save_dir, 'conv_layer_summary.csv'), index=False)
        print(f"卷积层摘要已保存到 {os.path.join(self.save_dir, 'conv_layer_summary.csv')}")
    
    def cleanup(self):
        """移除钩子函数"""
        for hook in self.hooks:
            hook.remove()


def analyze_anova(save_dir, p_threshold=0.01):
    """
    对每个卷积层特征图（神经元）进行ANOVA分析以检测数量选择性
    
    Args:
        save_dir: 激活值CSV文件的目录
        p_threshold: 显著性阈值
    """
    try:
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print("正在进行卷积层ANOVA分析...")
        
        # 创建汇总结果目录
        summary_dir = os.path.join(save_dir, 'anova_summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 查找所有卷积层目录
        layer_dirs = [d for d in os.listdir(save_dir) 
                     if os.path.isdir(os.path.join(save_dir, d)) and d != 'anova_summary']
        
        # 存储所有层的结果
        layer_results = []
        
        # 分析每一层
        for layer_name in layer_dirs:
            layer_dir = os.path.join(save_dir, layer_name)
            
            # 分析该层的特征图（神经元）
            selective_count, total_count = analyze_layer_filters(
                layer_dir, layer_name, p_threshold)
            
            # 保存结果
            if total_count > 0:
                percentage = (selective_count / total_count) * 100
                layer_results.append({
                    'layer_name': layer_name,
                    'selective_filters': selective_count,
                    'total_filters': total_count,
                    'percentage': percentage
                })
        
        # 创建汇总表格
        summary_df = pd.DataFrame(layer_results)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('layer_name')
            summary_df.to_csv(os.path.join(summary_dir, 'anova_results.csv'), index=False)
            
            # 创建汇总图表
            plt.figure(figsize=(10, 6))
            plt.bar(summary_df['layer_name'], summary_df['percentage'])
            plt.title(f'数量选择性特征图百分比 (p < {p_threshold})')
            plt.xlabel('卷积层')
            plt.ylabel('数量选择性特征图百分比 (%)')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'selectivity_by_layer.png'))
            plt.close()
            
            print(f"ANOVA分析汇总已保存到 {summary_dir}")
        else:
            print("没有层包含足够的特征图进行分析")
            
    except ImportError:
        print("需要安装scipy和matplotlib进行ANOVA分析")


def analyze_layer_filters(layer_dir, layer_name, p_threshold=0.01):
    """
    分析一个卷积层中所有特征图的数量选择性
    
    Args:
        layer_dir: 包含该层特征图CSV文件的目录
        layer_name: 层的名称
        p_threshold: ANOVA的p值阈值
        
    Returns:
        (selective_count, total_count): 数量选择性特征图数量和总特征图数量
    """
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # 创建保存分析结果的目录
    analysis_dir = os.path.join(layer_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 获取该层中的所有特征图CSV文件
    filter_files = [f for f in os.listdir(layer_dir) if f.startswith('filter_') and f.endswith('.csv')]
    
    if not filter_files:
        print(f"警告: {layer_name}层没有找到特征图文件")
        return 0, 0
    
    # 存储数量选择性特征图
    numerosity_selective_filters = []
    
    # 对每个特征图进行分析
    for filter_file in tqdm(filter_files, desc=f"分析{layer_name}特征图"):
        filter_idx = int(filter_file.split('_')[1].split('.')[0])
        
        try:
            df = pd.read_csv(os.path.join(layer_dir, filter_file))
            
            # 执行单因素ANOVA
            groups = [df[df['label'] == label]['activation'].values for label in range(10) 
                     if not df[df['label'] == label].empty]
            
            # 确保至少有两个组用于ANOVA
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                f_val, p_val = stats.f_oneway(*groups)
                
                # 如果p值小于阈值，认为该特征图对数量敏感
                if p_val < p_threshold:
                    numerosity_selective_filters.append(filter_idx)
                    
                    # 可视化该特征图的激活模式
                    plt.figure(figsize=(10, 6))
                    
                    # 计算每个标签的平均激活值和标准误差
                    mean_activations = []
                    std_errors = []
                    labels_present = []
                    
                    for label in range(10):
                        if not df[df['label'] == label].empty:
                            activations = df[df['label'] == label]['activation'].values
                            mean_activations.append(np.mean(activations))
                            std_errors.append(np.std(activations) / np.sqrt(len(activations)))
                            labels_present.append(label)
                    
                    # 绘制激活模式
                    plt.errorbar(labels_present, mean_activations, yerr=std_errors, fmt='o-', capsize=5)
                    plt.title(f'{layer_name} - Filter {filter_idx} Activation Pattern (p={p_val:.4f})')
                    plt.xlabel('Label (Number of Balls)')
                    plt.ylabel('Mean Activation')
                    plt.xticks(range(10))
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # 保存图像
                    plt.savefig(os.path.join(analysis_dir, f'filter_{filter_idx}_pattern.png'))
                    plt.close()
        except Exception as e:
            print(f"分析特征图 {filter_idx} 时出错: {e}")
    
    # 保存数量选择性特征图的信息
    with open(os.path.join(analysis_dir, 'numerosity_selective_filters.txt'), 'w') as f:
        f.write(f"数量选择性特征图数量: {len(numerosity_selective_filters)}/{len(filter_files)}\n")
        f.write(f"占比: {len(numerosity_selective_filters)/len(filter_files)*100:.2f}%\n\n")
        f.write("数量选择性特征图索引:\n")
        f.write(', '.join(map(str, numerosity_selective_filters)))
    
    print(f"{layer_name}层中发现{len(numerosity_selective_filters)}个数量选择性特征图，占比{len(numerosity_selective_filters)/len(filter_files)*100:.2f}%")
    
    return len(numerosity_selective_filters), len(filter_files)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='记录卷积层的激活值并分析数量选择性')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default='conv_activations', help='保存激活值的目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--samples', type=int, default=100, help='每个类别的样本数')
    parser.add_argument('--max_batches', type=int, default=None, help='处理的最大批次数(用于调试)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备 (cuda 或 cpu)')
    parser.add_argument('--analyze', action='store_true', help='执行ANOVA分析')
    parser.add_argument('--p_threshold', type=float, default=0.01, help='ANOVA显著性阈值')
    args = parser.parse_args()
    
    # 加载数据集
    print(f"加载数据集 {args.data_dir}...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        num_samples_per_class=args.samples,
        batch_size=args.batch_size,
        seed=42
    )
    
    # 创建模型
    print("创建模型...")
    model = SimplerBallCounterCNN(num_classes=10)
    model = model.to(args.device)
    
    # 加载检查点（如果提供）
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"加载检查点 {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    
    # 初始化激活记录器
    recorder = ConvLayerActivationRecorder(model, args.device, save_dir=args.save_dir)
    
    # 记录激活值
    print("记录卷积层激活值...")
    recorder.record_activations(test_loader, max_batches=args.max_batches)
    
    # 保存激活值到CSV
    recorder.save_to_csv()
    
    # 保存层摘要
    recorder.save_layer_summary()
    
    # 清理钩子函数
    recorder.cleanup()
    
    # 执行ANOVA分析（如果请求）
    if args.analyze:
        analyze_anova(args.save_dir, args.p_threshold)

if __name__ == "__main__":
    main()