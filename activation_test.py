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

class ActivationRecorder:
    def __init__(self, model, device, save_dir='activations'):
        """
        初始化激活记录器
        
        Args:
            model: 加载的模型
            device: 运行设备 ('cuda' 或 'cpu')
            save_dir: 保存激活值的目录
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化存储激活值的字典
        self.activations = {
            'last_conv': {},  # 最后一个卷积层的激活
            'fc': {}          # 全连接层的激活
        }
        
        # 为每个标签类别初始化激活值存储
        for label in range(10):
            self.activations['last_conv'][label] = []
            self.activations['fc'][label] = []
        
        # 注册钩子函数来捕获激活值
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向传播钩子函数来捕获激活值"""
        
        # 获取最后一个卷积层（conv3）的激活值
        def conv3_hook(module, input, output):
            self.last_conv_activation = output
        
        # 获取全连接层之前的激活值（全局平均池化后）
        def gap_hook(module, input, output):
            self.fc_activation = output
        
        # 注册钩子
        self.hooks.append(self.model.conv3.register_forward_hook(conv3_hook))
        self.hooks.append(self.model.gap.register_forward_hook(gap_hook))
    
    def record_activations(self, dataloader):
        """
        记录一个数据加载器中所有样本的激活值
        
        Args:
            dataloader: 包含图像和标签的数据加载器
        """
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Recording activations"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                _ = self.model(images)
                
                # 处理批次中的每个样本
                for i, label in enumerate(labels.cpu().numpy()):
                    # 获取单个样本的卷积激活值（取全局平均池化）
                    conv_act = self.last_conv_activation[i].mean(dim=(1, 2)).cpu().numpy()
                    
                    # 获取全连接层之前的激活值
                    fc_act = self.fc_activation[i].squeeze().cpu().numpy()
                    
                    # 将激活值添加到相应标签的列表中
                    self.activations['last_conv'][label].append(conv_act)
                    self.activations['fc'][label].append(fc_act)
    
    def save_to_csv(self):
        """将记录的激活值保存为CSV文件，每个神经元一个文件"""
        print("正在保存激活值到CSV文件...")
        
        # 处理最后一个卷积层的激活值
        conv_dir = os.path.join(self.save_dir, 'last_conv')
        os.makedirs(conv_dir, exist_ok=True)
        
        # 获取卷积层神经元数量
        num_conv_neurons = len(self.activations['last_conv'][0][0])
        
        # 为每个卷积层神经元创建CSV
        for neuron_idx in range(num_conv_neurons):
            data = []
            for label in range(10):
                for activation in self.activations['last_conv'][label]:
                    data.append({
                        'label': label,
                        'activation': activation[neuron_idx]
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(conv_dir, f'neuron_{neuron_idx}.csv'), index=False)
        
        # 处理全连接层之前的激活值
        fc_dir = os.path.join(self.save_dir, 'fc')
        os.makedirs(fc_dir, exist_ok=True)
        
        # 获取全连接层之前的神经元数量
        num_fc_neurons = len(self.activations['fc'][0][0])
        
        # 为每个全连接层之前的神经元创建CSV
        for neuron_idx in range(num_fc_neurons):
            data = []
            for label in range(10):
                for activation in self.activations['fc'][label]:
                    data.append({
                        'label': label,
                        'activation': activation[neuron_idx]
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(fc_dir, f'neuron_{neuron_idx}.csv'), index=False)
        
        print(f"激活值已保存到 {self.save_dir} 目录")
    
    def cleanup(self):
        """移除钩子函数"""
        for hook in self.hooks:
            hook.remove()


def analyze_anova(save_dir='activations'):
    """
    对每个神经元进行ANOVA分析以检测数量选择性
    
    Args:
        save_dir: 激活值CSV文件的目录
    """
    try:
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print("正在进行ANOVA分析...")
        
        # 分析最后一个卷积层
        conv_dir = os.path.join(save_dir, 'last_conv')
        analyze_layer_neurons(conv_dir, 'last_conv')
        
        # 分析全连接层之前的神经元
        fc_dir = os.path.join(save_dir, 'fc')
        analyze_layer_neurons(fc_dir, 'fc')
        
    except ImportError:
        print("需要安装scipy和matplotlib进行ANOVA分析")


def analyze_layer_neurons(layer_dir, layer_name):
    """
    分析一个层中所有神经元的数量选择性
    
    Args:
        layer_dir: 包含该层神经元CSV文件的目录
        layer_name: 层的名称（用于输出）
    """
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # 创建保存分析结果的目录
    analysis_dir = os.path.join(layer_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 获取该层中的所有神经元CSV文件
    neuron_files = [f for f in os.listdir(layer_dir) if f.startswith('neuron_') and f.endswith('.csv')]
    
    # 存储数量选择性神经元
    numerosity_selective_neurons = []
    
    # 对每个神经元进行分析
    for neuron_file in tqdm(neuron_files, desc=f"分析{layer_name}神经元"):
        neuron_idx = int(neuron_file.split('_')[1].split('.')[0])
        df = pd.read_csv(os.path.join(layer_dir, neuron_file))
        
        # 执行单因素ANOVA
        groups = [df[df['label'] == label]['activation'].values for label in range(10)]
        f_val, p_val = stats.f_oneway(*groups)
        
        # 如果p值小于0.01，认为该神经元对数量敏感
        if p_val < 0.01:
            numerosity_selective_neurons.append(neuron_idx)
            
            # 可视化该神经元的激活模式
            plt.figure(figsize=(10, 6))
            
            # 计算每个标签的平均激活值和标准误差
            mean_activations = []
            std_errors = []
            
            for label in range(10):
                activations = df[df['label'] == label]['activation'].values
                mean_activations.append(np.mean(activations))
                std_errors.append(np.std(activations) / np.sqrt(len(activations)))
            
            # 绘制激活模式
            plt.errorbar(range(10), mean_activations, yerr=std_errors, fmt='o-', capsize=5)
            plt.title(f'Neuron {neuron_idx} Activation Pattern (p={p_val:.4f})')
            plt.xlabel('Label (Number of Balls)')
            plt.ylabel('Mean Activation')
            plt.xticks(range(10))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图像
            plt.savefig(os.path.join(analysis_dir, f'neuron_{neuron_idx}_pattern.png'))
            plt.close()
    
    # 保存数量选择性神经元的信息
    with open(os.path.join(analysis_dir, 'numerosity_selective_neurons.txt'), 'w') as f:
        f.write(f"数量选择性神经元数量: {len(numerosity_selective_neurons)}/{len(neuron_files)}\n")
        f.write(f"占比: {len(numerosity_selective_neurons)/len(neuron_files)*100:.2f}%\n\n")
        f.write("数量选择性神经元索引:\n")
        f.write(', '.join(map(str, numerosity_selective_neurons)))
    
    print(f"{layer_name}层中发现{len(numerosity_selective_neurons)}个数量选择性神经元，占比{len(numerosity_selective_neurons)/len(neuron_files)*100:.2f}%")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='记录神经元激活并分析数量选择性')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default='activations', help='保存激活值的目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--samples', type=int, default=100, help='每个类别的样本数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备 (cuda 或 cpu)')
    parser.add_argument('--analyze', action='store_true', help='执行ANOVA分析')
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
    recorder = ActivationRecorder(model, args.device, save_dir=args.save_dir)
    
    # 记录激活值
    print("记录神经元激活值...")
    recorder.record_activations(test_loader)
    
    # 保存激活值到CSV
    recorder.save_to_csv()
    
    # 清理钩子函数
    recorder.cleanup()
    
    # 执行ANOVA分析（如果请求）
    if args.analyze:
        analyze_anova(args.save_dir)

if __name__ == "__main__":
    main()