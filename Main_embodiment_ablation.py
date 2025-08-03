"""
主文件 - 具身计数模型消融实验训练程序
"""

import argparse
import os
import torch
import numpy as np
import random
import json
from Train_embodiment_ablation import create_ablation_trainer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Embodied Counting Model Ablation Study')
    
    # 消融实验配置
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['counting_only', 'visual_only'],
                        help='Ablation model type: counting_only (embodied but no motion) or visual_only (no embodiment)')
    
    # 基础配置
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, 
                        default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                        help='Data root directory')
    parser.add_argument('--train_csv', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv',
                        help='Train CSV file path')
    parser.add_argument('--val_csv', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                        help='Validation CSV file path')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--total_epochs', type=int, default=300,
                        help='Total training epochs (reduced for ablation)')
    
    # 模型参数
    parser.add_argument('--cnn_layers', type=int, default=3,
                        help='Number of CNN layers')
    parser.add_argument('--cnn_channels', type=str, default='64,128,256',
                        help='CNN channels per layer, comma-separated')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm_hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension')
    parser.add_argument('--attention_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--joint_dim', type=int, default=7,
                        help='Joint dimension (7 for your robot joints)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # 图像处理参数
    parser.add_argument('--image_mode', type=str, default='rgb',
                        choices=['rgb', 'grayscale'],
                        help='Image processing mode (rgb or grayscale)')
    
    # 学习率调度参数
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for plateau scheduler')
    
    # 数据处理参数
    parser.add_argument('--sequence_length', type=int, default=11,
                        help='Sequence length')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize input data')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/checkpoints/ablation',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/logs/ablation',
                        help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency (in batches)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(config, model_type):
    """打印配置信息"""
    print("="*60)
    print(f"具身计数模型消融实验 - {model_type.upper()}")
    print("="*60)
    
    # 消融实验信息
    print("消融实验信息:")
    if model_type == 'counting_only':
        print("  模型类型: 纯计数具身模型")
        print("  特点: 保留具身信息，移除关节预测")
        print("  目的: 验证关节预测任务对计数性能的影响")
    elif model_type == 'visual_only':
        print("  模型类型: 纯视觉计数模型")
        print("  特点: 移除具身信息，只使用视觉")
        print("  目的: 验证具身信息对计数性能的价值")
    
    # 基础配置
    print("\n基础配置:")
    basic_keys = ['device', 'batch_size', 'learning_rate', 'total_epochs', 'image_mode']
    for key in basic_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 数据配置
    print("\n数据配置:")
    data_keys = ['data_root', 'train_csv', 'val_csv', 'sequence_length', 'normalize']
    for key in data_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 模型配置
    print("\n模型配置:")
    for key, value in config['model_config'].items():
        print(f"  {key}: {value}")
    
    # 保存配置
    print("\n保存配置:")
    save_keys = ['save_dir', 'log_dir', 'save_every']
    for key in save_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print("="*60)


def validate_paths(config):
    """验证文件路径是否存在"""
    errors = []
    
    # 检查数据根目录
    if not os.path.exists(config['data_root']):
        errors.append(f"数据根目录不存在: {config['data_root']}")
    
    # 检查CSV文件
    if not os.path.exists(config['train_csv']):
        errors.append(f"训练CSV文件不存在: {config['train_csv']}")
    
    if not os.path.exists(config['val_csv']):
        errors.append(f"验证CSV文件不存在: {config['val_csv']}")
    
    if errors:
        print("路径验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("所有路径验证通过")
    return True


def save_config(config, save_path, model_type):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建可序列化的配置副本
    serializable_config = {
        'model_type': model_type,
        'experiment_type': 'ablation_study'
    }
    
    for key, value in config.items():
        if isinstance(value, dict):
            serializable_config[key] = dict(value)
        else:
            serializable_config[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"配置保存到: {save_path}")


def build_config_from_args(args):
    """从命令行参数构建配置"""
    # 将CNN通道从字符串转换为列表
    cnn_channels = [int(c) for c in args.cnn_channels.split(',')]
    
    # 构建模型配置
    model_config = {
        'cnn_layers': args.cnn_layers,
        'cnn_channels': cnn_channels,
        'lstm_layers': args.lstm_layers,
        'lstm_hidden_size': args.lstm_hidden_size,
        'feature_dim': args.feature_dim,
        'attention_heads': args.attention_heads,
        'joint_dim': args.joint_dim,
        'dropout': args.dropout,
        # input_channels将在trainer中根据image_mode设置
    }
    
    # 构建完整配置
    config = {
        # 数据配置
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv,
        'sequence_length': args.sequence_length,
        'normalize': args.normalize,
        'image_mode': args.image_mode,
        
        # 模型配置
        'model_config': model_config,
        
        # 训练配置
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_betas': (0.9, 0.999),
        'grad_clip_norm': args.grad_clip_norm,
        'total_epochs': args.total_epochs,
        
        # 学习率调度器
        'scheduler_type': args.scheduler_type,
        'scheduler_patience': args.scheduler_patience,
        
        # 保存和日志
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'save_every': args.save_every,
        'print_freq': args.print_freq,
        
        # 设备和数据加载
        'device': args.device,
        'num_workers': args.num_workers,
        
        # 随机种子
        'seed': args.seed
    }
    
    return config


def main():
    """主函数"""
    # 解析命令行参数并构建配置
    args = parse_arguments()
    config = build_config_from_args(args)
    model_type = args.model_type
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 打印配置
    print_config(config, model_type)
    
    # 验证路径
    if not validate_paths(config):
        print("路径验证失败，程序退出")
        return
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config['save_dir'], f'{model_type}_config.json')
    save_config(config, config_save_path, model_type)
    
    # 创建消融实验训练器
    print(f"\n正在初始化消融实验训练器...")
    print(f"模型类型: {model_type}")
    print(f"图像模式: {config['image_mode'].upper()}")
    
    try:
        trainer = create_ablation_trainer(config, model_type)
    except Exception as e:
        print(f"初始化训练器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 如果指定了resume路径，加载检查点
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"找不到恢复检查点: {args.resume}")
            return
    
    # 开始训练
    print(f"\n开始训练消融实验模型...")
    
    if model_type == 'counting_only':
        print("实验目标: 验证关节预测任务对计数性能的影响")
        print("对比对象: 完整具身模型 vs 纯计数具身模型")
    elif model_type == 'visual_only':
        print("实验目标: 验证具身信息对计数性能的价值")
        print("对比对象: 完整具身模型 vs 纯视觉模型")
    
    try:
        trainer.train()
        print("\n训练成功完成！")
        
        # 打印实验结果总结
        print(f"\n=== 实验结果总结 ===")
        print(f"模型类型: {model_type}")
        print(f"最佳验证准确率: {trainer.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
        
        if model_type == 'counting_only':
            print("结论: 请与完整具身模型对比，分析关节预测任务的影响")
        elif model_type == 'visual_only':
            print("结论: 请与完整具身模型对比，分析具身信息的价值")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存当前状态
        current_epoch = trainer.start_epoch if hasattr(trainer, 'start_epoch') else 0
        trainer.save_checkpoint(
            epoch=current_epoch,
            val_loss=trainer.best_val_loss,
            val_accuracy=trainer.best_val_accuracy,
            checkpoint_type='interrupted'
        )
        print("已保存中断时的模型状态")
    except Exception as e:
        print(f"\n训练失败，错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前状态
        try:
            current_epoch = trainer.start_epoch if hasattr(trainer, 'start_epoch') else 0
            trainer.save_checkpoint(
                epoch=current_epoch,
                val_loss=trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else float('inf'),
                val_accuracy=trainer.best_val_accuracy if hasattr(trainer, 'best_val_accuracy') else 0.0,
                checkpoint_type='error'
            )
            print("已保存错误时的模型状态")
        except:
            print("无法保存错误时的模型状态")
    
    print("程序结束。")


def print_usage_examples():
    """打印使用示例"""
    print("\n使用示例:")
    print("="*60)
    
    print("1. 训练纯计数具身模型 (有具身信息，无关节预测):")
    print("python Main_embodiment_ablation.py --model_type counting_only")
    
    print("\n2. 训练纯视觉模型 (无具身信息):")
    print("python Main_embodiment_ablation.py --model_type visual_only")
    
    print("\n3. 自定义训练参数:")
    print("python Main_embodiment_ablation.py --model_type counting_only --batch_size 16 --learning_rate 5e-5")
    
    print("\n4. 灰度模式训练:")
    print("python Main_embodiment_ablation.py --model_type visual_only --image_mode grayscale")
    
    print("\n5. 从检查点恢复:")
    print("python Main_embodiment_ablation.py --model_type counting_only --resume ./checkpoints/counting_only_checkpoint.pth")
    
    print("\n6. 完整的消融实验流程:")
    print("# 第一步: 训练纯计数具身模型")
    print("python Main_embodiment_ablation.py --model_type counting_only --total_epochs 200")
    print()
    print("# 第二步: 训练纯视觉模型")
    print("python Main_embodiment_ablation.py --model_type visual_only --total_epochs 200")
    print()
    print("# 第三步: 对比三个模型的结果")
    print("# - 完整具身模型 (从 Main.py 训练)")
    print("# - 纯计数具身模型 (从上面第一步)")
    print("# - 纯视觉模型 (从上面第二步)")
    
    print("\n7. 快速测试 (较少epoch):")
    print("python Main_embodiment_ablation.py --model_type counting_only --total_epochs 50 --save_every 5")
    
    print("="*60)
    print("\n消融实验设计说明:")
    print("- counting_only: 验证多任务学习的影响 (关节预测是否有助于计数)")
    print("- visual_only: 验证具身信息的价值 (关节位置信息是否重要)")
    print("- 通过对比三个模型可以回答:")
    print("  1. 具身信息对计数任务的贡献")
    print("  2. 多任务学习(计数+关节预测)的效果")
    print("  3. 哪个组件对性能影响最大")


if __name__ == '__main__':
    # 检查是否请求帮助
    import sys
    if len(sys.argv) == 1:
        print("具身计数模型消融实验程序")
        print("用于验证具身信息和多任务学习的价值")
        print("使用 --help 查看完整参数列表")
        print_usage_examples()
    else:
        main()