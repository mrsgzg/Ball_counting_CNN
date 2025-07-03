"""
主文件 - 视觉/具身计数模型训练程序
"""

import argparse
import os
import torch
import numpy as np
import random
import json
from Train_embodiment import create_trainer  # 原来的具身模型训练器
from train_vision import create_vision_trainer  # 新的纯视觉模型训练器


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vision/Embodied Counting Model Training')
    
    # 选择模型类型
    parser.add_argument('--model_type', type=str, default='embodied',
                        choices=['embodied', 'vision'],
                        help='Type of model to train (embodied or vision)')
    
    # 基础配置
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, 
                        default='/home/embody_data/E_talk_project/Data_Set/Ball_dataset1',
                        help='Data root directory')
    parser.add_argument('--train_csv', type=str,
                        default='/home/embody_data/E_talk_project/Embodyment_research/Pointing_embody/Dataset_CSV/train_zipf_10.csv',
                        help='Train CSV file path')
    parser.add_argument('--val_csv', type=str,
                        default='/home/embody_data/E_talk_project/Embodyment_research/Pointing_embody/Dataset_CSV/val.csv',
                        help='Validation CSV file path')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--total_epochs', type=int, default=200,
                        help='Total training epochs')
    
    # 具身模型特有参数
    parser.add_argument('--stage_1_epochs', type=int, default=20,
                        help='Epochs for stage 1 (motion only) - only for embodied model')
    parser.add_argument('--stage_2_epochs', type=int, default=150,
                        help='Epochs for stage 2 (joint training) - only for embodied model')
    
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
    parser.add_argument('--joint_dim', type=int, default=8,
                        help='Joint dimension (only for embodied model)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
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
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: ./checkpoints/[model_type])')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save logs (default: ./logs/[model_type])')
    parser.add_argument('--save_every', type=int, default=5,
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


def print_config(config):
    """打印配置信息"""
    print("="*50)
    print(f"训练 {config['model_type'].upper()} 模型配置")
    print("="*50)
    
    for key, value in config.items():
        if key != 'model_config':
            print(f"{key}: {value}")
    
    print("\n模型配置:")
    for key, value in config['model_config'].items():
        print(f"  {key}: {value}")
    
    print("="*50)


def save_config(config, save_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"配置保存到: {save_path}")


def build_config_from_args(args):
    """从命令行参数构建配置"""
    # 设置保存和日志目录
    if args.save_dir is None:
        args.save_dir = f'./checkpoints/{args.model_type}'
    if args.log_dir is None:
        args.log_dir = f'./logs/{args.model_type}'
    
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
        'dropout': args.dropout
    }
    
    # 具身模型额外参数
    if args.model_type == 'embodied':
        model_config['joint_dim'] = args.joint_dim
    
    # 构建完整配置
    config = {
        # 模型类型
        'model_type': args.model_type,
        
        # 数据配置
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv,
        'sequence_length': args.sequence_length,
        'normalize': args.normalize,
        
        # 模型配置
        'model_config': model_config,
        
        # 训练配置
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_betas': (0.9, 0.999),  # Adam优化器的beta参数
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
    
    # 具身模型特有配置
    if args.model_type == 'embodied':
        config['stage_1_epochs'] = args.stage_1_epochs
        config['stage_2_epochs'] = args.stage_2_epochs
    
    return config


def main():
    """主函数"""
    # 解析命令行参数并构建配置
    args = parse_arguments()
    config = build_config_from_args(args)
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 打印配置
    print_config(config)
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config['save_dir'], f'{config["model_type"]}_config.json')
    save_config(config, config_save_path)
    
    # 根据模型类型创建相应的训练器
    print(f"\n正在初始化 {config['model_type']} 模型训练器...")
    if config['model_type'] == 'embodied':
        # 创建具身模型训练器
        trainer = create_trainer(config)
    else:
        # 创建纯视觉模型训练器
        trainer = create_vision_trainer(config)
    
    # 如果指定了resume路径，加载检查点
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"找不到恢复检查点: {args.resume}")
            exit(1)
    
    # 开始训练
    print(f"\n开始训练 {config['model_type']} 模型...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存当前状态
        trainer.save_checkpoint(
            epoch=trainer.start_epoch - 1,
            val_loss=trainer.best_val_loss,
            val_accuracy=trainer.best_val_accuracy,
            checkpoint_type='interrupted'
        )
    except Exception as e:
        print(f"\n训练失败，错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("训练程序完成。")


if __name__ == '__main__':
    main()