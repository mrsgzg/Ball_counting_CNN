"""
主文件 - 单图像分类模型训练程序
"""

import argparse
import os
import torch
import numpy as np
import random
import json
from Train_single_image import create_single_image_trainer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Single Image Classification Model Training')
    
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
    parser.add_argument('--total_epochs', type=int, default=500,
                        help='Total training epochs')
    
    # 模型参数
    parser.add_argument('--cnn_layers', type=int, default=3,
                        help='Number of CNN layers')
    parser.add_argument('--cnn_channels', type=str, default='64,128,256',
                        help='CNN channels per layer, comma-separated')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes (0-10 balls)')
    
    # 图像处理参数
    parser.add_argument('--image_mode', type=str, default='rgb',
                        choices=['rgb', 'grayscale'],
                        help='Image processing mode (rgb or grayscale)')
    parser.add_argument('--frame_selection', type=str, default='all',
                        choices=['all', 'final', 'keyframes'],
                        help='Frame selection strategy')
    
    # 注意力机制
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use spatial attention mechanism')
    parser.add_argument('--no_attention', action='store_true', default=False,
                        help='Disable attention mechanism')
    
    # 正则化参数
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor')
    
    # 学习率调度参数
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for plateau scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for step scheduler')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/checkpoints/single_image',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/logs/single_image',
                        help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_freq', type=int, default=50,
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
    print("="*60)
    print("单图像分类模型训练配置")
    print("="*60)
    
    # 基础配置
    print("基础配置:")
    basic_keys = ['device', 'batch_size', 'learning_rate', 'total_epochs', 'image_mode', 'frame_selection']
    for key in basic_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 数据配置
    print("\n数据配置:")
    data_keys = ['data_root', 'train_csv', 'val_csv']
    for key in data_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 模型配置
    print("\n模型配置:")
    for key, value in config['model_config'].items():
        print(f"  {key}: {value}")
    
    # 训练配置
    print("\n训练配置:")
    train_keys = ['scheduler_type', 'label_smoothing', 'grad_clip_norm']
    for key in train_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
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


def save_config(config, save_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建可序列化的配置副本
    serializable_config = {}
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
    # 处理注意力机制设置
    use_attention = args.use_attention and not args.no_attention
    
    # 将CNN通道从字符串转换为列表
    cnn_channels = [int(c) for c in args.cnn_channels.split(',')]
    
    # 构建模型配置
    model_config = {
        'cnn_layers': args.cnn_layers,
        'cnn_channels': cnn_channels,
        'feature_dim': args.feature_dim,
        'attention_heads': args.attention_heads,
        'dropout': args.dropout,
    }
    
    # 构建完整配置
    config = {
        # 数据配置
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv,
        'image_mode': args.image_mode,
        'frame_selection': args.frame_selection,
        
        # 模型配置
        'model_config': model_config,
        'num_classes': args.num_classes,
        'use_attention': use_attention,
        
        # 训练配置
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_betas': (0.9, 0.999),
        'grad_clip_norm': args.grad_clip_norm,
        'total_epochs': args.total_epochs,
        'label_smoothing': args.label_smoothing,
        
        # 学习率调度器
        'scheduler_type': args.scheduler_type,
        'scheduler_patience': args.scheduler_patience,
        'step_size': args.step_size,
        
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
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 打印配置
    print_config(config)
    
    # 验证路径
    if not validate_paths(config):
        print("路径验证失败，程序退出")
        return
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config['save_dir'], 'single_image_config.json')
    save_config(config, config_save_path)
    
    # 创建训练器
    print(f"\n正在初始化单图像分类模型训练器...")
    print(f"图像模式: {config['image_mode'].upper()}")
    print(f"帧选择策略: {config['frame_selection']}")
    print(f"使用注意力机制: {config['use_attention']}")
    
    try:
        trainer = create_single_image_trainer(config)
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
    print(f"\n开始训练单图像分类模型...")
    print(f"对比基线: 纯视觉CNN分类器")
    print(f"目标: 评估embodiment信息的价值")
    
    try:
        trainer.train()
        print("\n训练成功完成！")
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
    
    print("1. 基础训练 (RGB模式，使用所有帧):")
    print("python Main_single_image.py --image_mode rgb --frame_selection all")
    
    print("\n2. 灰度模式训练:")
    print("python Main_single_image.py --image_mode grayscale --batch_size 128")
    
    print("\n3. 只使用最终帧训练:")
    print("python Main_single_image.py --frame_selection final --learning_rate 5e-4")
    
    print("\n4. 不使用注意力机制:")
    print("python Main_single_image.py --no_attention")
    
    print("\n5. 从检查点恢复:")
    print("python Main_single_image.py --resume ./checkpoints/single_image/best_single_image_model.pth")
    
    print("\n6. 自定义训练参数:")
    print("python Main_single_image.py --learning_rate 2e-3 --total_epochs 150 --batch_size 32")
    
    print("\n7. 使用关键帧策略:")
    print("python Main_single_image.py --frame_selection keyframes --attention_heads 8")
    
    print("\n8. 带标签平滑的训练:")
    print("python Main_single_image.py --label_smoothing 0.1 --scheduler_type plateau")
    
    print("="*60)


if __name__ == '__main__':
    # 检查是否请求帮助
    import sys
    if len(sys.argv) == 1:
        print("单图像分类模型训练程序")
        print("用于与具身计数模型进行对比实验")
        print("使用 --help 查看完整参数列表")
        print_usage_examples()
    else:
        main()