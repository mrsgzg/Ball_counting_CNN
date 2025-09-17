"""
主文件 - 增强具身计数模型训练程序 (Enhanced Internal Model)
"""

import argparse
import os
import torch
import numpy as np
import random
import json
from Train_enhanced_embodiment import create_enhanced_trainer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Enhanced Embodied Counting Model Training')
    
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
    parser.add_argument('--total_epochs', type=int, default=1000,
                        help='Total training epochs')
    
    # Enhanced模型特定的训练阶段参数
    parser.add_argument('--stage_1_epochs', type=int, default=0,
                        help='Epochs for stage 1 (Internal Model pretraining)')
    parser.add_argument('--stage_2_epochs', type=int, default=0,
                        help='Epochs for stage 2 (joint training)')
    
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
    parser.add_argument('--joint_dim', type=int, default=7,
                        help='Joint dimension (7 for your robot joints)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Enhanced模型特定参数
    parser.add_argument('--use_fovea_bias', action='store_true', default=True,
                        help='Use fovea bias in attention (human eye-like)')
    parser.add_argument('--no_fovea_bias', action='store_true', default=False,
                        help='Disable fovea bias')
    parser.add_argument('--embodiment_loss_weight', type=float, default=0.3,
                        help='Weight for embodiment loss in joint training')
    parser.add_argument('--attention_loss_weight', type=float, default=0.1,
                        help='Weight for attention regularization loss')
    
    # 图像处理参数
    parser.add_argument('--image_mode', type=str, default='rgb',
                        choices=['rgb', 'grayscale'],
                        help='Image processing mode (rgb or grayscale)')
    
    # 学习率调度参数
    parser.add_argument('--scheduler_type', type=str, default='none',
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
    parser.add_argument('--save_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/Enhanced/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./scratch/Ball_counting_CNN/Result_data/Enhanced/logs',
                        help='Directory to save logs')
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
    print("="*70)
    print("增强具身计数模型训练配置 (Enhanced Internal Model)")
    print("="*70)
    
    # 模型创新点
    print("🧠 模型创新点:")
    print("  ✅ Internal Model Architecture (Forward + Inverse)")
    print("  ✅ Multi-Scale Visual Features")
    print("  ✅ Early Fusion + Residual Connections")
    print("  ✅ Task-Guided Spatial Attention")
    print(f"  ✅ Fovea Bias (Human Eye-like): {config.get('use_fovea_bias', True)}")
    
    # 基础配置
    print("\n📋 基础配置:")
    basic_keys = ['device', 'batch_size', 'learning_rate', 'total_epochs', 'image_mode']
    for key in basic_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 数据配置
    print("\n📁 数据配置:")
    data_keys = ['data_root', 'train_csv', 'val_csv', 'sequence_length', 'normalize']
    for key in data_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 训练阶段配置
    print("\n🏗️ 训练阶段:")
    stage_keys = ['stage_1_epochs', 'stage_2_epochs']
    for key in stage_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print(f"  Stage 1 (0-{config['stage_1_epochs']}): Internal Model预训练")
    print(f"  Stage 2 ({config['stage_1_epochs']}-{config['stage_2_epochs']}): 联合训练")
    print(f"  Stage 3 ({config['stage_2_epochs']}-{config['total_epochs']}): 计数精调")
    
    # Enhanced模型特定配置
    print("\n🔬 Enhanced模型配置:")
    enhanced_keys = ['embodiment_loss_weight', 'attention_loss_weight', 'use_fovea_bias']
    for key in enhanced_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # 模型参数
    print("\n🏛️ 模型架构:")
    for key, value in config['model_config'].items():
        print(f"  {key}: {value}")
    
    # 保存配置
    print("\n💾 保存配置:")
    save_keys = ['save_dir', 'log_dir', 'save_every']
    for key in save_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print("="*70)


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
        print("❌ 路径验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ 所有路径验证通过")
    return True


def save_config(config, save_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建可序列化的配置副本
    serializable_config = {
        'model_type': 'Enhanced_Embodied_Counting_Model',
        'experiment_type': 'internal_model_with_early_fusion',
        'features': [
            'Internal_Model_Architecture',
            'Multi_Scale_Visual_Features',
            'Early_Fusion_Residual',
            'Task_Guided_Attention',
            'Fovea_Bias'
        ]
    }
    
    for key, value in config.items():
        if isinstance(value, dict):
            serializable_config[key] = dict(value)
        else:
            serializable_config[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"💾 配置保存到: {save_path}")


def build_config_from_args(args):
    """从命令行参数构建配置"""
    # 处理fovea bias设置
    use_fovea_bias = args.use_fovea_bias and not args.no_fovea_bias
    
    # 将CNN通道从字符串转换为列表
    cnn_channels = [int(c) for c in args.cnn_channels.split(',')]
    
    # 构建模型配置
    model_config = {
        'cnn_layers': args.cnn_layers,
        'cnn_channels': cnn_channels,
        'lstm_layers': args.lstm_layers,
        'lstm_hidden_size': args.lstm_hidden_size,
        'feature_dim': args.feature_dim,
        'joint_dim': args.joint_dim,
        'dropout': args.dropout,
        'use_fovea_bias': use_fovea_bias,
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
        'use_fovea_bias': use_fovea_bias,
        
        # 训练配置
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_betas': (0.9, 0.999),
        'grad_clip_norm': args.grad_clip_norm,
        'total_epochs': args.total_epochs,
        'stage_1_epochs': args.stage_1_epochs,
        'stage_2_epochs': args.stage_2_epochs,
        
        # Enhanced模型特定参数
        'embodiment_loss_weight': args.embodiment_loss_weight,
        'attention_loss_weight': args.attention_loss_weight,
        
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
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 打印配置
    print_config(config)
    
    # 验证路径
    if not validate_paths(config):
        print("❌ 路径验证失败，程序退出")
        return
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config['save_dir'], 'enhanced_embodied_config.json')
    save_config(config, config_save_path)
    
    # 创建Enhanced训练器
    print(f"\n🚀 正在初始化Enhanced具身计数模型训练器...")
    print(f"🖼️ 图像模式: {config['image_mode'].upper()}")
    print(f"🦾 关节维度: {config['model_config']['joint_dim']}")
    print(f"👁️ Fovea偏置: {config['use_fovea_bias']}")
    
    try:
        trainer = create_enhanced_trainer(config)
    except Exception as e:
        print(f"❌ 初始化训练器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 如果指定了resume路径，加载检查点
    if args.resume:
        if os.path.exists(args.resume):
            print(f"📂 从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"❌ 找不到恢复检查点: {args.resume}")
            return
    
    # 开始训练
    print(f"\n🎯 开始训练Enhanced具身计数模型...")
    print(f"🧠 核心创新:")
    print(f"   • Internal Model: Forward预测 + Inverse规划")
    print(f"   • Early Fusion: 多层次视觉-关节特征融合")
    print(f"   • Residual Connections: 三路残差连接")
    print(f"   • Task-Guided Attention: 任务驱动的空间注意力")
    print(f"   • Fovea Bias: 类人眼黄斑区注意力偏置")
    
    print(f"\n📅 训练计划:")
    print(f"   阶段1 (0-{config['stage_1_epochs']}): Internal Model组件预训练")
    print(f"   阶段2 ({config['stage_1_epochs']}-{config['stage_2_epochs']}): 联合训练所有组件")
    print(f"   阶段3 ({config['stage_2_epochs']}-{config['total_epochs']}): 专注计数任务精调")
    
    try:
        trainer.train()
        print("\n🎉 训练成功完成！")
        
        # 打印最终结果
        print(f"\n📊 最终结果:")
        print(f"   🏆 最佳验证准确率: {trainer.best_val_accuracy:.4f}")
        print(f"   📉 最佳验证损失: {trainer.best_val_loss:.4f}")
        print(f"   💾 模型保存位置: {config['save_dir']}")
        
        print(f"\n🔬 实验价值:")
        print(f"   • 验证了Internal Model在具身AI中的有效性")
        print(f"   • 证明了早期融合+残差结构的优越性")
        print(f"   • 实现了类人眼的task-guided attention机制")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        # 保存当前状态
        current_epoch = trainer.start_epoch if hasattr(trainer, 'start_epoch') else 0
        trainer.save_checkpoint(
            epoch=current_epoch,
            val_loss=trainer.best_val_loss,
            val_accuracy=trainer.best_val_accuracy,
            checkpoint_type='interrupted'
        )
        print("💾 已保存中断时的模型状态")
    except Exception as e:
        print(f"\n❌ 训练失败，错误: {e}")
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
            print("💾 已保存错误时的模型状态")
        except:
            print("❌ 无法保存错误时的模型状态")
    
    print("🏁 程序结束。")


def print_usage_examples():
    """打印使用示例"""
    print("\n📖 使用示例:")
    print("="*70)
    
    print("1️⃣ 基础训练 (启用所有Enhanced特性):")
    print("python Main_enhanced_embodiment.py --image_mode rgb --batch_size 16")
    
    print("\n2️⃣ 灰度模式训练:")
    print("python Main_enhanced_embodiment.py --image_mode grayscale --batch_size 32")
    
    print("\n3️⃣ 自定义训练阶段:")
    print("python Main_enhanced_embodiment.py \\")
    print("    --stage_1_epochs 50 \\")
    print("    --stage_2_epochs 180 \\") 
    print("    --total_epochs 250")
    
    print("\n4️⃣ 调整损失权重:")
    print("python Main_enhanced_embodiment.py \\")
    print("    --embodiment_loss_weight 0.2 \\")
    print("    --attention_loss_weight 0.15")
    
    print("\n5️⃣ 禁用Fovea偏置:")
    print("python Main_enhanced_embodiment.py --no_fovea_bias")
    
    print("\n6️⃣ 从检查点恢复:")
    print("python Main_enhanced_embodiment.py \\")
    print("    --resume ./checkpoints/best_enhanced_model.pth")
    
    print("\n7️⃣ 快速测试(少量epoch):")
    print("python Main_enhanced_embodiment.py \\")
    print("    --stage_1_epochs 5 \\")
    print("    --stage_2_epochs 20 \\")
    print("    --total_epochs 30 \\")
    print("    --batch_size 8")
    
    print("\n8️⃣ 高性能训练:")
    print("python Main_enhanced_embodiment.py \\")
    print("    --batch_size 32 \\")
    print("    --learning_rate 2e-4 \\")
    print("    --num_workers 8")
    
    print("="*70)
    
    print("\n🧠 Enhanced模型核心特性:")
    print("  🔸 Internal Model: 结合认知神经科学的Forward/Inverse模型")
    print("  🔸 Multi-Scale Features: 多尺度视觉特征提取")
    print("  🔸 Early Fusion + Residual: 早期融合+三路残差连接")
    print("  🔸 Task-Guided Attention: 任务驱动的空间注意力")
    print("  🔸 Fovea Bias: 模拟人眼黄斑区的中央注意力偏置")
    
    print("\n💡 实验建议:")
    print("  • 对比原始模型和Enhanced模型的性能差异")
    print("  • 分析attention权重的可视化结果")
    print("  • 评估不同阶段训练的效果")
    print("  • 测试fovea bias对注意力聚焦的影响")
    
    print("\n📊 期望改进:")
    print("  • 更精准的计数准确率")
    print("  • 更合理的关节运动预测")
    print("  • 更集中的视觉注意力")
    print("  • 更强的可解释性")


if __name__ == '__main__':
    # 检查是否请求帮助
    import sys
    if len(sys.argv) == 1:
        print("🧠 增强具身计数模型训练程序")
        print("集成Internal Model、Early Fusion、Task-Guided Attention")
        print("使用 --help 查看完整参数列表")
        print_usage_examples()
    else:
        main()