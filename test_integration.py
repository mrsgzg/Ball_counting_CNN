"""
集成测试脚本 - 测试修改后的模型、数据加载器和训练器是否能正常工作
"""

import torch
import os
import sys

# 添加当前目录到Python路径
sys.path.append('.')

from DataLoader_embodiment import get_ball_counting_data_loaders, BallCountingDataset
from Model_embodiment import EmbodiedCountingModel
from Train_embodiment import create_trainer


def test_data_loader():
    """测试数据加载器"""
    print("="*50)
    print("测试数据加载器")
    print("="*50)
    
    # 测试路径（请根据你的实际路径修改）
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
    
    # 检查文件是否存在
    if not all(os.path.exists(path) for path in [data_root, train_csv, val_csv]):
        print("❌ 数据文件不存在，跳过数据加载器测试")
        return False
    
    try:
        # 测试RGB模式
        print("测试RGB模式...")
        train_loader_rgb, val_loader_rgb, normalizer_rgb = get_ball_counting_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=2,  # 小批次便于测试
            sequence_length=11,
            normalize=True,
            image_mode="rgb",
            num_workers=0  # 避免多进程问题
        )
        
        # 获取一个batch测试
        for batch in train_loader_rgb:
            print(f"✅ RGB批次数据形状:")
            print(f"   - sample_id: {len(batch['sample_id'])}")
            print(f"   - label: {batch['label'].shape}")
            print(f"   - images: {batch['sequence_data']['images'].shape}")
            print(f"   - joints: {batch['sequence_data']['joints'].shape}")
            print(f"   - timestamps: {batch['sequence_data']['timestamps'].shape}")
            print(f"   - labels: {batch['sequence_data']['labels'].shape}")
            break
        
        # 测试灰度模式
        print("\n测试灰度模式...")
        train_loader_gray, val_loader_gray, normalizer_gray = get_ball_counting_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=2,
            sequence_length=11,
            normalize=True,
            image_mode="grayscale",
            num_workers=0
        )
        
        for batch in train_loader_gray:
            print(f"✅ 灰度批次数据形状:")
            print(f"   - images: {batch['sequence_data']['images'].shape}")
            break
        
        print("✅ 数据加载器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试模型"""
    print("\n" + "="*50)
    print("测试模型")
    print("="*50)
    
    try:
        # 测试RGB模式模型
        print("测试RGB模式模型...")
        model_rgb = EmbodiedCountingModel(
            cnn_layers=2,  # 简化模型便于测试
            cnn_channels=[32, 64],
            lstm_layers=1,
            lstm_hidden_size=128,
            feature_dim=128,
            attention_heads=2,
            joint_dim=7,
            input_channels=3,  # RGB
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size, seq_len = 2, 5
        test_sequence_data = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224),
            'joints': torch.randn(batch_size, seq_len, 7),
            'timestamps': torch.randn(batch_size, seq_len),
            'labels': torch.randint(0, 11, (batch_size, seq_len))
        }
        
        # 前向传播
        outputs = model_rgb(test_sequence_data, use_teacher_forcing=True)
        
        print(f"✅ RGB模型输出形状:")
        print(f"   - counts: {outputs['counts'].shape}")
        print(f"   - joints: {outputs['joints'].shape}")
        
        # 测试灰度模式模型
        print("\n测试灰度模式模型...")
        model_gray = EmbodiedCountingModel(
            cnn_layers=2,
            cnn_channels=[32, 64],
            lstm_layers=1,
            lstm_hidden_size=128,
            feature_dim=128,
            attention_heads=2,
            joint_dim=7,
            input_channels=1,  # 灰度
            dropout=0.1
        )
        
        test_sequence_data_gray = test_sequence_data.copy()
        test_sequence_data_gray['images'] = torch.randn(batch_size, seq_len, 1, 224, 224)
        
        outputs_gray = model_gray(test_sequence_data_gray, use_teacher_forcing=True)
        
        print(f"✅ 灰度模型输出形状:")
        print(f"   - counts: {outputs_gray['counts'].shape}")
        print(f"   - joints: {outputs_gray['joints'].shape}")
        
        # 测试模块冻结功能
        print("\n测试模块冻结功能...")
        model_rgb.freeze_module('motion')
        model_rgb.unfreeze_module('motion')
        print("✅ 模块冻结/解冻功能正常")
        
        print("✅ 模型测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_creation():
    """测试训练器创建"""
    print("\n" + "="*50)
    print("测试训练器创建")
    print("="*50)
    
    try:
        # 创建测试配置
        config = {
            # 数据配置
            'data_root': "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection",
            'train_csv': "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv",
            'val_csv': "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv",
            'sequence_length': 11,
            'normalize': True,
            'image_mode': 'rgb',
            
            # 模型配置
            'model_config': {
                'cnn_layers': 2,
                'cnn_channels': [32, 64],
                'lstm_layers': 1,
                'lstm_hidden_size': 128,
                'feature_dim': 128,
                'attention_heads': 2,
                'joint_dim': 7,
                'dropout': 0.1
            },
            
            # 训练配置
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'adam_betas': (0.9, 0.999),
            'grad_clip_norm': 1.0,
            'total_epochs': 10,
            'stage_1_epochs': 3,
            'stage_2_epochs': 7,
            
            # 调度器
            'scheduler_type': 'none',
            'scheduler_patience': 5,
            
            # 保存和日志
            'save_dir': './test_checkpoints',
            'log_dir': './test_logs',
            'save_every': 2,
            'print_freq': 1,
            
            # 设备和数据加载
            'device': 'cpu',  # 使用CPU避免CUDA问题
            'num_workers': 0,
            
            # 随机种子
            'seed': 42
        }
        
        # 检查数据文件是否存在
        if not all(os.path.exists(path) for path in [config['data_root'], config['train_csv'], config['val_csv']]):
            print("❌ 数据文件不存在，跳过训练器测试")
            return False
        
        # 创建训练器
        print("创建训练器...")
        trainer = create_trainer(config)
        
        print(f"✅ 训练器创建成功")
        print(f"   - 训练数据: {len(trainer.train_loader.dataset)} 样本")
        print(f"   - 验证数据: {len(trainer.val_loader.dataset)} 样本")
        print(f"   - 模型参数: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # 测试训练阶段获取
        stage_name, stage_config = trainer.get_current_stage(0)
        print(f"   - 初始阶段: {stage_name}")
        
        # 清理测试文件
        import shutil
        if os.path.exists('./test_checkpoints'):
            shutil.rmtree('./test_checkpoints')
        if os.path.exists('./test_logs'):
            shutil.rmtree('./test_logs')
        
        print("✅ 训练器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "="*50)
    print("测试训练步骤")
    print("="*50)
    
    try:
        # 检查数据文件是否存在
        data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
        train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
        val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
        
        if not all(os.path.exists(path) for path in [data_root, train_csv, val_csv]):
            print("❌ 数据文件不存在，跳过训练步骤测试")
            return False
        
        # 创建简化的训练配置
        config = {
            'data_root': data_root,
            'train_csv': train_csv,
            'val_csv': val_csv,
            'sequence_length': 11,
            'normalize': True,
            'image_mode': 'rgb',
            'model_config': {
                'cnn_layers': 2,
                'cnn_channels': [16, 32],
                'lstm_layers': 1,
                'lstm_hidden_size': 64,
                'feature_dim': 64,
                'attention_heads': 2,
                'joint_dim': 7,
                'dropout': 0.1
            },
            'batch_size': 2,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'adam_betas': (0.9, 0.999),
            'grad_clip_norm': 1.0,
            'total_epochs': 2,
            'stage_1_epochs': 1,
            'stage_2_epochs': 2,
            'scheduler_type': 'none',
            'scheduler_patience': 5,
            'save_dir': './test_checkpoints',
            'log_dir': './test_logs',
            'save_every': 1,
            'print_freq': 1,
            'device': 'cpu',
            'num_workers': 0,
            'seed': 42
        }
        
        # 创建训练器
        trainer = create_trainer(config)
        
        # 测试一个训练步骤
        print("测试训练步骤...")
        stage_name, stage_config = trainer.get_current_stage(0)
        train_loss, train_metrics = trainer.train_one_epoch(0, stage_config)
        
        print(f"✅ 训练步骤完成:")
        print(f"   - 训练损失: {train_loss:.4f}")
        print(f"   - 计数准确率: {train_metrics['count_accuracy']:.4f}")
        print(f"   - 关节MSE: {train_metrics['joint_mse']:.6f}")
        
        # 测试验证步骤
        print("测试验证步骤...")
        val_loss, val_metrics, cm = trainer.validate(0, stage_config)
        
        print(f"✅ 验证步骤完成:")
        print(f"   - 验证损失: {val_loss:.4f}")
        print(f"   - 计数准确率: {val_metrics['count_accuracy']:.4f}")
        
        # 清理测试文件
        import shutil
        if os.path.exists('./test_checkpoints'):
            shutil.rmtree('./test_checkpoints')
        if os.path.exists('./test_logs'):
            shutil.rmtree('./test_logs')
        
        print("✅ 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始集成测试...")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("数据加载器", test_data_loader()))
    test_results.append(("模型", test_model()))
    test_results.append(("训练器创建", test_trainer_creation()))
    test_results.append(("训练步骤", test_training_step()))
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！你的代码修改是成功的。")
        print("\n现在你可以运行:")
        print("python Main.py --batch_size 4 --total_epochs 10")
    else:
        print("⚠️  部分测试失败，请检查失败的部分。")
    
    return passed == total


if __name__ == "__main__":
    main()