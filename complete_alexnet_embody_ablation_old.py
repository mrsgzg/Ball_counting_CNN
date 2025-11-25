"""
并行消融实验主脚本
支持在单个A100 GPU上同时运行多个实验（通过CUDA MPS或多进程）
"""

import torch
import torch.multiprocessing as mp
import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import signal
import sys

# 设置多进程启动方法
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ==================== 消融实验配置 ====================
ABLATION_CONFIGS = {
    'full_model': {
        'name': 'Full Model',
        'description': 'Complete model with all components',
        'model_variant': None,  # 使用原始模型
        'data_wrapper': None,
        'loss_modifications': {}
    },
    
    'no_forward_model': {
        'name': 'No Forward Model',
        'description': 'Remove motion prediction (forward model)',
        'model_variant': 'no_forward_model',
        'data_wrapper': None,
        'loss_modifications': {
            'embodiment_loss_weight': 0.0  # 没有motion loss
        }
    },
    
    'no_attention': {
        'name': 'No Spatial Attention',
        'description': 'Replace attention with global average pooling',
        'model_variant': 'no_attention',
        'data_wrapper': None,
        'loss_modifications': {}
    },
    
    'late_fusion': {
        'name': 'Late Fusion',
        'description': 'Fuse vision and joints after LSTM',
        'model_variant': 'late_fusion',
        'data_wrapper': None,
        'loss_modifications': {}
    },
    
    'shuffled_batch': {
        'name': 'Shuffled Batch',
        'description': 'Shuffle vision-joint pairing across samples',
        'model_variant': None,
        'data_wrapper': 'shuffled_batch',
        'loss_modifications': {}
    },
    
    'shuffled_temporal': {
        'name': 'Shuffled Temporal',
        'description': 'Shuffle temporal order within sequences',
        'model_variant': None,
        'data_wrapper': 'shuffled_temporal',
        'loss_modifications': {}
    }
}


def run_single_ablation_experiment(
    ablation_type,
    model_type,
    seed,
    data_config,
    base_config,
    save_dir,
    gpu_id,
    process_id
):
    """
    运行单个消融实验
    
    Args:
        ablation_type: 消融类型
        model_type: 视觉编码器类型 ('baseline', 'alexnet_pretrain', 'alexnet_no_pretrain')
        seed: 随机种子
        data_config: 数据配置
        base_config: 基础配置
        save_dir: 保存目录
        gpu_id: GPU ID
        process_id: 进程ID（用于日志）
    """
    try:
        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')  # 因为CUDA_VISIBLE_DEVICES已设置，所以总是0
        
        # 导入必要的模块
        from complete_alexnet_embody_experiment import EmbodiedTrainer
        from DataLoader_embodiment import get_ball_counting_data_loaders
        from DataLoader_embodiment_ablation import wrap_dataloader
        from Model_alexnet_embodiment_ablation import create_ablation_model
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 创建实验目录
        experiment_name = f"{ablation_type}_{model_type}_seed{seed}"
        experiment_dir = os.path.join(save_dir, experiment_name)
        log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"[Process {process_id}] Starting: {experiment_name} on GPU {gpu_id}")
        
        # 获取消融配置
        ablation_config = ABLATION_CONFIGS[ablation_type]
        
        # 构建完整配置
        config = base_config.copy()
        config['model_type'] = model_type
        config['seed'] = seed
        config['ablation_type'] = ablation_type
        config['ablation_name'] = ablation_config['name']
        
        # 应用损失修改
        config.update(ablation_config['loss_modifications'])
        
        # 保存配置
        config_path = os.path.join(experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # 创建数据加载器
        train_loader, val_loader, normalizer = get_ball_counting_data_loaders(
            train_csv_path=data_config['train_csv'],
            val_csv_path=data_config['val_csv'],
            data_root=data_config['data_root'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            normalize=config['normalize'],
            num_workers=config['num_workers'],
            image_mode=config['image_mode']
        )
        
        # 应用数据包装（如果需要）
        if ablation_config['data_wrapper'] is not None:
            print(f"[Process {process_id}] Applying data wrapper: {ablation_config['data_wrapper']}")
            train_loader = wrap_dataloader(train_loader, ablation_config['data_wrapper'], seed=seed)
            val_loader = wrap_dataloader(val_loader, ablation_config['data_wrapper'], seed=seed)
        
        # 创建模型
        if ablation_config['model_variant'] is not None:
            # 使用消融变体
            model = create_ablation_model(config, ablation_config['model_variant'])
        else:
            # 使用原始模型
            from Model_alexnet_embodiment import create_model
            model = create_model(config, model_type=model_type)
        
        model = model.to(device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Process {process_id}] Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # 创建训练器
        trainer = EmbodiedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir
        )
        
        # 开始训练
        start_time = time.time()
        print(f"[Process {process_id}] Training started: {experiment_name}")
        
        history = trainer.train(num_epochs=config['total_epochs'])
        
        training_time = time.time() - start_time
        
        # 收集结果
        final_metrics = history[-1] if history else {}
        result = {
            'ablation_type': ablation_type,
            'ablation_name': ablation_config['name'],
            'model_type': model_type,
            'seed': seed,
            'best_val_accuracy': trainer.best_val_accuracy,
            'best_val_loss': trainer.best_val_loss,
            'final_val_accuracy': final_metrics.get('val_count_acc', 0.0),
            'final_val_final_accuracy': final_metrics.get('val_final_acc', 0.0),
            'final_val_true_final_accuracy': final_metrics.get('val_true_final_acc', 0.0),
            'final_joint_mse': final_metrics.get('joint_mse', 0.0) if 'joint_mse' in final_metrics else None,
            'total_epochs': config['total_epochs'],
            'training_time_hours': training_time / 3600,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'experiment_dir': experiment_dir,
            'gpu_id': gpu_id,
            'process_id': process_id
        }
        
        # 保存结果
        result_file = os.path.join(experiment_dir, 'result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"[Process {process_id}] Completed: {experiment_name}")
        print(f"[Process {process_id}] Best accuracy: {trainer.best_val_accuracy:.4f}")
        print(f"[Process {process_id}] Training time: {training_time/3600:.2f} hours")
        
        return result
        
    except Exception as e:
        print(f"[Process {process_id}] ERROR in {ablation_type}_{model_type}_seed{seed}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def worker_process(task_queue, result_queue, gpu_id, worker_id, data_config, base_config, save_dir):
    """
    工作进程：从队列获取任务并执行
    
    Args:
        task_queue: 任务队列
        result_queue: 结果队列
        gpu_id: GPU ID
        worker_id: 工作进程ID
        data_config: 数据配置
        base_config: 基础配置
        save_dir: 保存目录
    """
    print(f"[Worker {worker_id}] Started on GPU {gpu_id}")
    
    while True:
        try:
            # 从队列获取任务
            task = task_queue.get(timeout=1)
            
            if task is None:  # 终止信号
                print(f"[Worker {worker_id}] Received termination signal")
                break
            
            ablation_type, model_type, seed = task
            
            # 执行实验
            result = run_single_ablation_experiment(
                ablation_type=ablation_type,
                model_type=model_type,
                seed=seed,
                data_config=data_config,
                base_config=base_config,
                save_dir=save_dir,
                gpu_id=gpu_id,
                process_id=worker_id
            )
            
            # 将结果放入结果队列
            if result is not None:
                result_queue.put(result)
            
        except Exception as e:
            if "Empty" not in str(e):
                print(f"[Worker {worker_id}] Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"[Worker {worker_id}] Finished")


def main():
    parser = argparse.ArgumentParser(description='并行消融实验 - A100优化版')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection')
    parser.add_argument('--train_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv')
    parser.add_argument('--val_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv')
    
    # 实验参数
    parser.add_argument('--total_epochs', type=int, default=100,
                       help='训练总epoch数')
    parser.add_argument('--ablations', nargs='+', 
                       default=['no_forward_model', 'no_attention', 
                               'late_fusion', 'shuffled_batch', 'shuffled_temporal'],
                       choices=list(ABLATION_CONFIGS.keys()),
                       help='要运行的消融实验')
    parser.add_argument('--model_types', nargs='+',
                       default=['alexnet_no_pretrain'],
                       choices=['baseline', 'alexnet_no_pretrain', 'alexnet_pretrain'],
                       help='视觉编码器类型')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[2048, 4096, 9999],
                       help='随机种子列表')
    
    # 并行配置
    parser.add_argument('--num_parallel', type=int, default=3,
                       help='同时运行的实验数量（A100可以跑3个）')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='使用的GPU ID')
    
    # 保存配置
    parser.add_argument('--save_dir', type=str, default='./ablation_experiments',
                       help='结果保存目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称（可选）')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f'ablation_parallel_{timestamp}'
    save_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据配置
    data_config = {
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv
    }
    
    # 基础配置
    base_config = {
        'total_epochs': args.total_epochs,
        'batch_size': 16,
        'sequence_length': 11,
        'learning_rate': 1e-4,
        'image_mode': 'rgb',
        'num_workers': 4,
        'save_checkpoints': True,
        'save_every': 10,
        'print_every': 10,
        'model_config': {
            'cnn_layers': 3,
            'cnn_channels': [64, 128, 256],
            'lstm_layers': 2,
            'lstm_hidden_size': 512,
            'feature_dim': 256,
            'joint_dim': 7,
            'dropout': 0.1,
            'use_fovea_bias': True
        },
        'adam_betas': (0.9, 0.999),
        'weight_decay': 1e-5,
        'grad_clip_norm': 1.0,
        'scheduler_type': 'cosine',
        'normalize': True,
        'embodiment_loss_weight': 0.3,
        'attention_loss_weight': 0.1
    }
    
    # 生成所有任务
    tasks = []
    for ablation in args.ablations:
        for model_type in args.model_types:
            for seed in args.seeds:
                tasks.append((ablation, model_type, seed))
    
    total_experiments = len(tasks)
    
    print("\n" + "="*80)
    print("🚀 并行消融实验启动")
    print("="*80)
    print(f"消融类型: {args.ablations}")
    print(f"模型类型: {args.model_types}")
    print(f"随机种子: {args.seeds}")
    print(f"总实验数: {total_experiments}")
    print(f"并行数量: {args.num_parallel}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"每个实验epochs: {args.total_epochs}")
    print(f"保存目录: {save_dir}")
    print("="*80 + "\n")
    
    # 保存实验配置
    experiment_config = {
        'timestamp': timestamp,
        'ablations': args.ablations,
        'model_types': args.model_types,
        'seeds': args.seeds,
        'total_experiments': total_experiments,
        'num_parallel': args.num_parallel,
        'gpu_id': args.gpu_id,
        'total_epochs': args.total_epochs,
        'data_config': data_config,
        'base_config': base_config
    }
    
    config_file = os.path.join(save_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将所有任务放入队列
    for task in tasks:
        task_queue.put(task)
    
    # 添加终止信号
    for _ in range(args.num_parallel):
        task_queue.put(None)
    
    # 启动工作进程
    processes = []
    for i in range(args.num_parallel):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, args.gpu_id, i, data_config, base_config, save_dir)
        )
        p.start()
        processes.append(p)
        print(f"✓ 启动工作进程 {i}")
    
    # 收集结果
    all_results = []
    results_file = os.path.join(save_dir, 'all_results.csv')
    
    start_time = time.time()
    completed = 0
    
    print("\n" + "="*80)
    print("📊 实验进度监控")
    print("="*80)
    
    # 实时收集结果
    while completed < total_experiments:
        try:
            result = result_queue.get(timeout=10)
            completed += 1
            all_results.append(result)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (total_experiments - completed)
            
            print(f"\n[{completed}/{total_experiments}] 完成实验:")
            print(f"  消融: {result['ablation_name']}")
            print(f"  模型: {result['model_type']}")
            print(f"  种子: {result['seed']}")
            print(f"  最佳准确率: {result['best_val_accuracy']:.4f}")
            print(f"  训练时间: {result['training_time_hours']:.2f}h")
            print(f"  已用时: {elapsed/3600:.1f}h, 预计剩余: {remaining/3600:.1f}h")
            
            # 实时保存结果
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_file, index=False)
            
        except Exception as e:
            if "Empty" not in str(e):
                print(f"⚠️  结果队列错误: {e}")
    
    # 等待所有进程结束
    print("\n等待所有进程结束...")
    for i, p in enumerate(processes):
        p.join()
        print(f"✓ 进程 {i} 已结束")
    
    total_time = time.time() - start_time
    
    # 生成汇总报告
    print("\n" + "="*80)
    print("📊 生成汇总报告")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    
    # 按消融类型分组统计
    summary = results_df.groupby(['ablation_type', 'model_type']).agg({
        'best_val_accuracy': ['mean', 'std', 'min', 'max'],
        'final_val_true_final_accuracy': ['mean', 'std'],
        'training_time_hours': ['mean', 'sum']
    }).round(4)
    
    summary_file = os.path.join(save_dir, 'summary_statistics.csv')
    summary.to_csv(summary_file)
    
    print("\n统计摘要:")
    print(summary)
    
    # 生成Markdown报告
    report_content = f"""# 消融实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
实验名称: {experiment_name}

## 实验配置

- **消融类型**: {', '.join(args.ablations)}
- **模型类型**: {', '.join(args.model_types)}
- **随机种子**: {args.seeds}
- **总实验数**: {total_experiments}
- **并行数量**: {args.num_parallel}
- **训练epochs**: {args.total_epochs}
- **总耗时**: {total_time/3600:.2f} 小时

## 消融类型说明

"""
    
    for ablation, config in ABLATION_CONFIGS.items():
        if ablation in args.ablations:
            report_content += f"### {config['name']}\n{config['description']}\n\n"
    
    report_content += """## 实验结果

### 按消融类型统计

| 消融类型 | 模型 | 平均准确率 | 标准差 | 最小值 | 最大值 |
|---------|------|-----------|--------|--------|--------|
"""
    
    for (ablation, model), row in results_df.groupby(['ablation_type', 'model_type']):
        mean_acc = row['best_val_accuracy'].mean()
        std_acc = row['best_val_accuracy'].std()
        min_acc = row['best_val_accuracy'].min()
        max_acc = row['best_val_accuracy'].max()
        abl_name = ABLATION_CONFIGS[ablation]['name']
        report_content += f"| {abl_name} | {model} | {mean_acc:.4f} | {std_acc:.4f} | {min_acc:.4f} | {max_acc:.4f} |\n"
    
    report_content += f"""

## 文件说明

- `all_results.csv`: 所有实验的详细结果
- `summary_statistics.csv`: 统计摘要
- `experiment_config.json`: 实验配置
- 各实验目录包含:
  - `config.json`: 实验配置
  - `result.json`: 实验结果
  - `checkpoints/`: 模型检查点
  - `tensorboard_logs/`: TensorBoard日志

## 查看TensorBoard

```bash
tensorboard --logdir {save_dir}
```

## 主要发现

TODO: 根据实验结果填写关键发现

"""
    
    report_file = os.path.join(save_dir, 'REPORT.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n" + "="*80)
    print("🎉 所有实验完成！")
    print("="*80)
    print(f"⏱️  总耗时: {total_time/3600:.1f} 小时")
    print(f"📊 结果文件: {results_file}")
    print(f"📈 统计摘要: {summary_file}")
    print(f"📋 报告: {report_file}")
    print(f"💾 所有文件: {save_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()