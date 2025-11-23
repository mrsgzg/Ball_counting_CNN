"""
AlexNet纯视觉模型训练脚本 - 对比预训练与非预训练效果
使用单图像数据加载器，保持与具身模型实验一致的训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import time
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib为非交互式后端
plt.switch_backend('Agg')

# 导入我们的模型和数据加载器
from Model_alexnet_visual import create_visual_model
from DataLoader_single_image import get_single_image_data_loaders


class VisualOnlyTrainer:
    """纯视觉模型训练器 - 与具身模型保持一致的训练流程"""
    
    def __init__(self, model, train_loader, val_loader, config, device, log_dir, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # 创建checkpoint目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard记录器
        self.writer = SummaryWriter(log_dir)
        
        # 优化器 - 与具身模型保持一致
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=config.get('adam_betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 梯度裁剪阈值
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)
        
        # 训练状态记录
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # 记录配置到TensorBoard
        config_text = f"Model: Visual Only - {'Pretrained' if config['use_pretrain'] else 'No Pretrain'}\n"
        config_text += f"Learning Rate: {config['learning_rate']}\n"
        config_text += f"Batch Size: {config['batch_size']}\n"
        config_text += f"Image Mode: {config.get('image_mode', 'rgb')}\n"
        self.writer.add_text('Config', config_text, 0)
        
    def _create_scheduler(self):
        """创建学习率调度器 - 与具身模型保持一致"""
        scheduler_type = self.config.get('scheduler_type', 'none')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('total_epochs', 1000)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=self.config.get('scheduler_patience', 5)
            )
        else:
            return None
    
    def compute_loss(self, logits, labels):
        """计算损失 - 简单的分类损失"""
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def compute_metrics(self, logits, labels):
        """计算评估指标"""
        metrics = {}
        
        # 预测标签
        pred_labels = torch.argmax(logits, dim=-1)
        
        # 准确率
        metrics['accuracy'] = (pred_labels == labels).float().mean().item()
        
        # Top-3准确率
        top3_pred = torch.topk(logits, k=min(3, logits.size(1)), dim=-1)[1]
        top3_correct = (top3_pred == labels.unsqueeze(1)).any(dim=1)
        metrics['top3_accuracy'] = top3_correct.float().mean().item()
        
        return metrics
    
    def compute_per_digit_accuracy(self, all_preds, all_labels):
        """计算每个数字的准确率"""
        per_digit_acc = {}
        unique_labels = torch.unique(all_labels)
        
        for digit in range(11):  # 0-10的球数
            mask = all_labels == digit
            if mask.sum() > 0:
                digit_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
                per_digit_acc[f'digit_{digit}_accuracy'] = digit_acc
            else:
                per_digit_acc[f'digit_{digit}_accuracy'] = 0.0
        
        return per_digit_acc
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 收集所有预测用于计算per-digit accuracy
        all_preds = []
        all_labels = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # 数据准备
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            logits = self.model(images)
            
            # 计算损失
            loss = self.compute_loss(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * images.size(0)
            pred_labels = torch.argmax(logits, dim=-1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += images.size(0)
            
            # 收集预测
            all_preds.append(pred_labels.cpu())
            all_labels.append(labels.cpu())
            
            # 更新进度条
            current_acc = total_correct / total_samples
            pbar.set_postfix({'loss': loss.item(), 'acc': f'{current_acc:.4f}'})
            
            # 记录batch级别的损失到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
        
        # 计算epoch级别的指标
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        
        # 合并所有预测
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算per-digit accuracy
        per_digit_metrics = self.compute_per_digit_accuracy(all_preds, all_labels)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Train_Accuracy', avg_accuracy, epoch)
        
        for key, value in per_digit_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        return avg_loss, avg_accuracy, per_digit_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证 - 包含每个数字的准确率"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 收集所有预测和真实标签
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            # 数据准备
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            logits = self.model(images)
            
            # 计算损失
            loss = self.compute_loss(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            # 计算准确率
            pred_labels = torch.argmax(logits, dim=-1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += images.size(0)
            
            # 收集预测
            all_preds.append(pred_labels.cpu())
            all_labels.append(labels.cpu())
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        
        # 合并所有预测
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算per-digit accuracy
        per_digit_metrics = self.compute_per_digit_accuracy(all_preds, all_labels)
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=list(range(11)))
        
        # 绘制并保存混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(11), yticklabels=range(11))
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图像到TensorBoard
        self.writer.add_figure('Confusion_Matrix', plt.gcf(), epoch)
        plt.close()
        
        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/Val_Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Val_Accuracy', avg_accuracy, epoch)
        
        for key, value in per_digit_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # 检查是否是最佳模型
        is_best = avg_accuracy > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = avg_accuracy
            self.best_val_loss = avg_loss
        
        return avg_loss, avg_accuracy, per_digit_metrics, is_best
    
    def save_checkpoint(self, epoch, val_loss, val_accuracy, is_best=False):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'model_info': self.model.get_model_info()
        }
        
        # 保存最新的checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型 (准确率: {val_accuracy:.4f})")
        
        # 定期保存checkpoint
        if epoch % self.config.get('save_every', 100) == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self, num_epochs):
        """完整的训练流程"""
        print(f"\n🚀 开始训练 - {'预训练' if self.config['use_pretrain'] else '无预训练'} AlexNet")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        # 💾 保存初始模型（epoch 0）
        if self.config.get('save_checkpoints', True):
            print("💾 保存初始模型 (epoch 0)...")
            
            # 先进行一次验证，获取初始性能
            print("📊 评估初始模型性能...")
            initial_val_loss, initial_val_acc, initial_per_digit, _ = self.validate(0)
            
            # 保存初始checkpoint
            initial_checkpoint = {
                'epoch': 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': initial_val_loss,
                'val_accuracy': initial_val_acc,
                'best_val_accuracy': 0.0,
                'config': self.config,
                'model_info': self.model.get_model_info()
            }
            
            # 保存为 checkpoint_epoch_0.pth
            epoch0_path = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_0.pth')
            torch.save(initial_checkpoint, epoch0_path)
            
            # 同时更新 latest_checkpoint.pth
            latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
            torch.save(initial_checkpoint, latest_path)
            
            print(f"✅ 初始模型已保存")
            print(f"   初始验证损失: {initial_val_loss:.4f}")
            print(f"   初始验证准确率: {initial_val_acc:.4f}")
            
            # 记录初始性能到训练历史
            initial_history = {
                'epoch': 0,
                'train_loss': float('inf'),
                'train_acc': 0.0,
                'val_loss': initial_val_loss,
                'val_acc': initial_val_acc,
                'learning_rate': self.config['learning_rate'],
                'epoch_time': 0.0,
                **initial_per_digit,
                **{f'val_{k}': v for k, v in initial_per_digit.items()}
            }
            self.training_history.append(initial_history)

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc, train_per_digit = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_per_digit, is_best = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 保存训练历史
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                **train_per_digit,
                **{f'val_{k}': v for k, v in val_per_digit.items()}
            }
            self.training_history.append(history_entry)
            
            # 保存checkpoint
            if self.config.get('save_checkpoints', True):
                self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # 打印进度
            if epoch % self.config.get('print_every', 10) == 0:
                elapsed_time = time.time() - start_time
                avg_epoch_time = elapsed_time / epoch
                remaining_epochs = num_epochs - epoch
                eta = avg_epoch_time * remaining_epochs
                
                print(f"\nEpoch [{epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")
                
                # 打印部分per-digit准确率
                print("Per-digit Val Accuracy:", end=" ")
                for digit in [0, 1, 5, 10]:  # 打印几个关键数字
                    key = f'digit_{digit}_accuracy'
                    if key in val_per_digit:
                        print(f"[{digit}]: {val_per_digit[key]:.3f}", end=" ")
                print()
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成!")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        
        # 保存训练历史
        history_df = pd.DataFrame(self.training_history)
        history_path = os.path.join(self.checkpoint_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"训练历史已保存: {history_path}")
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        return self.training_history


def run_single_experiment(config, data_config, save_dir):
    """运行单个实验"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 创建保存目录
    model_name = f"alexnet_{'pretrain' if config['use_pretrain'] else 'no_pretrain'}_seed_{seed}"
    experiment_dir = os.path.join(save_dir, model_name)
    log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"实验: {model_name}")
    print(f"保存目录: {experiment_dir}")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_loader, val_loader = get_single_image_data_loaders(
        train_csv_path=data_config['train_csv'],
        val_csv_path=data_config['val_csv'],
        data_root=data_config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        image_mode=config.get('image_mode', 'rgb'),
        normalize_images=config.get('normalize_images', True)
    )
    
    # 创建模型
    model = create_visual_model(config, use_pretrain=config['use_pretrain'])
    model = model.to(device)
    
    # 创建训练器
    trainer = VisualOnlyTrainer(
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
    history = trainer.train(num_epochs=config['total_epochs'])
    training_time = time.time() - start_time
    
    # 返回实验结果
    result = {
        'model_type': model_name,
        'use_pretrain': config['use_pretrain'],
        'seed': seed,
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_val_loss': trainer.best_val_loss,
        'final_val_accuracy': history[-1]['val_acc'] if history else 0.0,
        'total_epochs': config['total_epochs'],
        'training_time_hours': training_time / 3600,
        'experiment_dir': experiment_dir
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='AlexNet纯视觉模型训练 - 预训练对比实验')
    
    # 数据路径 - 与原实验保持一致
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection')
    parser.add_argument('--train_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv')
    parser.add_argument('--val_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv')
    
    # 实验参数
    parser.add_argument('--total_epochs', type=int, default=1000,
                       help='训练总epoch数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[2048, 4096, 9999],
                       help='随机种子列表')
    
    # 模型参数
    parser.add_argument('--image_mode', type=str, default='rgb',
                       choices=['rgb', 'grayscale'],
                       help='图像模式')
    parser.add_argument('--run_both', action='store_true', default=True,
                       help='同时运行预训练和非预训练实验')
    parser.add_argument('--use_pretrain', action='store_true',
                       help='只运行预训练模型')
    parser.add_argument('--no_pretrain', action='store_true',
                       help='只运行非预训练模型')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./alexnet_visual_only_experiments',
                       help='结果保存目录')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器进程数')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='是否保存模型checkpoints')
    parser.add_argument('--save_every', type=int, default=100,
                       help='每多少个epoch保存一次checkpoint')
    parser.add_argument('--print_every', type=int, default=10,
                       help='每多少个epoch打印一次进度')
    
    args = parser.parse_args()
    
    # 确定要运行的实验
    experiments = []
    if args.run_both or (not args.use_pretrain and not args.no_pretrain):
        experiments = [True, False]  # 同时运行预训练和非预训练
    elif args.use_pretrain:
        experiments = [True]  # 只运行预训练
    elif args.no_pretrain:
        experiments = [False]  # 只运行非预训练
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'alexnet_visual_comparison_{timestamp}')
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
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_mode': args.image_mode,
        'num_workers': args.num_workers,
        'save_checkpoints': args.save_checkpoints,
        'save_every': args.save_every,
        'print_every': args.print_every,
        'model_config': {
            'feature_dim': 256,
            'dropout': 0.5
        },
        # 与具身模型保持一致的训练参数
        'adam_betas': (0.9, 0.999),
        'weight_decay': 1e-5,
        'grad_clip_norm': 1.0,
        'scheduler_type': 'cosine',
        'normalize_images': True
    }
    
    # 记录所有实验结果
    all_results = []
    results_file = os.path.join(save_dir, 'experiment_results.csv')
    
    print(f"🚀 开始AlexNet纯视觉对比实验")
    print(f"实验类型: {['预训练', '非预训练'] if len(experiments) == 2 else ['预训练' if experiments[0] else '非预训练']}")
    print(f"随机种子: {args.seeds}")
    print(f"总实验数: {len(experiments) * len(args.seeds)}")
    print(f"每个实验训练epochs: {args.total_epochs}")
    print(f"结果保存: {save_dir}")
    
    # 运行所有实验
    total_experiments = len(experiments) * len(args.seeds)
    current_exp = 0
    start_time = time.time()
    
    for use_pretrain in experiments:
        for seed in args.seeds:
            current_exp += 1
            
            # 更新配置
            config = base_config.copy()
            config['use_pretrain'] = use_pretrain
            config['seed'] = seed
            
            # 显示进度
            elapsed_time = time.time() - start_time
            avg_time_per_exp = elapsed_time / current_exp if current_exp > 0 else 0
            remaining_time = avg_time_per_exp * (total_experiments - current_exp)
            
            print(f"\n📊 进度: {current_exp}/{total_experiments}")
            print(f"⏱️  已用时: {elapsed_time/3600:.1f}h, 预计剩余: {remaining_time/3600:.1f}h")
            
            # 运行实验
            result = run_single_experiment(config, data_config, save_dir)
            all_results.append(result)
            
            # 保存中间结果
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_file, index=False)
            print(f"💾 保存中间结果: {results_file}")
    
    # 生成最终报告
    print(f"\n📊 生成实验报告...")
    results_df = pd.DataFrame(all_results)
    
    # 计算统计摘要
    summary = results_df.groupby('use_pretrain').agg({
        'best_val_accuracy': ['mean', 'std', 'max'],
        'final_val_accuracy': ['mean', 'std'],
        'training_time_hours': ['mean', 'sum']
    }).round(4)
    
    # 保存摘要
    summary_file = os.path.join(save_dir, 'summary_stats.csv')
    summary.to_csv(summary_file)
    
    # 打印摘要
    print("\n📈 实验结果摘要:")
    print("="*60)
    print(summary)
    print("="*60)
    
    # 生成Markdown报告
    report_content = f"""# AlexNet纯视觉模型对比实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验概述

- **模型类型**: AlexNet纯视觉分类
- **对比内容**: 预训练 vs 非预训练
- **任务**: 单图像球数分类 (0-10)
- **训练epochs**: {args.total_epochs}
- **随机种子**: {args.seeds}
- **批次大小**: {args.batch_size}
- **学习率**: {args.learning_rate}
- **图像模式**: {args.image_mode}

## 实验结果

### 准确率对比

| 模型类型 | 最佳验证准确率 (mean±std) | 最高准确率 | 最终验证准确率 (mean±std) |
|---------|-------------------------|----------|-------------------------|
"""
    
    for use_pretrain in [True, False]:
        model_type = "预训练AlexNet" if use_pretrain else "无预训练AlexNet"
        pretrain_results = results_df[results_df['use_pretrain'] == use_pretrain]
        if len(pretrain_results) > 0:
            best_mean = pretrain_results['best_val_accuracy'].mean()
            best_std = pretrain_results['best_val_accuracy'].std()
            best_max = pretrain_results['best_val_accuracy'].max()
            final_mean = pretrain_results['final_val_accuracy'].mean()
            final_std = pretrain_results['final_val_accuracy'].std()
            
            report_content += f"| {model_type} | {best_mean:.4f}±{best_std:.4f} | {best_max:.4f} | {final_mean:.4f}±{final_std:.4f} |\n"
    
    report_content += f"""

### 训练效率

| 模型类型 | 平均训练时间 (小时) | 总训练时间 (小时) |
|---------|------------------|----------------|
"""
    
    for use_pretrain in [True, False]:
        model_type = "预训练AlexNet" if use_pretrain else "无预训练AlexNet"
        pretrain_results = results_df[results_df['use_pretrain'] == use_pretrain]
        if len(pretrain_results) > 0:
            avg_time = pretrain_results['training_time_hours'].mean()
            total_time = pretrain_results['training_time_hours'].sum()
            report_content += f"| {model_type} | {avg_time:.2f} | {total_time:.2f} |\n"
    
    report_content += f"""

## 结论

基于实验结果：
1. 预训练模型相比非预训练模型的性能提升: {((results_df[results_df['use_pretrain']==True]['best_val_accuracy'].mean() / results_df[results_df['use_pretrain']==False]['best_val_accuracy'].mean() - 1) * 100):.1f}%
2. 两种模型的训练稳定性（标准差）对比
3. 训练效率差异

## 文件说明

- 详细结果: `experiment_results.csv`
- 统计摘要: `summary_stats.csv`
- TensorBoard日志: 各模型的 `tensorboard_logs/` 目录
- 模型checkpoints: 各模型的 `checkpoints/` 目录

## 查看TensorBoard

```bash
tensorboard --logdir {save_dir}
```

## 与具身模型对比

此实验可与具身模型实验结果进行对比，以评估具身信息对计数任务的贡献。
"""
    
    report_file = os.path.join(save_dir, 'experiment_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    total_time = time.time() - start_time
    print(f"\n🎉 所有实验完成!")
    print(f"⏱️  总耗时: {total_time/3600:.1f} 小时")
    print(f"📊 详细结果: {results_file}")
    print(f"📈 统计摘要: {summary_file}")
    print(f"📋 实验报告: {report_file}")
    print(f"💾 所有文件保存在: {save_dir}")


if __name__ == "__main__":
    main()