"""
AlexNet具身模型完整训练脚本 - 修复版
保持与纯视觉训练版本一致的结构，修复TensorBoard显示异常问题
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
from Model_alexnet_embodiment import create_model
from DataLoader_embodiment import get_ball_counting_data_loaders


class EmbodiedTrainer:
    """具身模型训练器 - 修复版，与纯视觉训练保持一致的结构"""
    
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
        
        # 优化器 - 与纯视觉模型保持一致
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
        
        # 损失权重
        self.embodiment_loss_weight = config.get('embodiment_loss_weight', 0.3)
        self.attention_loss_weight = config.get('attention_loss_weight', 0.1)
        
        # 训练状态记录
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # 记录配置到TensorBoard
        config_text = f"Model: {config['model_type']}\n"
        config_text += f"Learning Rate: {config['learning_rate']}\n"
        config_text += f"Batch Size: {config['batch_size']}\n"
        config_text += f"Embodiment Loss Weight: {self.embodiment_loss_weight}\n"
        config_text += f"Image Mode: {config.get('image_mode', 'rgb')}\n"
        self.writer.add_text('Config', config_text, 0)
        
    def _create_scheduler(self):
        """创建学习率调度器 - 与纯视觉模型保持一致"""
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
    
    def compute_loss(self, outputs, targets):
        """计算损失 - 修复维度问题"""
        losses = {}
        
        # 1. 计数分类损失
        count_logits = outputs['counts']  # [batch, seq_len, 11]
        target_counts = targets['labels'].long()  # [batch, seq_len]
        
        # 展平用于计算损失
        batch_size, seq_len = count_logits.shape[:2]
        count_loss = F.cross_entropy(
            count_logits.view(-1, 11),
            target_counts.view(-1),
            ignore_index=-1
        )
        losses['count_loss'] = count_loss
        
        # 2. 动作回归损失（预测下一帧的关节位置）
        if outputs['joints'].shape[1] > 1:
            pred_joints = outputs['joints'][:, :-1]  # [batch, seq_len-1, 7]
            target_joints = targets['joints'][:, 1:]  # [batch, seq_len-1, 7]
            motion_loss = F.mse_loss(pred_joints, target_joints)
        else:
            motion_loss = torch.tensor(0.0, device=self.device)
        losses['motion_loss'] = motion_loss
        
        # 3. 注意力正则化损失（可选）
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']  # [batch, seq_len, H, W]
            batch_size, seq_len, H, W = attention_weights.shape
            attention_flat = attention_weights.view(batch_size * seq_len, -1)
            # 计算熵作为正则化
            attention_entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=1).mean()
            losses['attention_loss'] = -attention_entropy  # 负熵，鼓励集中注意力
        else:
            losses['attention_loss'] = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = (count_loss + 
                     self.embodiment_loss_weight * motion_loss +
                     self.attention_loss_weight * losses['attention_loss'])
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_metrics(self, outputs, targets):
        """计算评估指标 - 修复准确率计算"""
        metrics = {}
        
        # 计数分类指标
        count_logits = outputs['counts']  # [batch, seq_len, 11]
        pred_labels = torch.argmax(count_logits, dim=-1)  # [batch, seq_len]
        target_counts = targets['labels'].long()  # [batch, seq_len]
        
        # 1. 序列准确率（所有时间步的平均）
        valid_mask = target_counts >= 0
        if valid_mask.sum() > 0:
            metrics['count_accuracy'] = (pred_labels[valid_mask] == target_counts[valid_mask]).float().mean().item()
        else:
            metrics['count_accuracy'] = 0.0
        
        # 2. 最终计数准确率（序列最后一个时间步）
        final_pred = pred_labels[:, -1]
        final_target = target_counts[:, -1]
        metrics['final_count_accuracy'] = (final_pred == final_target).float().mean().item()
        
        # 3. 真实最终计数准确率（考虑实际序列长度）
        batch_size = pred_labels.shape[0]
        true_final_correct = 0
        
        for i in range(batch_size):
            # 找到真实的最终位置（最大标签值的位置）
            max_label = target_counts[i].max()
            final_positions = (target_counts[i] == max_label).nonzero(as_tuple=True)[0]
            if len(final_positions) > 0:
                true_final_pos = final_positions[0].item()
                if pred_labels[i, true_final_pos] == target_counts[i, true_final_pos]:
                    true_final_correct += 1
        
        metrics['true_final_count_accuracy'] = true_final_correct / batch_size
        
        # 4. 动作指标
        if outputs['joints'].shape[1] > 1:
            pred_joints = outputs['joints'][:, :-1]
            target_joints = targets['joints'][:, 1:]
            metrics['joint_mse'] = F.mse_loss(pred_joints, target_joints).item()
            metrics['joint_mae'] = F.l1_loss(pred_joints, target_joints).item()
        else:
            metrics['joint_mse'] = 0.0
            metrics['joint_mae'] = 0.0
        
        return metrics
    
    def compute_per_digit_accuracy(self, all_preds, all_labels):
        """计算每个数字的准确率"""
        per_digit_acc = {}
        
        # 展平预测和标签
        all_preds_flat = all_preds.view(-1)
        all_labels_flat = all_labels.view(-1)
        
        # 过滤有效标签
        valid_mask = all_labels_flat >= 0
        all_preds_flat = all_preds_flat[valid_mask]
        all_labels_flat = all_labels_flat[valid_mask]
        
        for digit in range(11):  # 0-10的球数
            mask = all_labels_flat == digit
            if mask.sum() > 0:
                digit_acc = (all_preds_flat[mask] == all_labels_flat[mask]).float().mean().item()
                per_digit_acc[f'digit_{digit}_accuracy'] = digit_acc
            else:
                per_digit_acc[f'digit_{digit}_accuracy'] = 0.0
        
        return per_digit_acc
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_count = 0
        epoch_metrics = defaultdict(float)
        
        # 收集所有预测用于计算per-digit accuracy
        all_preds = []
        all_labels = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # 数据准备
            sequence_data = {
                'images': batch['sequence_data']['images'].to(self.device),
                'joints': batch['sequence_data']['joints'].to(self.device),
                'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                'labels': batch['sequence_data']['labels'].to(self.device)
            }
            
            # 前向传播
            outputs = self.model(
                sequence_data=sequence_data,
                use_teacher_forcing=True,
                return_attention=True  # 获取注意力权重用于可视化
            )
            
            # 计算损失
            targets = {
                'labels': sequence_data['labels'],
                'joints': sequence_data['joints']
            }
            losses = self.compute_loss(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            # 统计
            total_loss += losses['total_loss'].item()
            total_count += 1
            
            # 计算指标
            with torch.no_grad():
                batch_metrics = self.compute_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                
                # 收集预测
                count_logits = outputs['counts']
                pred_labels = torch.argmax(count_logits, dim=-1)
                all_preds.append(pred_labels.cpu())
                all_labels.append(sequence_data['labels'].cpu())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': losses['total_loss'].item(), 
                'acc': batch_metrics['count_accuracy']
            })
            
            # 记录batch级别的损失到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Train_Loss', losses['total_loss'].item(), global_step)
            self.writer.add_scalar('Batch/Count_Loss', losses['count_loss'].item(), global_step)
            self.writer.add_scalar('Batch/Motion_Loss', losses['motion_loss'].item(), global_step)
        
        # 计算epoch级别的指标
        avg_loss = total_loss / total_count
        avg_metrics = {key: value / total_count for key, value in epoch_metrics.items()}
        
        # 合并所有预测
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算per-digit accuracy
        per_digit_metrics = self.compute_per_digit_accuracy(all_preds, all_labels)
        avg_metrics.update(per_digit_metrics)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Train_Count_Accuracy', avg_metrics['count_accuracy'], epoch)
        self.writer.add_scalar('Epoch/Train_Final_Accuracy', avg_metrics['final_count_accuracy'], epoch)
        
        # 记录Motion指标
        self.writer.add_scalar('Epoch/Train_Joint_MSE', avg_metrics['joint_mse'], epoch)
        self.writer.add_scalar('Epoch/Train_Joint_MAE', avg_metrics['joint_mae'], epoch)
        
        for key, value in per_digit_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证 - 包含每个数字的准确率和混淆矩阵"""
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        total_count = 0
        
        # 收集所有预测和真实标签
        all_preds = []
        all_labels = []
        all_final_preds = []
        all_final_labels = []
        
        for batch in self.val_loader:
            # 数据准备
            sequence_data = {
                'images': batch['sequence_data']['images'].to(self.device),
                'joints': batch['sequence_data']['joints'].to(self.device),
                'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                'labels': batch['sequence_data']['labels'].to(self.device)
            }
            
            # 前向传播（不使用teacher forcing）
            outputs = self.model(
                sequence_data=sequence_data,
                use_teacher_forcing=False,
                return_attention=True
            )
            
            # 计算损失
            targets = {
                'labels': sequence_data['labels'],
                'joints': sequence_data['joints']
            }
            losses = self.compute_loss(outputs, targets)
            total_loss += losses['total_loss'].item()
            
            # 计算指标
            metrics = self.compute_metrics(outputs, targets)
            for key, value in metrics.items():
                total_metrics[key] += value
            
            # 收集预测
            count_logits = outputs['counts']
            pred_labels = torch.argmax(count_logits, dim=-1)
            all_preds.append(pred_labels.cpu())
            all_labels.append(sequence_data['labels'].cpu())
            
            # 收集最终预测（用于混淆矩阵）
            all_final_preds.append(pred_labels[:, -1].cpu())
            all_final_labels.append(sequence_data['labels'][:, -1].cpu())
            
            total_count += 1
        
        # 计算平均指标
        avg_loss = total_loss / total_count
        avg_metrics = {key: value / total_count for key, value in total_metrics.items()}
        
        # 合并所有预测
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_final_preds = torch.cat(all_final_preds, dim=0)
        all_final_labels = torch.cat(all_final_labels, dim=0)
        
        # 计算per-digit accuracy
        per_digit_metrics = self.compute_per_digit_accuracy(all_preds, all_labels)
        avg_metrics.update(per_digit_metrics)
        
        # 计算混淆矩阵（基于最终预测）
        cm = confusion_matrix(all_final_labels.numpy(), all_final_preds.numpy(), labels=list(range(11)))
        
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
        self.writer.add_scalar('Epoch/Val_Count_Accuracy', avg_metrics['count_accuracy'], epoch)
        self.writer.add_scalar('Epoch/Val_Final_Accuracy', avg_metrics['final_count_accuracy'], epoch)
        self.writer.add_scalar('Epoch/Val_True_Final_Accuracy', avg_metrics['true_final_count_accuracy'], epoch)
        
        # 记录Motion指标
        self.writer.add_scalar('Epoch/Val_Joint_MSE', avg_metrics['joint_mse'], epoch)
        self.writer.add_scalar('Epoch/Val_Joint_MAE', avg_metrics['joint_mae'], epoch)
        
        for key, value in per_digit_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # 检查是否是最佳模型
        is_best = avg_metrics['true_final_count_accuracy'] > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = avg_metrics['true_final_count_accuracy']
            self.best_val_loss = avg_loss
        
        return avg_loss, avg_metrics, is_best
    
    def save_checkpoint(self, epoch, val_loss, val_metrics, is_best=False):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
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
            print(f"💾 保存最佳模型 (准确率: {val_metrics['true_final_count_accuracy']:.4f})")
        
        # 定期保存checkpoint
        if epoch % self.config.get('save_every', 100) == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self, num_epochs):
        """完整的训练流程"""
        print(f"\n🚀 开始训练 - {self.config['model_type']} 具身模型")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        # 💾 保存初始模型（epoch 0）
        if self.config.get('save_checkpoints', True):
            print("\n💾 保存初始模型 (epoch 0)...")
            
            # 先进行一次验证，获取初始性能
            print("📊 评估初始模型性能...")
            initial_val_loss, initial_val_metrics, _ = self.validate(0)
            
            # 保存初始checkpoint
            self.save_checkpoint(
                epoch=0, 
                val_loss=initial_val_loss, 
                val_metrics=initial_val_metrics, 
                is_best=False
            )
            
            # 额外保存为 checkpoint_epoch_0.pth
            checkpoint = {
                'epoch': 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': initial_val_loss,
                'val_metrics': initial_val_metrics,
                'best_val_accuracy': 0.0,
                'config': self.config,
                'model_info': self.model.get_model_info()
            }
            epoch0_path = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_0.pth')
            torch.save(checkpoint, epoch0_path)
            
            print(f"✅ 初始模型已保存到: checkpoint_epoch_0.pth")
            print(f"   初始验证损失: {initial_val_loss:.4f}")
            print(f"   初始验证准确率: {initial_val_metrics['count_accuracy']:.4f}")
            print(f"   初始最终准确率: {initial_val_metrics['true_final_count_accuracy']:.4f}")
            
            # 记录初始性能到训练历史
            initial_history = {
                'epoch': 0,
                'train_loss': float('inf'),
                'train_count_acc': 0.0,
                'train_final_acc': 0.0,
                'val_loss': initial_val_loss,
                'val_count_acc': initial_val_metrics['count_accuracy'],
                'val_final_acc': initial_val_metrics['final_count_accuracy'],
                'val_true_final_acc': initial_val_metrics['true_final_count_accuracy'],
                'joint_mse': initial_val_metrics['joint_mse'],
                'joint_mae': initial_val_metrics['joint_mae'],
                'learning_rate': self.config['learning_rate'],
                'epoch_time': 0.0,
                **{k: v for k, v in initial_val_metrics.items() if k.startswith('digit_')}
            }
            self.training_history.append(initial_history)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证（每个epoch都验证，与纯视觉保持一致）
            val_loss, val_metrics, is_best = self.validate(epoch)
            
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
                'train_count_acc': train_metrics['count_accuracy'],
                'train_final_acc': train_metrics['final_count_accuracy'],
                'val_loss': val_loss,
                'val_count_acc': val_metrics['count_accuracy'],
                'val_final_acc': val_metrics['final_count_accuracy'],
                'val_true_final_acc': val_metrics['true_final_count_accuracy'],
                'joint_mse': val_metrics['joint_mse'],
                'joint_mae': val_metrics['joint_mae'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                **{f'train_{k}': v for k, v in train_metrics.items() if k.startswith('digit_')},
                **{f'val_{k}': v for k, v in val_metrics.items() if k.startswith('digit_')}
            }
            self.training_history.append(history_entry)
            
            # 保存checkpoint
            if self.config.get('save_checkpoints', True):
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best)
            
            # 打印进度
            if epoch % self.config.get('print_every', 10) == 0:
                elapsed_time = time.time() - start_time
                avg_epoch_time = elapsed_time / epoch
                remaining_epochs = num_epochs - epoch
                eta = avg_epoch_time * remaining_epochs
                
                print(f"\nEpoch [{epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['count_accuracy']:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['count_accuracy']:.4f} | "
                      f"Final Acc: {val_metrics['true_final_count_accuracy']:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")
                
                # 打印部分per-digit准确率
                print("Per-digit Val Accuracy:", end=" ")
                for digit in [0, 1, 5, 10]:  # 打印几个关键数字
                    key = f'digit_{digit}_accuracy'
                    if key in val_metrics:
                        print(f"[{digit}]: {val_metrics[key]:.3f}", end=" ")
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


def run_single_experiment(model_type, seed, data_config, save_dir, total_epochs, config_overrides=None):
    """运行单个实验"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 创建保存目录
    model_name = f"{model_type}_seed_{seed}"
    experiment_dir = os.path.join(save_dir, model_name)
    log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 配置
    config = {
        'model_type': model_type,
        'seed': seed,
        'total_epochs': total_epochs,
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
        # 与纯视觉保持一致的训练参数
        'adam_betas': (0.9, 0.999),
        'weight_decay': 1e-5,
        'grad_clip_norm': 1.0,
        'scheduler_type': 'cosine',
        'normalize': True,
        'embodiment_loss_weight': 0.3,
        'attention_loss_weight': 0.1
    }
    
    # 应用配置覆盖
    if config_overrides:
        config.update(config_overrides)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"实验: {model_name}")
    print(f"保存目录: {experiment_dir}")
    print(f"{'='*60}")
    
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
    
    # 创建模型
    model = create_model(config, model_type=model_type)
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
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
    history = trainer.train(num_epochs=config['total_epochs'])
    training_time = time.time() - start_time
    
    # 返回实验结果
    final_metrics = history[-1] if history else {}
    result = {
        'model_type': model_type,
        'seed': seed,
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_val_loss': trainer.best_val_loss,
        'final_val_accuracy': final_metrics.get('val_count_acc', 0.0),
        'final_val_final_accuracy': final_metrics.get('val_final_acc', 0.0),
        'final_val_true_final_accuracy': final_metrics.get('val_true_final_acc', 0.0),
        'final_joint_mse': final_metrics.get('joint_mse', 0.0),
        'final_joint_mae': final_metrics.get('joint_mae', 0.0),
        'total_epochs': config['total_epochs'],
        'training_time_hours': training_time / 3600,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'experiment_dir': experiment_dir
    }
    
    return result


from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='AlexNet具身模型训练 - 修复版')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection')
    parser.add_argument('--train_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv')
    parser.add_argument('--val_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv')
    
    # 实验参数
    parser.add_argument('--total_epochs', type=int, default=1000,
                       help='训练总epoch数')
    parser.add_argument('--model_types', nargs='+', 
                       default=['baseline', 'alexnet_no_pretrain', 'alexnet_pretrain'],
                       help='要测试的模型类型')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[2048, 4096, 9999],  # 默认3个种子
                       help='随机种子列表')
    
    # 结果保存
    parser.add_argument('--save_dir', type=str, default='./embodied_experiments_fixed',
                       help='结果保存目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称，用于子目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f'alexnet_embodied_comparison_{timestamp}'
    save_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据配置
    data_config = {
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv
    }
    
    # 记录所有实验结果
    all_results = []
    results_file = os.path.join(save_dir, 'experiment_results.csv')
    
    print(f"🚀 开始AlexNet具身模型对比实验")
    print(f"模型类型: {args.model_types}")
    print(f"随机种子: {args.seeds}")
    print(f"总实验数: {len(args.model_types) * len(args.seeds)}")
    print(f"每个实验训练epochs: {args.total_epochs}")
    print(f"结果保存: {save_dir}")
    
    # 运行所有实验
    total_experiments = len(args.model_types) * len(args.seeds)
    current_exp = 0
    start_time = time.time()
    
    for model_type in args.model_types:
        for seed in args.seeds:
            current_exp += 1
            
            # 显示进度
            elapsed_time = time.time() - start_time
            avg_time_per_exp = elapsed_time / current_exp if current_exp > 0 else 0
            remaining_time = avg_time_per_exp * (total_experiments - current_exp)
            
            print(f"\n📊 进度: {current_exp}/{total_experiments}")
            print(f"⏱️  已用时: {elapsed_time/3600:.1f}h, 预计剩余: {remaining_time/3600:.1f}h")
            
            # 运行实验
            result = run_single_experiment(
                model_type=model_type,
                seed=seed,
                data_config=data_config,
                save_dir=save_dir,
                total_epochs=args.total_epochs
            )
            all_results.append(result)
            
            # 保存中间结果
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_file, index=False)
            print(f"💾 保存中间结果: {results_file}")
    
    # 生成最终报告
    print(f"\n📊 生成实验报告...")
    results_df = pd.DataFrame(all_results)
    
    # 计算统计摘要
    summary = results_df.groupby('model_type').agg({
        'best_val_accuracy': ['mean', 'std', 'max'],
        'final_val_true_final_accuracy': ['mean', 'std'],
        'final_joint_mse': ['mean', 'std'],
        'training_time_hours': ['mean', 'sum']
    }).round(4)
    
    # 保存摘要
    summary_file = os.path.join(save_dir, 'summary_stats.csv')
    summary.to_csv(summary_file)
    
    # 打印摘要
    print("\n📈 实验结果摘要:")
    print("="*80)
    print(summary)
    print("="*80)
    
    # 生成Markdown报告
    report_content = f"""# AlexNet具身模型实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验概述

- **任务**: 具身球数计数（多模态序列预测）
- **模型对比**: Baseline vs AlexNet (无预训练) vs AlexNet (预训练)
- **训练epochs**: {args.total_epochs}
- **随机种子**: {args.seeds}
- **序列长度**: 11帧

## 实验结果

### 准确率对比

| 模型类型 | 最佳验证准确率 (mean±std) | 最终准确率 (mean±std) | 关节MSE (mean±std) |
|---------|--------------------------|---------------------|-------------------|
"""
    
    for model_type in args.model_types:
        model_results = results_df[results_df['model_type'] == model_type]
        if len(model_results) > 0:
            best_mean = model_results['best_val_accuracy'].mean()
            best_std = model_results['best_val_accuracy'].std()
            final_mean = model_results['final_val_true_final_accuracy'].mean()
            final_std = model_results['final_val_true_final_accuracy'].std()
            joint_mean = model_results['final_joint_mse'].mean()
            joint_std = model_results['final_joint_mse'].std()
            
            type_names = {
                'baseline': 'Baseline CNN',
                'alexnet_pretrain': '预训练AlexNet',
                'alexnet_no_pretrain': '无预训练AlexNet'
            }
            display_name = type_names.get(model_type, model_type)
            
            report_content += f"| {display_name} | {best_mean:.4f}±{best_std:.4f} | "
            report_content += f"{final_mean:.4f}±{final_std:.4f} | {joint_mean:.4f}±{joint_std:.4f} |\n"
    
    report_content += f"""

### 训练效率

| 模型类型 | 平均训练时间 (小时) | 参数量 |
|---------|------------------|--------|
"""
    
    for model_type in args.model_types:
        model_results = results_df[results_df['model_type'] == model_type]
        if len(model_results) > 0:
            avg_time = model_results['training_time_hours'].mean()
            params = model_results['total_parameters'].iloc[0]
            
            type_names = {
                'baseline': 'Baseline CNN',
                'alexnet_pretrain': '预训练AlexNet',
                'alexnet_no_pretrain': '无预训练AlexNet'
            }
            display_name = type_names.get(model_type, model_type)
            
            report_content += f"| {display_name} | {avg_time:.2f} | {params:,} |\n"
    
    report_content += f"""

## 文件说明

- 详细结果: `experiment_results.csv`
- 统计摘要: `summary_stats.csv`
- TensorBoard日志: 各模型的 `tensorboard_logs/` 目录
- 模型checkpoints: 各模型的 `checkpoints/` 目录
- 训练历史: 各模型的 `training_history.csv`

## 查看TensorBoard

```bash
tensorboard --logdir {save_dir}
```

## 与纯视觉模型对比

此具身模型实验可与纯视觉模型实验结果进行对比，以评估具身信息（关节位置）对计数任务的贡献。
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