"""
批量AlexNet对比实验脚本 - 完整版
包含TensorBoard记录、per-digit accuracy、完整训练监控、模型checkpoint保存
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


class CompleteTrainer:
    """完整的训练器 - 包含TensorBoard记录和checkpoint保存"""
    
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
        
        # 优化器
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
        self.writer.add_text('Config', config_text, 0)
        
    def _create_scheduler(self):
        """创建学习率调度器"""
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
        """计算损失"""
        losses = {}
        
        # 计数分类损失
        count_logits = outputs['counts']
        target_counts = targets['labels'].long()
        
        count_loss = F.cross_entropy(
            count_logits.view(-1, 11),
            target_counts.view(-1),
            ignore_index=-1
        )
        losses['count_loss'] = count_loss
        
        # 动作回归损失
        pred_joints = outputs['joints'][:, :-1]
        target_joints = targets['joints'][:, 1:]
        motion_loss = F.mse_loss(pred_joints, target_joints)
        losses['motion_loss'] = motion_loss
        
        # 注意力正则化损失
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            batch_size, seq_len, H, W = attention_weights.shape
            attention_flat = attention_weights.view(batch_size * seq_len, -1)
            attention_entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=1).mean()
            losses['attention_loss'] = attention_entropy
        else:
            losses['attention_loss'] = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = (count_loss + 
                     self.embodiment_loss_weight * motion_loss +
                     self.attention_loss_weight * losses['attention_loss'])
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_metrics(self, outputs, targets):
        """计算指标"""
        metrics = {}
        
        # 计数分类指标
        count_logits = outputs['counts']
        pred_labels = torch.argmax(count_logits, dim=-1)
        target_counts = targets['labels'].long()
        
        # 整体准确率
        valid_mask = target_counts >= 0
        if valid_mask.sum() > 0:
            metrics['count_accuracy'] = (pred_labels[valid_mask] == target_counts[valid_mask]).float().mean().item()
        else:
            metrics['count_accuracy'] = 0.0
        
        # 最终计数准确率
        metrics['final_count_accuracy'] = (pred_labels[:, -1] == target_counts[:, -1]).float().mean().item()
        
        # 真实的最终计数准确率
        try:
            true_final_positions = self.find_true_final_positions(target_counts)
            batch_size = pred_labels.shape[0]
            
            true_final_correct = 0
            for i in range(batch_size):
                true_pos = true_final_positions[i]
                if pred_labels[i, true_pos] == target_counts[i, true_pos]:
                    true_final_correct += 1
            
            metrics['true_final_count_accuracy'] = true_final_correct / batch_size
        except:
            metrics['true_final_count_accuracy'] = metrics['final_count_accuracy']
        
        # 动作指标
        if outputs['joints'].shape[1] > 1:
            pred_joints = outputs['joints'][:, :-1]
            target_joints = targets['joints'][:, 1:]
            metrics['joint_mse'] = F.mse_loss(pred_joints, target_joints).item()
            metrics['joint_mae'] = F.l1_loss(pred_joints, target_joints).item()
        else:
            metrics['joint_mse'] = 0.0
            metrics['joint_mae'] = 0.0
        
        return metrics
    
    def find_true_final_positions(self, target_counts):
        """找到每个样本的真实final count位置"""
        batch_size = target_counts.shape[0]
        true_final_positions = []
        
        for i in range(batch_size):
            max_count = target_counts[i].max().item()
            final_pos = (target_counts[i] == max_count).nonzero(as_tuple=True)[0]
            if len(final_pos) > 0:
                true_final_positions.append(final_pos[0].item())
            else:
                true_final_positions.append(target_counts.shape[1] - 1)
        
        return torch.tensor(true_final_positions, device=target_counts.device)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_count = 0
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
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
                use_teacher_forcing=True
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
                    epoch_metrics[key] = epoch_metrics.get(key, 0) + value
            
            # 记录batch级别的损失到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Train_Loss', losses['total_loss'].item(), global_step)
            self.writer.add_scalar('Batch/Count_Loss', losses['count_loss'].item(), global_step)
            self.writer.add_scalar('Batch/Motion_Loss', losses['motion_loss'].item(), global_step)
        
        # 计算平均值
        avg_loss = total_loss / total_count
        avg_metrics = {key: value / total_count for key, value in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证 - 包含每个数字的准确率"""
        self.model.eval()
        total_loss = 0
        total_metrics = {}
        total_count = 0
        
        # 收集所有预测和真实标签
        all_pred_labels = []
        all_target_labels = []
        
        for batch in self.val_loader:
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
                use_teacher_forcing=False
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
                total_metrics[key] = total_metrics.get(key, 0) + value
            
            # 收集预测标签
            count_logits = outputs['counts']
            pred_labels = torch.argmax(count_logits, dim=-1)
            all_pred_labels.append(pred_labels.cpu())
            all_target_labels.append(sequence_data['labels'].cpu())
            
            total_count += 1
        
        # 平均指标
        avg_loss = total_loss / total_count
        avg_metrics = {key: value / total_count for key, value in total_metrics.items()}
        
        # 计算混淆矩阵和每个数字的准确率
        all_pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
        all_target_labels = torch.cat(all_target_labels, dim=0).numpy()
        
        # 只使用最终时刻的预测
        final_pred = all_pred_labels[:, -1]
        final_target = all_target_labels[:, -1]
        
        # 过滤无效标签
        valid_mask = final_target >= 0
        if valid_mask.sum() > 0:
            cm = confusion_matrix(final_target[valid_mask], final_pred[valid_mask], labels=list(range(11)))
            
            # 计算每个数字的准确率
            per_digit_accuracy = {}
            for digit in range(1, 11):  # 1到10
                digit_mask = final_target[valid_mask] == digit
                if digit_mask.sum() > 0:
                    digit_correct = (final_pred[valid_mask][digit_mask] == final_target[valid_mask][digit_mask]).sum()
                    per_digit_accuracy[f'acc_digit_{digit}'] = digit_correct / digit_mask.sum()
                else:
                    per_digit_accuracy[f'acc_digit_{digit}'] = 0.0
            
            # 添加到平均指标中
            avg_metrics.update(per_digit_accuracy)
            
            # 记录per-digit accuracy到TensorBoard
            for digit in range(1, 11):
                self.writer.add_scalar(f'Accuracy_per_Digit/Digit_{digit}', 
                                     per_digit_accuracy[f'acc_digit_{digit}'], epoch)
        else:
            cm = np.zeros((11, 11), dtype=int)
            # 如果没有有效数据，设置所有数字准确率为0
            for digit in range(1, 11):
                avg_metrics[f'acc_digit_{digit}'] = 0.0
        
        return avg_loss, avg_metrics, cm
    
    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        # 添加这一行确保使用正确的后端
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(11)),
                    yticklabels=list(range(11)))
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title(f'AlexNet Model - Confusion Matrix - Epoch {epoch}')
        return plt.gcf()

    def save_checkpoint(self, epoch, val_loss, val_metrics, checkpoint_type='regular'):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # 确定保存路径
        if checkpoint_type == 'best':
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                         f"best_{self.config['model_type']}_seed_{self.config.get('seed', 0)}.pth")
        elif checkpoint_type == 'final':
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                         f"final_{self.config['model_type']}_seed_{self.config.get('seed', 0)}.pth")
        elif checkpoint_type == 'periodic':
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                         f"epoch_{epoch}_{self.config['model_type']}_seed_{self.config.get('seed', 0)}.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                         f"{checkpoint_type}_{self.config['model_type']}_seed_{self.config.get('seed', 0)}.pth")
        
        try:
            # 先保存到临时文件
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            # 验证保存的文件
            test_checkpoint = torch.load(temp_path, map_location='cpu')
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for key in required_keys:
                if key not in test_checkpoint:
                    raise ValueError(f"Missing key in checkpoint: {key}")
            
            # 重命名为最终文件
            os.rename(temp_path, checkpoint_path)
            
            print(f"✅ Checkpoint saved: {os.path.basename(checkpoint_path)}")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    
    def log_to_tensorboard(self, epoch, train_loss, train_metrics, val_loss, val_metrics,confusion_matrix):
        """记录到TensorBoard"""
        # 损失
        self.writer.add_scalars('Loss/Total', {
            'Train': train_loss,
            'Val': val_loss
        }, epoch)
        
        self.writer.add_scalars('Loss/Count', {
            'Train': train_metrics.get('count_loss', 0),
            'Val': val_metrics.get('count_loss', 0)
        }, epoch)
        
        self.writer.add_scalars('Loss/Motion', {
            'Train': train_metrics.get('motion_loss', 0),
            'Val': val_metrics.get('motion_loss', 0)
        }, epoch)
        
        if confusion_matrix.sum() > 0:  # 确保混淆矩阵不为空
            cm_figure = self.plot_confusion_matrix(confusion_matrix, epoch)
            self.writer.add_figure('Confusion_Matrix/Final_Count', cm_figure, epoch)
            plt.close(cm_figure)

        # 准确率
        self.writer.add_scalars('Accuracy/Count', {
            'Train': train_metrics['count_accuracy'],
            'Val': val_metrics['count_accuracy']
        }, epoch)
        
        self.writer.add_scalars('Accuracy/Final_Count', {
            'Train': train_metrics['final_count_accuracy'],
            'Val': val_metrics['final_count_accuracy']
        }, epoch)
        
        self.writer.add_scalars('Accuracy/True_Final_Count', {
            'Train': train_metrics['true_final_count_accuracy'],
            'Val': val_metrics['true_final_count_accuracy']
        }, epoch)
        
        # 动作指标
        self.writer.add_scalars('Motion/MSE', {
            'Train': train_metrics['joint_mse'],
            'Val': val_metrics['joint_mse']
        }, epoch)
        
        self.writer.add_scalars('Motion/MAE', {
            'Train': train_metrics['joint_mae'],
            'Val': val_metrics['joint_mae']
        }, epoch)
        
        # 学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def train_full(self, total_epochs, validate_every=50, print_every=100, save_checkpoints=True):
        """完整训练过程 - 定期验证并记录"""
        print(f"开始训练 {total_epochs} epochs...")
        print(f"模型类型: {self.config['model_type']}")
        print(f"随机种子: {self.config.get('seed', 'N/A')}")
        print(f"Checkpoint目录: {self.checkpoint_dir}")
        
        start_time = time.time()
        final_results = None
        
        # 保存初始checkpoint
        if save_checkpoints:
            self.save_checkpoint(-1, float('inf'), {}, 'initial')
        
        for epoch in range(total_epochs):
            # 训练一个epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 需要验证损失，如果没有则跳过
                    pass
                else:
                    self.scheduler.step()
            
            # 定期验证
            if (epoch + 1) % validate_every == 0 or epoch == total_epochs - 1:
                val_loss, val_metrics, confusion_matrix = self.validate(epoch)
                
                # 记录训练历史
                history_entry = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['count_accuracy'],
                    'final_count_accuracy': val_metrics['final_count_accuracy'],
                    'timestamp': datetime.now().isoformat()
                }
                self.training_history.append(history_entry)
                
                # 记录到TensorBoard
                self.log_to_tensorboard(epoch, train_loss, train_metrics, val_loss, val_metrics,confusion_matrix)
                
                # 更新最佳模型
                is_best = False
                if val_metrics['count_accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['count_accuracy']
                    self.best_val_loss = val_loss
                    is_best = True
                    
                    # 保存最佳模型
                    if save_checkpoints:
                        self.save_checkpoint(epoch, val_loss, val_metrics, 'best')
                
                # 如果是最后一个epoch，保存最终结果和模型
                if epoch == total_epochs - 1:
                    final_results = {
                        'final_val_loss': val_loss,
                        'final_val_metrics': val_metrics,
                        'confusion_matrix': confusion_matrix,
                        'training_time_hours': (time.time() - start_time) / 3600,
                        'best_val_accuracy': self.best_val_accuracy,
                        'best_val_loss': self.best_val_loss,
                        'training_history': self.training_history
                    }
                    
                    # 保存最终模型
                    if save_checkpoints:
                        final_checkpoint_path = self.save_checkpoint(epoch, val_loss, val_metrics, 'final')
                        final_results['final_checkpoint_path'] = final_checkpoint_path
                
                # 打印验证结果
                status = "🌟 NEW BEST!" if is_best else ""
                print(f"Epoch {epoch+1}/{total_epochs} - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_metrics['count_accuracy']:.4f} - "
                      f"Final Acc: {val_metrics['final_count_accuracy']:.4f} {status}")
            
            # 定期打印训练进度
            elif (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / (epoch + 1) * (total_epochs - epoch - 1)
                print(f"Epoch {epoch+1}/{total_epochs} - Loss: {train_loss:.4f} - "
                      f"Elapsed: {elapsed/3600:.1f}h - Remaining: {remaining/3600:.1f}h")
            
            # 定期保存checkpoint (可选)
            if save_checkpoints and (epoch + 1) % (validate_every * 5) == 0:  # 每250个epoch保存一次
                self.save_checkpoint(epoch, train_loss, train_metrics, 'periodic')
        
        # 关闭TensorBoard
        self.writer.close()
        
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最终验证准确率: {final_results['final_val_metrics']['count_accuracy']:.4f}")
        print(f"训练时间: {final_results['training_time_hours']:.2f} 小时")
        
        return final_results


def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_config(model_type, seed=None):
    """创建配置"""
    config = {
        # 数据配置
        'image_mode': 'rgb',
        'batch_size': 16,
        'sequence_length': 11,
        'normalize': True,
        'num_workers': 4,
        
        # 模型配置
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
        
        # 训练配置
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'adam_betas': (0.9, 0.999),
        'grad_clip_norm': 1.0,
        
        # 损失权重
        'embodiment_loss_weight': 0.3,
        'attention_loss_weight': 0.1,
        
        # 学习率调度
        'scheduler_type': 'none',
        'scheduler_patience': 5,
        
        # 其他
        'model_type': model_type,
        'seed': seed
    }
    
    return config


def run_single_experiment(model_type, seed, data_config, results_file, log_base_dir, 
                         checkpoint_base_dir, total_epochs=1000, save_checkpoints=True):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {model_type} - 种子: {seed}")
    print(f"{'='*60}")
    
    # 设置随机种子
    set_random_seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 创建配置
        config = create_config(model_type, seed)
        config['total_epochs'] = total_epochs
        
        # 创建目录
        experiment_name = f'{model_type}_seed_{seed}'
        log_dir = os.path.join(log_base_dir, experiment_name)
        checkpoint_dir = os.path.join(checkpoint_base_dir, experiment_name)
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建数据加载器
        print("创建数据加载器...")
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
        print("创建模型...")
        model = create_model(config, model_type=model_type).to(device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 创建训练器
        trainer = CompleteTrainer(model, train_loader, val_loader, config, 
                                device, log_dir, checkpoint_dir)
        
        # 训练
        results = trainer.train_full(total_epochs, validate_every=100, 
                                   print_every=100, save_checkpoints=save_checkpoints)
        
        # 整理结果
        final_results = {
            'model_type': model_type,
            'seed': seed,
            'total_epochs': total_epochs,
            'final_val_accuracy': results['final_val_metrics']['count_accuracy'],
            'final_count_accuracy': results['final_val_metrics']['final_count_accuracy'],
            'true_final_count_accuracy': results['final_val_metrics']['true_final_count_accuracy'],
            'joint_mse': results['final_val_metrics']['joint_mse'],
            'joint_mae': results['final_val_metrics']['joint_mae'],
            'val_loss': results['final_val_loss'],
            'best_val_accuracy': results['best_val_accuracy'],
            'best_val_loss': results['best_val_loss'],
            'training_time_hours': results['training_time_hours'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'tensorboard_log': log_dir,
            'checkpoint_dir': checkpoint_dir,
            'final_checkpoint_path': results.get('final_checkpoint_path', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加每个数字的准确率
        for digit in range(1, 11):
            acc_key = f'acc_digit_{digit}'
            final_results[acc_key] = results['final_val_metrics'].get(acc_key, 0.0)
        
        # 保存结果到CSV
        save_result_to_csv(final_results, results_file)
        
        # 保存详细训练历史
        history_file = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(results['training_history'], f, indent=2)
        
        print(f"实验完成!")
        print(f"最终验证准确率: {final_results['final_val_accuracy']:.4f}")
        print(f"最佳验证准确率: {final_results['best_val_accuracy']:.4f}")
        print(f"训练时间: {final_results['training_time_hours']:.2f} 小时")
        print(f"TensorBoard日志: {log_dir}")
        print(f"模型checkpoints: {checkpoint_dir}")
        
        return final_results
        
    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 记录失败的实验
        failure_result = {
            'model_type': model_type,
            'seed': seed,
            'total_epochs': total_epochs,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'FAILED'
        }
        
        return failure_result
        
    finally:
        # 清理内存
        if 'model' in locals():
            del model
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def save_result_to_csv(result, csv_file):
    """保存结果到CSV文件"""
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result)


def generate_summary_stats(results_file, summary_file):
    """生成统计摘要"""
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    
    # 过滤掉失败的实验 - 修复pandas语法
    if 'status' in df.columns:
        successful_df = df[df['status'] != 'FAILED'].copy()
    else:
        # 如果没有status列，假设都是成功的
        successful_df = df.copy()
    
    if len(successful_df) == 0:
        print("没有成功的实验结果")
        return
    
    # 按模型类型分组统计
    summary_stats = []
    
    for model_type in successful_df['model_type'].unique():
        model_data = successful_df[successful_df['model_type'] == model_type]
        
        if len(model_data) > 0:
            stats = {
                'model_type': model_type,
                'n_experiments': len(model_data),
                'mean_final_val_accuracy': model_data['final_val_accuracy'].mean(),
                'std_final_val_accuracy': model_data['final_val_accuracy'].std(),
                'max_final_val_accuracy': model_data['final_val_accuracy'].max(),
                'min_final_val_accuracy': model_data['final_val_accuracy'].min(),
                'mean_best_val_accuracy': model_data['best_val_accuracy'].mean(),
                'std_best_val_accuracy': model_data['best_val_accuracy'].std(),
                'mean_training_time_hours': model_data['training_time_hours'].mean(),
                'std_training_time_hours': model_data['training_time_hours'].std(),
                'total_parameters': model_data['total_parameters'].iloc[0] if 'total_parameters' in model_data.columns else None,
                'trainable_parameters': model_data['trainable_parameters'].iloc[0] if 'trainable_parameters' in model_data.columns else None
            }
            
            # 添加每个数字的平均准确率
            for digit in range(1, 11):
                acc_col = f'acc_digit_{digit}'
                if acc_col in model_data.columns:
                    stats[f'mean_{acc_col}'] = model_data[acc_col].mean()
                    stats[f'std_{acc_col}'] = model_data[acc_col].std()
            
            # 添加关节预测指标
            if 'joint_mse' in model_data.columns:
                stats['mean_joint_mse'] = model_data['joint_mse'].mean()
                stats['std_joint_mse'] = model_data['joint_mse'].std()
            
            if 'joint_mae' in model_data.columns:
                stats['mean_joint_mae'] = model_data['joint_mae'].mean()
                stats['std_joint_mae'] = model_data['joint_mae'].std()
            
            summary_stats.append(stats)
    
    # 保存摘要
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_file, index=False)
    
    # 打印摘要表格
    print(f"\n📊 实验统计摘要:")
    print("="*80)
    display_cols = ['model_type', 'n_experiments', 'mean_final_val_accuracy', 
                   'std_final_val_accuracy', 'mean_best_val_accuracy', 'mean_training_time_hours']
    display_df = summary_df[display_cols].round(4)
    print(display_df.to_string(index=False))
    
    # 打印失败的实验
    if 'status' in df.columns:
        failed_df = df[df['status'] == 'FAILED']
        if len(failed_df) > 0:
            print(f"\n❌ 失败的实验 ({len(failed_df)} 个):")
            for _, row in failed_df.iterrows():
                print(f"  {row['model_type']} - 种子 {row['seed']}: {row.get('error', 'Unknown error')}")
    else:
        print("\n✅ 所有实验都成功完成")
    
    print(f"\n📄 完整摘要已保存: {summary_file}")
    return summary_df


def create_experiment_report(results_file, summary_file, report_file):
    """创建详细的实验报告"""
    if not os.path.exists(results_file) or not os.path.exists(summary_file):
        print("缺少必要的结果文件，无法生成报告")
        return
    
    df = pd.read_csv(results_file)
    summary_df = pd.read_csv(summary_file)
    
    report_content = f"""# AlexNet对比实验报告

## 实验概述
- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总实验数: {len(df)}
- 成功实验数: {len(df[df.get('status', 'SUCCESS') != 'FAILED'])}
- 失败实验数: {len(df[df.get('status', 'SUCCESS') == 'FAILED'])}

## 模型对比

### 性能总结
"""
    
    for _, row in summary_df.iterrows():
        report_content += f"""
#### {row['model_type']}
- 实验次数: {row['n_experiments']}
- 最终验证准确率: {row['mean_final_val_accuracy']:.4f} ± {row['std_final_val_accuracy']:.4f}
- 最佳验证准确率: {row['mean_best_val_accuracy']:.4f} ± {row['std_best_val_accuracy']:.4f}
- 平均训练时间: {row['mean_training_time_hours']:.2f} ± {row['std_training_time_hours']:.2f} 小时
- 参数数量: {row.get('total_parameters', 'N/A'):,} (可训练: {row.get('trainable_parameters', 'N/A'):,})
"""

    # 添加每个数字的准确率分析
    report_content += "\n### 各数字准确率分析\n"
    
    for model_type in summary_df['model_type']:
        model_row = summary_df[summary_df['model_type'] == model_type].iloc[0]
        report_content += f"\n#### {model_type}\n"
        
        for digit in range(1, 11):
            mean_key = f'mean_acc_digit_{digit}'
            std_key = f'std_acc_digit_{digit}'
            if mean_key in model_row and pd.notna(model_row[mean_key]):
                report_content += f"- 数字 {digit}: {model_row[mean_key]:.3f} ± {model_row[std_key]:.3f}\n"
    
    # 添加建议和结论
    report_content += f"""

## 实验结论

### 主要发现
1. **性能排序**: 根据最终验证准确率进行排序
2. **训练效率**: 训练时间与性能的权衡
3. **稳定性**: 不同随机种子下的性能方差

### 最佳模型
"""
    
    if len(summary_df) > 0:
        best_model = summary_df.loc[summary_df['mean_final_val_accuracy'].idxmax()]
        report_content += f"""
- 模型类型: {best_model['model_type']}
- 最终准确率: {best_model['mean_final_val_accuracy']:.4f} ± {best_model['std_final_val_accuracy']:.4f}
- 最佳准确率: {best_model['mean_best_val_accuracy']:.4f} ± {best_model['std_best_val_accuracy']:.4f}
- 平均训练时间: {best_model['mean_training_time_hours']:.2f} 小时
"""

    report_content += f"""

## 技术细节

### 实验设置
- 训练epochs: {df['total_epochs'].iloc[0] if 'total_epochs' in df.columns else 'N/A'}
- 批次大小: 16
- 学习率: 1e-4
- 优化器: Adam
- 随机种子: {list(df['seed'].unique()) if 'seed' in df.columns else 'N/A'}

### 数据集信息
- 序列长度: 11
- 图像模式: RGB
- 数据归一化: 是

### 评估指标
- 最终验证准确率: 模型在验证集上的最终准确率
- 最佳验证准确率: 训练过程中达到的最佳验证准确率
- 各数字准确率: 每个数字类别的准确率
- 关节预测误差: MSE和MAE

## 文件说明
- 详细结果: {os.path.basename(results_file)}
- 统计摘要: {os.path.basename(summary_file)}
- TensorBoard日志: 各模型的tensorboard_logs目录
- 模型checkpoints: 各模型的checkpoints目录

## 复现说明
使用相同的随机种子和超参数设置可以复现实验结果。每个实验的具体配置保存在对应的checkpoint目录中。
"""

    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📋 详细实验报告已保存: {report_file}")


def load_and_test_checkpoint(checkpoint_path, data_config, device='cuda'):
    """加载checkpoint并进行测试"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint文件不存在: {checkpoint_path}")
        return None
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # 创建模型
        model = create_model(config, model_type=config['model_type'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   模型类型: {config['model_type']}")
        print(f"   训练epoch: {checkpoint['epoch']}")
        print(f"   验证准确率: {checkpoint['val_metrics']['count_accuracy']:.4f}")
        
        return model, config, checkpoint
        
    except Exception as e:
        print(f"❌ 加载checkpoint失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='AlexNet对比实验 - 完整版')
    
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
                       default=[42, 123, 456, 789, 1024, 2048, 3071, 4096, 5555, 9999],
                       help='随机种子列表')
    
    # 结果保存
    parser.add_argument('--results_dir', type=str, default='./alexnet_experiment_results',
                       help='结果保存目录')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='是否保存模型checkpoints')
    parser.add_argument('--no_checkpoints', action='store_true', default=False,
                       help='不保存模型checkpoints')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='experiment',
                       choices=['experiment', 'analyze', 'test_checkpoint'],
                       help='运行模式')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='用于测试的checkpoint路径')
    
    args = parser.parse_args()
    
    # 处理checkpoint保存选项
    save_checkpoints = args.save_checkpoints and not args.no_checkpoints
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.results_dir, f'detailed_results_{timestamp}.csv')
    summary_file = os.path.join(args.results_dir, f'summary_stats_{timestamp}.csv')
    report_file = os.path.join(args.results_dir, f'experiment_report_{timestamp}.md')
    log_base_dir = os.path.join(args.results_dir, f'tensorboard_logs_{timestamp}')
    checkpoint_base_dir = os.path.join(args.results_dir, f'checkpoints_{timestamp}')
    
    # 数据配置
    data_config = {
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv
    }
    
    if args.mode == 'experiment':
        # 运行实验模式
        print("🚀 开始AlexNet对比实验 - 完整版")
        print(f"模型类型: {args.model_types}")
        print(f"随机种子: {args.seeds}")
        print(f"总实验数: {len(args.model_types) * len(args.seeds)}")
        print(f"每个实验训练epochs: {args.total_epochs}")
        print(f"保存checkpoints: {'是' if save_checkpoints else '否'}")
        print(f"结果保存: {results_file}")
        print(f"TensorBoard日志: {log_base_dir}")
        if save_checkpoints:
            print(f"模型checkpoints: {checkpoint_base_dir}")
        
        # 运行所有实验
        total_experiments = len(args.model_types) * len(args.seeds)
        current_exp = 0
        start_time = time.time()
        
        for model_type in args.model_types:
            for seed in args.seeds:
                current_exp += 1
                elapsed_time = time.time() - start_time
                avg_time_per_exp = elapsed_time / current_exp if current_exp > 0 else 0
                remaining_time = avg_time_per_exp * (total_experiments - current_exp)
                
                print(f"\n📊 进度: {current_exp}/{total_experiments}")
                print(f"⏱️ 已用时: {elapsed_time/3600:.1f}h, 预计剩余: {remaining_time/3600:.1f}h")
                
                result = run_single_experiment(
                    model_type, seed, data_config, results_file, 
                    log_base_dir, checkpoint_base_dir, args.total_epochs, save_checkpoints
                )
        
        # 生成统计摘要和报告
        print(f"\n📈 生成统计摘要...")
        summary_df = generate_summary_stats(results_file, summary_file)
        
        print(f"\n📋 生成实验报告...")
        create_experiment_report(results_file, summary_file, report_file)
        
        total_time = time.time() - start_time
        print(f"\n🎉 所有实验完成!")
        print(f"⏱️ 总耗时: {total_time/3600:.1f} 小时")
        print(f"📊 详细结果: {results_file}")
        print(f"📈 统计摘要: {summary_file}")
        print(f"📋 实验报告: {report_file}")
        print(f"📺 TensorBoard: tensorboard --logdir {log_base_dir}")
        if save_checkpoints:
            print(f"💾 模型checkpoints: {checkpoint_base_dir}")
    
    elif args.mode == 'analyze':
        # 分析现有结果
        print("📊 分析模式 - 重新生成统计摘要")
        
        # 查找最新的结果文件
        result_files = [f for f in os.listdir(args.results_dir) if f.startswith('detailed_results_') and f.endswith('.csv')]
        if not result_files:
            print("未找到实验结果文件")
            return
        
        latest_result_file = os.path.join(args.results_dir, sorted(result_files)[-1])
        print(f"使用结果文件: {latest_result_file}")
        
        summary_df = generate_summary_stats(latest_result_file, summary_file)
        create_experiment_report(latest_result_file, summary_file, report_file)
        
        print(f"📈 统计摘要: {summary_file}")
        print(f"📋 实验报告: {report_file}")
    
    elif args.mode == 'test_checkpoint':
        # 测试checkpoint模式
        if not args.checkpoint_path:
            print("测试模式需要指定 --checkpoint_path")
            return
        
        print("🧪 测试checkpoint模式")
        result = load_and_test_checkpoint(args.checkpoint_path, data_config)
        
        if result:
            model, config, checkpoint = result
            print("✅ Checkpoint测试成功")
            print(f"可以使用此模型进行推理或进一步分析")
        else:
            print("❌ Checkpoint测试失败")


if __name__ == "__main__":
    main()