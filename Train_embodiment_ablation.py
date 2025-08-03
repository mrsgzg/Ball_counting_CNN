"""
训练模块 - 具身计数模型消融实验训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from Model_embodiment_ablation import create_ablation_model
from DataLoader_embodiment import get_ball_counting_data_loaders


class AblationTrainer:
    """消融实验训练器"""
    
    def __init__(self, config, model_type):
        """
        初始化消融实验训练器
        
        Args:
            config: 训练配置
            model_type: 模型类型 ('counting_only' 或 'visual_only')
        """
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config['device'])
        
        # 根据模型类型设置训练策略
        if model_type == 'counting_only':
            self.model_name = "EmbodiedCountingOnly"
            self.has_embodiment = True
            self.has_motion_decoder = False
        elif model_type == 'visual_only':
            self.model_name = "VisualOnlyCountingModel"
            self.has_embodiment = False
            self.has_motion_decoder = False
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 确定图像模式
        self.image_mode = config.get('image_mode', 'rgb')
        input_channels = 3 if self.image_mode == 'rgb' else 1
        
        # 初始化模型
        self.model = create_ablation_model(model_type, config).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=config['adam_betas'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.normalizer = get_ball_counting_data_loaders(
            train_csv_path=config['train_csv'],
            val_csv_path=config['val_csv'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            normalize=config['normalize'],
            num_workers=config['num_workers'],
            image_mode=self.image_mode,
            normalize_images=True
        )
        
        # 初始化TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(config['log_dir'], f'{model_type}_run_{timestamp}')
        self.writer = SummaryWriter(log_dir)
        
        # 保存配置到TensorBoard
        config_text = f'Model Type: {model_type}\n'
        config_text += f'Model Name: {self.model_name}\n'
        config_text += f'Has Embodiment: {self.has_embodiment}\n'
        config_text += f'Has Motion Decoder: {self.has_motion_decoder}\n\n'
        config_text += '\n'.join([f'{k}: {v}' for k, v in config.items() if k != 'model_config'])
        config_text += '\nModel Config:\n' + '\n'.join([f'  {k}: {v}' for k, v in config['model_config'].items()])
        self.writer.add_text('Config', config_text, 0)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        print(f"AblationTrainer initialized:")
        print(f"  Model Type: {model_type}")
        print(f"  Model Name: {self.model_name}")
        print(f"  Has Embodiment: {self.has_embodiment}")
        print(f"  Has Motion Decoder: {self.has_motion_decoder}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.train_loader.dataset):,}")
        print(f"  Validation samples: {len(self.val_loader.dataset):,}")
        print(f"  Image mode: {self.image_mode.upper()}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config['scheduler_type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['total_epochs']
            )
        elif self.config['scheduler_type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['scheduler_patience']
            )
        elif self.config['scheduler_type'] == 'none':
            return None
    
    def compute_loss(self, outputs, targets):
        """
        计算损失 - 只包含计数损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 计数分类损失
        count_logits = outputs['counts']  # [batch, seq_len, 11]
        target_counts = targets['labels'].long()  # [batch, seq_len]
        
        # 计算序列损失
        count_loss = F.cross_entropy(
            count_logits.view(-1, 11),
            target_counts.view(-1),
            ignore_index=-1
        )
        losses['count_loss'] = count_loss
        
        # 总损失就是计数损失
        losses['total_loss'] = count_loss
        
        return losses
    
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
    
    def compute_metrics(self, outputs, targets):
        """计算指标"""
        metrics = {}
        
        # 计数分类指标
        count_logits = outputs['counts']  # [batch, seq_len, 11]
        pred_labels = torch.argmax(count_logits, dim=-1)  # [batch, seq_len]
        target_counts = targets['labels'].long()  # [batch, seq_len]
        
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
        
        # 消融实验没有关节预测，所以不计算关节指标
        metrics['joint_mse'] = 0.0
        metrics['joint_mae'] = 0.0
        
        return metrics
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {}
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据准备
            sequence_data = {
                'images': batch['sequence_data']['images'].to(self.device),
                'joints': batch['sequence_data']['joints'].to(self.device),
                'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                'labels': batch['sequence_data']['labels'].to(self.device)
            }
            
            # 前向传播
            outputs = self.model(sequence_data=sequence_data)
            
            # 计算损失
            targets = {
                'labels': sequence_data['labels']
            }
            losses = self.compute_loss(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            
            self.optimizer.step()
            
            # 累积损失和指标
            total_loss += losses['total_loss'].item()
            
            # 计算指标
            with torch.no_grad():
                metrics = self.compute_metrics(outputs, targets)
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0) + value
            
            batch_count += 1
            
            # 打印进度
            if batch_idx % self.config['print_freq'] == 0:
                print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {losses["total_loss"].item():.4f} '
                      f'Count Acc: {metrics["count_accuracy"]:.4f}')
        
        # 平均指标
        avg_loss = total_loss / batch_count
        avg_metrics = {key: value / batch_count for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_metrics = {}
        batch_count = 0
        
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
            outputs = self.model(sequence_data=sequence_data)
            
            # 计算损失
            targets = {
                'labels': sequence_data['labels']
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
            
            batch_count += 1
        
        # 平均指标
        avg_loss = total_loss / batch_count
        avg_metrics = {key: value / batch_count for key, value in total_metrics.items()}
        
        # 计算混淆矩阵
        all_pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
        all_target_labels = torch.cat(all_target_labels, dim=0).numpy()
        
        # 只使用最终时刻的预测计算混淆矩阵
        final_pred = all_pred_labels[:, -1]
        final_target = all_target_labels[:, -1]
        
        # 过滤无效标签
        valid_mask = final_target >= 0
        if valid_mask.sum() > 0:
            cm = confusion_matrix(final_target[valid_mask], final_pred[valid_mask], labels=list(range(11)))
        else:
            cm = np.zeros((11, 11), dtype=int)
        
        return avg_loss, avg_metrics, cm
    
    def save_checkpoint(self, epoch, val_loss, val_accuracy, is_best=False, checkpoint_type='periodic'):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'normalizer_stats': self.normalizer.stats if hasattr(self.normalizer, 'stats') else None,
            'image_mode': self.image_mode
        }
        
        # 保存路径
        if checkpoint_type == 'best':
            checkpoint_path = os.path.join(self.config['save_dir'], f'best_{self.model_type}_model.pth')
        elif checkpoint_type == 'init':
            checkpoint_path = os.path.join(self.config['save_dir'], f'initial_{self.model_type}_model.pth')
        elif checkpoint_type == 'interrupted':
            checkpoint_path = os.path.join(self.config['save_dir'], f'interrupted_{self.model_type}_model.pth')
        else:
            checkpoint_path = os.path.join(self.config['save_dir'], f'{self.model_type}_checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 保存模型配置
        config_path = os.path.join(self.config['save_dir'], f'{self.model_type}_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        # 验证模型类型
        if checkpoint.get('model_type') != self.model_type:
            print(f"Warning: 检查点模型类型 {checkpoint.get('model_type')} 与当前模型类型 {self.model_type} 不匹配")
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Model type: {checkpoint.get('model_type', 'unknown')}")
        print(f"Resuming from epoch {self.start_epoch}")
    
    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(11)),
                    yticklabels=list(range(11)))
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title(f'{self.model_name} - Count Confusion Matrix - Epoch {epoch}')
        return plt.gcf()
    
    def log_to_tensorboard(self, epoch, train_loss, train_metrics, val_loss, val_metrics, confusion_matrix):
        """记录到TensorBoard"""
        # 损失
        self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Val': val_loss
        }, epoch)
        
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
        
        # 模型类型信息
        self.writer.add_text('Model_Info', f'Type: {self.model_type}, Name: {self.model_name}', epoch)
        
        # 学习率
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # 混淆矩阵
        if confusion_matrix.sum() > 0:
            cm_figure = self.plot_confusion_matrix(confusion_matrix, epoch)
            self.writer.add_figure('Confusion_Matrix/Final_Count', cm_figure, epoch)
            plt.close(cm_figure)
        
        # 每个数字的准确率
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_norm = confusion_matrix.astype('float') / row_sums
        
        for digit in range(11):
            if confusion_matrix.sum(axis=1)[digit] > 0:
                accuracy = cm_norm[digit, digit]
                self.writer.add_scalar(f'Accuracy_per_Digit/Digit_{digit}', accuracy, epoch)
    
    def train(self):
        """主训练循环"""
        print(f"\n开始训练 {self.model_name}")
        print(f"总计 {self.config['total_epochs']} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.model_type}")
        
        # 初始验证
        print(f"进行初始验证...")
        val_loss_init, val_metrics_init, confusion_matrix_init = self.validate(0)
        self.save_checkpoint(0, val_loss_init, val_metrics_init['count_accuracy'], 
                           checkpoint_type='init')
        print(f"初始模型已保存，初始验证准确率: {val_metrics_init['count_accuracy']:.4f}")
        
        # 主训练循环
        for epoch in range(self.start_epoch, self.config['total_epochs']):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_metrics = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, val_metrics, confusion_matrix = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录到TensorBoard
            self.log_to_tensorboard(epoch, train_loss, train_metrics, val_loss, 
                                   val_metrics, confusion_matrix)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch [{epoch}] - Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Train Count Acc: {train_metrics["count_accuracy"]:.4f} | '
                  f'Val Count Acc: {val_metrics["count_accuracy"]:.4f}')
            print(f'Final Count Acc: {val_metrics["final_count_accuracy"]:.4f}')
            print(f'True Final Count Acc: {val_metrics["true_final_count_accuracy"]:.4f}')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_metrics['count_accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['count_accuracy']
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_metrics['count_accuracy'], 
                                   is_best=True, checkpoint_type='best')
                print(f"新的最佳模型! 验证准确率: {self.best_val_accuracy:.4f}")
            
            # 定期保存
            if (epoch+1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_loss, val_metrics['count_accuracy'], 
                                   checkpoint_type='periodic')
        
        print(f"\n训练完成!")
        print(f"模型类型: {self.model_type}")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        self.writer.close()


def create_ablation_trainer(config, model_type):
    """创建消融实验训练器"""
    return AblationTrainer(config, model_type)