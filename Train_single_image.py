"""
训练模块 - 单图像分类模型训练
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from Model_single_image import create_single_image_model
from DataLoader_single_image import get_single_image_data_loaders


class SingleImageTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 确定图像模式
        self.image_mode = config.get('image_mode', 'rgb')
        
        # 初始化模型
        self.model = create_single_image_model(config).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=config.get('adam_betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建数据加载器
        self.train_loader, self.val_loader = get_single_image_data_loaders(
            train_csv_path=config['train_csv'],
            val_csv_path=config['val_csv'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            image_mode=self.image_mode,
            normalize_images=True,
            frame_selection=config.get('frame_selection', 'all')
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # 初始化TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(config['log_dir'], f'single_image_run_{timestamp}')
        self.writer = SummaryWriter(log_dir)
        
        # 保存配置到TensorBoard
        config_text = '\n'.join([f'{k}: {v}' for k, v in config.items() if k != 'model_config'])
        config_text += '\nModel Config:\n' + '\n'.join([f'  {k}: {v}' for k, v in config['model_config'].items()])
        self.writer.add_text('Config', config_text, 0)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        print(f"SingleImageTrainer initialized:")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.train_loader.dataset):,}")
        print(f"  Validation samples: {len(self.val_loader.dataset):,}")
        print(f"  Image mode: {self.image_mode.upper()}")
        print(f"  Frame selection: {config.get('frame_selection', 'all')}")
        print(f"  Use attention: {config.get('use_attention', True)}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['total_epochs']
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=0.5
            )
        else:
            return None
    
    def compute_metrics(self, outputs, targets):
        """计算评估指标"""
        predictions = torch.argmax(outputs, dim=-1)
        
        # 转换为numpy数组
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # 基础指标
        accuracy = accuracy_score(targets_np, predictions_np)
        
        # 每个类别的准确率
        class_accuracies = {}
        for class_id in range(1, 11):  # 假设类别从1到10
            class_mask = targets_np == class_id
            if class_mask.sum() > 0:
                class_acc = (predictions_np[class_mask] == targets_np[class_mask]).mean()
                class_accuracies[class_id] = class_acc
            else:
                class_accuracies[class_id] = 0.0
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集预测结果
            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            
            # 打印进度
            if batch_idx % self.config.get('print_freq', 100) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} LR: {current_lr:.6f}')
        
        # 计算平均损失和指标
        avg_loss = total_loss / total_samples
        metrics = self.compute_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_targets)
        )
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集结果
            predictions = torch.argmax(outputs, dim=-1)
            probabilities = F.softmax(outputs, dim=-1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())
        
        # 计算指标
        avg_loss = total_loss / total_samples
        metrics = self.compute_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_targets)
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(
            all_targets, 
            all_predictions, 
            labels=list(range(1, 11))  # 假设类别从1到10
        )
        
        return avg_loss, metrics, cm, all_predictions, all_targets
    
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
            'image_mode': self.image_mode
        }
        
        # 确定保存路径
        if checkpoint_type == 'best':
            checkpoint_path = os.path.join(self.config['save_dir'], 'best_single_image_model.pth')
        elif checkpoint_type == 'final':
            checkpoint_path = os.path.join(self.config['save_dir'], 'final_single_image_model.pth')
        else:
            checkpoint_path = os.path.join(self.config['save_dir'], f'single_image_checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
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
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")
    
    def plot_confusion_matrix(self, cm, epoch, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        # 计算归一化的混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=list(range(1,11)),
                    yticklabels=list(range(1,11)))
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title(f'Single Image Model - Confusion Matrix (Epoch {epoch})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return plt.gcf()
    
    def log_to_tensorboard(self, epoch, train_loss, train_metrics, val_loss, val_metrics, confusion_matrix):
        """记录到TensorBoard"""
        # 损失
        self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Val': val_loss
        }, epoch)
        
        # 总体准确率
        self.writer.add_scalars('Accuracy', {
            'Train': train_metrics['accuracy'],
            'Val': val_metrics['accuracy']
        }, epoch)
        
        # 每个类别的准确率
        for class_id in range(1,11):
            self.writer.add_scalars(f'Class_Accuracy/Count_{class_id}', {
                'Train': train_metrics['class_accuracies'].get(class_id, 0.0),
                'Val': val_metrics['class_accuracies'].get(class_id, 0.0)
            }, epoch)
        
        # 学习率
        if self.scheduler:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 混淆矩阵
        if confusion_matrix.sum() > 0:
            cm_figure = self.plot_confusion_matrix(confusion_matrix, epoch)
            self.writer.add_figure('Confusion_Matrix', cm_figure, epoch)
            plt.close(cm_figure)
        
        # 每个类别的样本分布（只在第一个epoch记录）
        if epoch == 0:
            train_dist = self.train_loader.dataset.get_class_distribution()
            val_dist = self.val_loader.dataset.get_class_distribution()
            
            for class_id in range(1,11):
                train_count = train_dist.get(class_id, 0)
                val_count = val_dist.get(class_id, 0)
                self.writer.add_scalars(f'Data_Distribution/Count_{class_id}', {
                    'Train': train_count,
                    'Val': val_count
                }, epoch)
    
    def train(self):
        """主训练循环"""
        print(f"\n开始训练单图像分类模型")
        print(f"总计 {self.config['total_epochs']} 个epoch")
        print(f"设备: {self.device}")
        print(f"图像模式: {self.image_mode}")
        
        for epoch in range(self.start_epoch, self.config['total_epochs']):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_metrics = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, val_metrics, confusion_matrix, val_predictions, val_targets = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录到TensorBoard
            self.log_to_tensorboard(
                epoch, train_loss, train_metrics, 
                val_loss, val_metrics, confusion_matrix
            )
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'\nEpoch [{epoch+1}/{self.config["total_epochs"]}] - Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_metrics["accuracy"]:.4f} | Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 打印每个类别的验证准确率
            print("Per-class validation accuracy:")
            for class_id in range(1,11):
                class_acc = val_metrics['class_accuracies'].get(class_id, 0.0)
                print(f"  Count {class_id}: {class_acc:.3f}", end="  ")
                if (class_id + 1) % 6 == 0:  # 每6个换行
                    print()
            print()
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    epoch, val_loss, val_metrics['accuracy'], 
                    is_best=True, checkpoint_type='best'
                )
                print(f"新的最佳模型! 验证准确率: {self.best_val_accuracy:.4f}")
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_every', 20) == 0:
                self.save_checkpoint(
                    epoch, val_loss, val_metrics['accuracy'], 
                    checkpoint_type='periodic'
                )
        
        # 保存最终模型
        self.save_checkpoint(
            self.config['total_epochs'] - 1, 
            val_loss, val_metrics['accuracy'], 
            checkpoint_type='final'
        )
        
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 保存最终的混淆矩阵
        final_cm_path = os.path.join(self.config['save_dir'], 'final_confusion_matrix.png')
        self.plot_confusion_matrix(confusion_matrix, self.config['total_epochs'] - 1, final_cm_path)
        print(f"最终混淆矩阵保存到: {final_cm_path}")
        
        # 保存详细分类报告
        report = classification_report(
            val_targets, val_predictions, 
            target_names=[f'Count_{i}' for i in range(1,11)],
            output_dict=True
        )
        
        report_path = os.path.join(self.config['save_dir'], 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Single Image Classification Model - Final Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(
                val_targets, val_predictions, 
                target_names=[f'Count_{i}' for i in range(1,11)]
            ))
        
        print(f"详细报告保存到: {report_path}")
        
        self.writer.close()


def create_single_image_trainer(config):
    """创建单图像分类训练器"""
    return SingleImageTrainer(config)