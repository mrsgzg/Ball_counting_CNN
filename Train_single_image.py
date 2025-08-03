"""
训练模块 - 单图像分类模型训练 (修复版 - 解决初始模型保存问题)
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

from Model_single_image import create_single_image_model, convert_labels_for_loss, convert_predictions_to_labels
from DataLoader_single_image import get_single_image_data_loaders


class SingleImageTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 确定图像模式
        self.image_mode = config.get('image_mode', 'rgb')
        
        # 初始化模型
        self.model = create_single_image_model(config).to(self.device)
        
        # 验证模型初始化是否正确
        self._validate_model_initialization()
        
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
            normalize_images=True
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
        print(f"  Use attention: {config.get('use_attention', True)}")
        print(f"  Label mapping: 1-10 -> 0-9 (for loss calculation)")

    def _validate_model_initialization(self):
        """验证模型初始化是否正确"""
        try:
            # 创建一个虚拟输入来测试模型
            if self.image_mode == 'rgb':
                dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
            else:  # grayscale
                dummy_input = torch.randn(2, 1, 224, 224).to(self.device)
            
            # 测试前向传播
            self.model.eval()
            with torch.no_grad():
                output = self.model(dummy_input)
            
            # 验证输出形状
            expected_shape = (2, 10)  # batch_size=2, num_classes=10
            if output.shape != expected_shape:
                raise ValueError(f"Model output shape {output.shape} != expected {expected_shape}")
            
            # 验证输出值是否合理（不应该全是NaN或Inf）
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("Model output contains NaN or Inf values")
            
            print("✓ Model initialization validation passed")
            
        except Exception as e:
            print(f"✗ Model initialization validation failed: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

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
    
    def compute_metrics(self, logits, labels):
        """
        计算评估指标
        
        Args:
            logits: [batch, 10] - 模型输出的logits
            labels: [batch] - 原始标签 (1-10)
        """
        # 获取预测的类别索引 (0-9)
        pred_indices = torch.argmax(logits, dim=-1)
        # 转换为预测标签 (1-10)
        predictions = convert_predictions_to_labels(pred_indices)
        
        # 转换为numpy数组
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 基础指标
        accuracy = accuracy_score(labels_np, predictions_np)
        
        # 每个类别的准确率
        class_accuracies = {}
        for class_id in range(1, 11):
            class_mask = labels_np == class_id
            if class_mask.sum() > 0:
                class_acc = (predictions_np[class_mask] == labels_np[class_mask]).mean()
                class_accuracies[class_id] = class_acc
            else:
                class_accuracies[class_id] = 0.0
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'predictions': predictions_np,
            'targets': labels_np
        }

    def save_checkpoint(self, epoch, val_loss=None, val_accuracy=None, is_best=False, checkpoint_type='periodic'):
        """保存检查点 - 修复版本"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy,
                'config': self.config,
                'image_mode': self.image_mode,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'save_timestamp': datetime.now().isoformat()
            }
            
            # 确定保存路径
            if checkpoint_type == 'best':
                checkpoint_path = os.path.join(self.config['save_dir'], 'best_single_image_model.pth')
            elif checkpoint_type == 'final':
                checkpoint_path = os.path.join(self.config['save_dir'], 'final_single_image_model.pth')
            elif checkpoint_type == 'initial':
                checkpoint_path = os.path.join(self.config['save_dir'], 'initial_single_image_model.pth')
            else:
                checkpoint_path = os.path.join(self.config['save_dir'], f'single_image_checkpoint_epoch_{epoch}.pth')
            
            # 先保存到临时文件，然后重命名（原子操作）
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            # 验证保存的文件是否可以加载
            try:
                test_checkpoint = torch.load(temp_path, map_location='cpu')
                # 基本验证
                required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'config']
                for key in required_keys:
                    if key not in test_checkpoint:
                        raise ValueError(f"Missing key in checkpoint: {key}")
                
                # 验证模型状态字典
                if not isinstance(test_checkpoint['model_state_dict'], dict):
                    raise ValueError("Invalid model_state_dict")
                
            except Exception as e:
                os.remove(temp_path)  # 删除损坏的文件
                raise RuntimeError(f"Saved checkpoint is corrupted: {e}")
            
            # 重命名为最终文件
            os.rename(temp_path, checkpoint_path)
            
            print(f"✓ Checkpoint saved and validated: {checkpoint_path}")
            
            # 如果是最佳模型，额外记录信息
            if is_best and val_accuracy is not None:
                print(f"  → New best model! Validation accuracy: {val_accuracy:.4f}")
            
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            raise

    def save_initial_checkpoint(self):
        """保存初始模型状态"""
        print("Saving initial model state...")
        self.save_checkpoint(
            epoch=-1,  # 使用-1表示初始状态
            val_loss=None,
            val_accuracy=None,
            checkpoint_type='initial'
        )

    def test_model_save_load(self):
        """测试模型的保存和加载功能"""
        print("Testing model save/load functionality...")
        
        try:
            # 保存当前状态
            test_path = os.path.join(self.config['save_dir'], 'test_model.pth')
            
            # 记录原始参数
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.clone()
            
            # 保存模型
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'test': True
            }
            torch.save(checkpoint, test_path)
            
            # 修改模型参数
            with torch.no_grad():
                for param in self.model.parameters():
                    param.fill_(0.0)
            
            # 加载模型
            checkpoint = torch.load(test_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 验证参数是否恢复
            all_match = True
            for name, param in self.model.named_parameters():
                if not torch.allclose(param, original_params[name]):
                    all_match = False
                    print(f"Parameter mismatch: {name}")
                    break
            
            # 清理测试文件
            os.remove(test_path)
            
            if all_match:
                print("✓ Model save/load test passed")
                return True
            else:
                print("✗ Model save/load test failed")
                return False
                
        except Exception as e:
            print(f"✗ Model save/load test failed with exception: {e}")
            return False

    def load_checkpoint(self, checkpoint_path):
        """加载检查点 - 添加验证"""
        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 验证检查点内容
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Missing key in checkpoint: {key}")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            
            print(f"✓ Checkpoint loaded successfully")
            print(f"  Resuming from epoch {self.start_epoch}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Best val accuracy: {self.best_val_accuracy:.4f}")
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            raise

    # [其余方法保持不变...]
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        all_logits = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)  # 原始标签 1-10
            
            # 转换标签用于计算loss (1-10 -> 0-9)
            class_indices = convert_labels_for_loss(labels)
            
            # 前向传播
            logits = self.model(images)  # [batch, 10]
            loss = self.criterion(logits, class_indices)  # 使用0-9的索引计算loss
            
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
            
            # 收集预测结果 (用原始标签1-10计算指标)
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            
            # 打印进度
            if batch_idx % self.config.get('print_freq', 100) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} LR: {current_lr:.6f}')
        
        # 计算平均损失和指标
        avg_loss = total_loss / total_samples
        
        # 合并所有batch的结果
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self.compute_metrics(all_logits, all_labels)
        
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        all_logits = []
        all_labels = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)  # 原始标签 1-10
            
            # 转换标签用于计算loss (1-10 -> 0-9)
            class_indices = convert_labels_for_loss(labels)
            
            # 前向传播
            logits = self.model(images)  # [batch, 10]
            loss = self.criterion(logits, class_indices)  # 使用0-9的索引计算loss
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集结果 (用原始标签1-10计算指标)
            all_logits.append(logits)
            all_labels.append(labels)
        
        # 计算指标
        avg_loss = total_loss / total_samples
        
        # 合并所有batch的结果
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self.compute_metrics(all_logits, all_labels)
        
        # 计算混淆矩阵 (使用1-10标签)
        cm = confusion_matrix(
            metrics['targets'], 
            metrics['predictions'], 
            labels=list(range(1, 11))
        )
        
        return avg_loss, metrics, cm

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
        """主训练循环 - 修复版本"""
        print(f"\n开始训练单图像分类模型")
        print(f"总计 {self.config['total_epochs']} 个epoch")
        print(f"设备: {self.device}")
        print(f"图像模式: {self.image_mode}")
        
        # 测试模型保存加载功能
        if not self.test_model_save_load():
            print("⚠️  Model save/load test failed, but continuing training...")
        
        # 保存初始模型状态
        self.save_initial_checkpoint()
        
        # 进行初始验证以获得基线
        print("\nPerforming initial validation...")
        initial_val_loss, initial_val_metrics, initial_cm = self.validate(-1)
        print(f"Initial validation - Loss: {initial_val_loss:.4f}, Accuracy: {initial_val_metrics['accuracy']:.4f}")
        
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
                if class_id % 5 == 0:  # 每5个换行
                    print()
            if 10 % 5 != 0:  # 如果最后一行没满，补充换行
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
        final_epoch = self.config['total_epochs'] - 1
        self.save_checkpoint(
            final_epoch, 
            val_loss, val_metrics['accuracy'], 
            checkpoint_type='final'
        )
        
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 保存最终的混淆矩阵
        final_cm_path = os.path.join(self.config['save_dir'], 'final_confusion_matrix.png')
        self.plot_confusion_matrix(confusion_matrix, final_epoch, final_cm_path)
        print(f"最终混淆矩阵保存到: {final_cm_path}")
        
        # 保存详细分类报告
        report = classification_report(
            val_metrics['targets'], 
            val_metrics['predictions'], 
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
                val_metrics['targets'], 
                val_metrics['predictions'], 
                target_names=[f'Count_{i}' for i in range(1,11)]
            ))
        
        print(f"详细报告保存到: {report_path}")
        
        self.writer.close()


def create_single_image_trainer(config):
    """创建单图像分类训练器"""
    return SingleImageTrainer(config)