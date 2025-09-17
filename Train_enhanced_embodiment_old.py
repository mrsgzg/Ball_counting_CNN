"""
训练模块 - 增强具身计数模型训练 (Enhanced Internal Model)
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

from Model_enhanced_embodiment import EnhancedEmbodiedCountingModel
from DataLoader_embodiment import get_ball_counting_data_loaders


class EnhancedModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 设置损失权重
        self.embodiment_loss_weight = config.get('embodiment_loss_weight', 0.3)
        self.attention_loss_weight = config.get('attention_loss_weight', 0.1)
        
        # 训练阶段配置
        self.training_stages = {
            'stage_1': {
                'epochs': (0, config['stage_1_epochs']),
                'description': '预训练Internal Model组件',
                'frozen_modules': [],
                'loss_weights': {'motion': 1.0, 'count': 0.5, 'attention': 0.1}
            },
            'stage_2': {
                'epochs': (config['stage_1_epochs'], config['stage_2_epochs']),
                'description': '联合训练所有组件',
                'frozen_modules': [],
                'loss_weights': {'motion': self.embodiment_loss_weight, 'count': 1.0, 'attention': self.attention_loss_weight}
            },
            'stage_3': {
                'epochs': (config['stage_2_epochs'], config['total_epochs']),
                'description': '精调计数任务',
                'frozen_modules': ['motion_decoder'],  # 冻结motion解码器，专注计数
                'loss_weights': {'motion': 0.1, 'count': 1.0, 'attention': 0.05}
            }
        }
        
        # 确定图像模式
        self.image_mode = config.get('image_mode', 'rgb')
        input_channels = 3 if self.image_mode == 'rgb' else 1
        
        # 初始化增强模型
        model_config = config['model_config'].copy()
        model_config['input_channels'] = input_channels
        model_config['use_fovea_bias'] = config.get('use_fovea_bias', True)
        self.model = EnhancedEmbodiedCountingModel(**model_config).to(self.device)
        
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
        log_dir = os.path.join(config['log_dir'], f'enhanced_run_{timestamp}')
        self.writer = SummaryWriter(log_dir)
        
        # 保存配置到TensorBoard
        config_text = 'Enhanced Embodied Counting Model with Internal Model\n'
        config_text += f'Early Fusion: True\n'
        config_text += f'Residual Connections: True\n'
        config_text += f'Task-Guided Attention: True\n'
        config_text += f'Fovea Bias: {model_config.get("use_fovea_bias", True)}\n\n'
        config_text += '\n'.join([f'{k}: {v}' for k, v in config.items() if k != 'model_config'])
        config_text += '\nModel Config:\n' + '\n'.join([f'  {k}: {v}' for k, v in config['model_config'].items()])
        self.writer.add_text('Config', config_text, 0)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.current_stage = None
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        print(f"EnhancedModelTrainer initialized:")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.train_loader.dataset):,}")
        print(f"  Validation samples: {len(self.val_loader.dataset):,}")
        print(f"  Image mode: {self.image_mode.upper()} ({input_channels} channels)")
        print(f"  Use fovea bias: {model_config.get('use_fovea_bias', True)}")
    
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
    
    def get_current_stage(self, epoch):
        """获取当前训练阶段"""
        for stage_name, stage_config in self.training_stages.items():
            start_epoch, end_epoch = stage_config['epochs']
            if start_epoch <= epoch < end_epoch:
                return stage_name, stage_config
        return 'stage_3', self.training_stages['stage_3']
    
    def compute_loss(self, outputs, targets, stage_config):
        """计算损失 - 增强版包含attention监督"""
        losses = {}
        
        # 计数分类损失 (Inverse Model)
        if stage_config['loss_weights']['count'] > 0:
            count_logits = outputs['counts']
            target_counts = targets['labels'].long()
            
            count_loss = F.cross_entropy(
                count_logits.view(-1, 11),
                target_counts.view(-1),
                ignore_index=-1
            )
            losses['count_loss'] = count_loss
        else:
            losses['count_loss'] = torch.tensor(0.0, device=self.device)
        
        # 动作回归损失 (Forward Model)
        if stage_config['loss_weights']['motion'] > 0:
            pred_joints = outputs['joints'][:, :-1]
            target_joints = targets['joints'][:, 1:]
            motion_loss = F.mse_loss(pred_joints, target_joints)
            losses['motion_loss'] = motion_loss
        else:
            losses['motion_loss'] = torch.tensor(0.0, device=self.device)
        
        # 注意力正则化损失 (可选)
        if stage_config['loss_weights'].get('attention', 0) > 0 and 'attention_weights' in outputs:
            # 鼓励注意力集中（降低熵）
            attention_weights = outputs['attention_weights']  # [batch, seq_len, H, W]
            batch_size, seq_len, H, W = attention_weights.shape
            
            # 计算注意力熵（越小越集中）
            attention_flat = attention_weights.view(batch_size * seq_len, -1)
            attention_entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=1).mean()
            
            # 我们希望注意力集中，所以minimize entropy
            losses['attention_loss'] = attention_entropy
        else:
            losses['attention_loss'] = torch.tensor(0.0, device=self.device)
        
        # 总损失
        weights = stage_config['loss_weights']
        total_loss = (weights['count'] * losses['count_loss'] + 
                     weights['motion'] * losses['motion_loss'] +
                     weights.get('attention', 0) * losses['attention_loss'])
        losses['total_loss'] = total_loss
        
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
        
        # 注意力集中度
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            # 计算平均注意力熵（越小越集中）
            batch_size, seq_len, H, W = attention_weights.shape
            attention_flat = attention_weights.view(-1, H * W)
            attention_entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=1).mean()
            metrics['attention_entropy'] = attention_entropy.item()
        else:
            metrics['attention_entropy'] = 0.0
        
        return metrics
    
    def train_one_epoch(self, epoch, stage_config):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {}
        total_losses = {}
        batch_count = 0
        
        # Teacher forcing比例
        teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.01)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据准备
            sequence_data = {
                'images': batch['sequence_data']['images'].to(self.device),
                'joints': batch['sequence_data']['joints'].to(self.device),
                'timestamps': batch['sequence_data']['timestamps'].to(self.device),
                'labels': batch['sequence_data']['labels'].to(self.device)
            }
            
            # 前向传播
            use_tf = np.random.random() < teacher_forcing_ratio
            outputs = self.model(
                sequence_data=sequence_data,
                use_teacher_forcing=use_tf,
                return_attention=True  # 返回attention用于loss计算
            )
            
            # 计算损失
            targets = {
                'labels': sequence_data['labels'],
                'joints': sequence_data['joints']
            }
            losses = self.compute_loss(outputs, targets, stage_config)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            
            self.optimizer.step()
            
            # 累积损失和指标
            total_loss += losses['total_loss'].item()
            for loss_name, loss_value in losses.items():
                total_losses[loss_name] = total_losses.get(loss_name, 0) + loss_value.item()
            
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
                      f'Count: {losses["count_loss"].item():.4f} '
                      f'Motion: {losses["motion_loss"].item():.4f} '
                      f'Attention: {losses["attention_loss"].item():.4f} '
                      f'Count Acc: {metrics["count_accuracy"]:.4f}')
        
        # 平均指标
        avg_loss = total_loss / batch_count
        avg_metrics = {key: value / batch_count for key, value in total_metrics.items()}
        avg_losses = {key: value / batch_count for key, value in total_losses.items()}
        avg_metrics.update(avg_losses)
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch, stage_config):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_metrics = {}
        total_losses = {}
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
            losses = self.compute_loss(outputs, targets, stage_config)
            total_loss += losses['total_loss'].item()
            
            for loss_name, loss_value in losses.items():
                total_losses[loss_name] = total_losses.get(loss_name, 0) + loss_value.item()
            
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
        avg_losses = {key: value / batch_count for key, value in total_losses.items()}
        avg_metrics.update(avg_losses)
        
        # 计算混淆矩阵
        all_pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
        all_target_labels = torch.cat(all_target_labels, dim=0).numpy()
        
        # 只使用最终时刻的预测
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
            'model_info': self.model.get_model_info(),
            'normalizer_stats': self.normalizer.stats if hasattr(self.normalizer, 'stats') else None,
            'image_mode': self.image_mode
        }
        
        # 保存路径
        if checkpoint_type == 'best':
            checkpoint_path = os.path.join(self.config['save_dir'], 'best_enhanced_model.pth')
        elif checkpoint_type == 'init':
            checkpoint_path = os.path.join(self.config['save_dir'], 'initial_enhanced_model.pth')
        elif checkpoint_type == 'interrupted':
            checkpoint_path = os.path.join(self.config['save_dir'], 'interrupted_enhanced_model.pth')
        else:
            checkpoint_path = os.path.join(self.config['save_dir'], f'enhanced_checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Enhanced checkpoint saved: {checkpoint_path}")
        
        # 保存模型配置
        config_path = os.path.join(self.config['save_dir'], 'enhanced_model_config.json')
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
        
        # 加载模型信息
        if 'model_info' in checkpoint:
            print(f"Loaded model info: {checkpoint['model_info']}")
        
        print(f"Enhanced checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")
    
    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(11)),
                    yticklabels=list(range(11)))
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title(f'Enhanced Model - Count Confusion Matrix - Epoch {epoch}')
        return plt.gcf()
    
    def log_to_tensorboard(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                          confusion_matrix, stage_config):
        """记录到TensorBoard - 增强版"""
        # 损失
        self.writer.add_scalars('Loss/Total', {
            'Train': train_loss,
            'Val': val_loss
        }, epoch)
        
        # 分别记录各项损失
        self.writer.add_scalars('Loss/Count', {
            'Train': train_metrics['count_loss'],
            'Val': val_metrics['count_loss']
        }, epoch)
        
        self.writer.add_scalars('Loss/Motion', {
            'Train': train_metrics['motion_loss'],
            'Val': val_metrics['motion_loss']
        }, epoch)
        
        self.writer.add_scalars('Loss/Attention', {
            'Train': train_metrics['attention_loss'],
            'Val': val_metrics['attention_loss']
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
        
        # 动作指标
        self.writer.add_scalars('Motion/MSE', {
            'Train': train_metrics['joint_mse'],
            'Val': val_metrics['joint_mse']
        }, epoch)
        
        # 注意力指标
        self.writer.add_scalars('Attention/Entropy', {
            'Train': train_metrics['attention_entropy'],
            'Val': val_metrics['attention_entropy']
        }, epoch)
        
        # 训练阶段信息
        self.writer.add_text('Training_Stage', stage_config['description'], epoch)
        self.writer.add_scalars('Loss_Weights', stage_config['loss_weights'], epoch)
        
        # 学习率
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # 混淆矩阵
        if confusion_matrix.sum() > 0:
            cm_figure = self.plot_confusion_matrix(confusion_matrix, epoch)
            self.writer.add_figure('Confusion_Matrix/Final_Count', cm_figure, epoch)
            plt.close(cm_figure)
    
    def train(self):
        """主训练循环 - 增强版多阶段训练"""
        print(f"\n开始训练Enhanced Embodied Counting Model")
        print(f"总计 {self.config['total_epochs']} 个epoch")
        print(f"设备: {self.device}")
        print(f"图像模式: {self.image_mode}")
        print(f"Model info: {self.model.get_model_info()}")
        
        # 初始验证
        stage_name, stage_config = self.get_current_stage(0)
        print(f"进行初始验证...")
        val_loss_init, val_metrics_init, confusion_matrix_init = self.validate(0, stage_config)
        self.save_checkpoint(0, val_loss_init, val_metrics_init['count_accuracy'], 
                           checkpoint_type='init')
        print(f"初始模型已保存，初始验证准确率: {val_metrics_init['count_accuracy']:.4f}")
        
        # 按阶段训练
        for stage_name, stage_config in self.training_stages.items():
            start_epoch, end_epoch = stage_config['epochs']
            stage_epochs = end_epoch - start_epoch
            
            # 跳过已完成的阶段
            if start_epoch < self.start_epoch and end_epoch <= self.start_epoch:
                print(f"跳过 {stage_name}: 已完成")
                continue
            elif start_epoch < self.start_epoch and end_epoch > self.start_epoch:
                start_epoch = self.start_epoch
            
            print(f"\n=== 开始 {stage_name}: {stage_config['description']} ===")
            
            # 设置当前阶段
            self.current_stage = stage_name
            
            # 冻结/解冻模块
            if 'motion_decoder' in stage_config['frozen_modules']:
                for param in self.model.motion_decoder.parameters():
                    param.requires_grad = False
                print("Motion decoder frozen")
            else:
                for param in self.model.motion_decoder.parameters():
                    param.requires_grad = True
            
            print(f"冻结模块: {stage_config['frozen_modules']}")
            print(f"损失权重: {stage_config['loss_weights']}")
            
            # 如果是阶段开始，重置学习率
            if start_epoch == stage_config['epochs'][0]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['learning_rate']
                
                if self.config['scheduler_type'] == 'cosine':
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=stage_epochs
                    )
                
                print(f"学习率已重置为 {self.config['learning_rate']}")
            
            # 阶段内训练循环
            for epoch in range(start_epoch, end_epoch):
                epoch_start_time = time.time()
                
                # 训练一个epoch
                train_loss, train_metrics = self.train_one_epoch(epoch, stage_config)
                
                # 验证
                val_loss, val_metrics, confusion_matrix = self.validate(epoch, stage_config)
                
                # 更新学习率
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # 记录到TensorBoard
                self.log_to_tensorboard(epoch, train_loss, train_metrics, val_loss, 
                                       val_metrics, confusion_matrix, stage_config)
                
                # 打印epoch结果
                epoch_time = time.time() - epoch_start_time
                print(f'\nEpoch [{epoch}] - Time: {epoch_time:.2f}s')
                print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
                print(f'Train Count Acc: {train_metrics["count_accuracy"]:.4f} | '
                      f'Val Count Acc: {val_metrics["count_accuracy"]:.4f}')
                print(f'Final Count Acc: {val_metrics["final_count_accuracy"]:.4f}')
                print(f'Joint MSE: {val_metrics["joint_mse"]:.6f}')
                print(f'Attention Entropy: {val_metrics["attention_entropy"]:.4f}')
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
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        self.writer.close()


def create_enhanced_trainer(config):
    """创建增强训练器"""
    return EnhancedModelTrainer(config)