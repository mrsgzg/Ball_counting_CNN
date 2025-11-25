"""
消融实验模型变体
基于UniversalEmbodiedCountingModel实现各种消融
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_alexnet_embodiment import (
    UniversalEmbodiedCountingModel,
    MultiScaleVisualEncoder,
    AlexNetMultiScaleEncoder,
    EmbodimentEncoder,
    CountingDecoder,
    MotionDecoder,
    TaskGuidedSpatialAttention
)


# ==================== 1. No Forward Model ====================
class NoForwardModelVariant(UniversalEmbodiedCountingModel):
    """消融1: 移除Forward Model（运动预测）
    
    只保留Inverse Model (counting)，移除Forward Model (motion prediction)
    测试: Forward model是否对counting有帮助
    """
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        count_predictions = []
        self.lstm_hidden_states.clear()
        self.attention_weights_history.clear()
        
        # 🔥 关键：始终使用ground truth joints（因为没有forward model来预测）
        for t in range(seq_len):
            current_joints = joints[:, t]
            
            # 视觉特征
            multi_scale_visual, spatial_visual = self.visual_encoder(images[:, t])
            
            # 具身特征
            embodiment_features = self.embodiment_encoder(current_joints)
            
            # 早期融合 + 残差
            fused_features = self.residual_fusion(multi_scale_visual, embodiment_features)
            
            # 空间注意力
            attended_features, attention_weights = self.task_guided_attention(
                query=fused_features,
                spatial_features=spatial_visual
            )
            
            # 组合特征
            combined_features = torch.cat([fused_features, attended_features], dim=1)
            
            # LSTM
            lstm_output, lstm_hidden = self.lstm(
                combined_features.unsqueeze(1), lstm_hidden
            )
            lstm_output = lstm_output.squeeze(1)
            
            # 🔥 只有counting decoder，没有motion decoder
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
            
            self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
            self.attention_weights_history.append(attention_weights.detach().clone())
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            # 🔥 不返回joints预测
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
            
        return outputs
    
    def get_model_info(self):
        info = super().get_model_info()
        info['ablation_type'] = 'no_forward_model'
        info['has_internal_model'] = False
        return info


# ==================== 2. No Spatial Attention ====================
class NoAttentionVariant(UniversalEmbodiedCountingModel):
    """消融2: 移除空间注意力机制
    
    用全局平均池化替代task-guided spatial attention
    测试: 空间注意力的必要性
    """
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        count_predictions = []
        joint_predictions = []
        self.lstm_hidden_states.clear()
        self.attention_weights_history.clear()
        
        current_joints = joints[:, 0]
        
        for t in range(seq_len):
            # 视觉特征
            multi_scale_visual, spatial_visual = self.visual_encoder(images[:, t])
            
            # 具身特征
            embodiment_features = self.embodiment_encoder(current_joints)
            
            # 早期融合 + 残差
            fused_features = self.residual_fusion(multi_scale_visual, embodiment_features)
            
            # 🔥 用全局平均池化替代注意力
            attended_features = F.adaptive_avg_pool2d(spatial_visual, 1).squeeze(-1).squeeze(-1)
            
            # 组合特征
            combined_features = torch.cat([fused_features, attended_features], dim=1)
            
            # LSTM
            lstm_output, lstm_hidden = self.lstm(
                combined_features.unsqueeze(1), lstm_hidden
            )
            lstm_output = lstm_output.squeeze(1)
            
            # Forward & Inverse Models
            joint_pred = self.motion_decoder(lstm_output)
            count_pred = self.counting_decoder(lstm_output)
            
            joint_predictions.append(joint_pred)
            count_predictions.append(count_pred)
            
            # 保存（用于可视化，虽然没有真正的attention）
            dummy_attention = torch.zeros(batch_size, spatial_visual.shape[2], 
                                         spatial_visual.shape[3], device=device)
            self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
            self.attention_weights_history.append(dummy_attention)
            
            # 更新关节
            if use_teacher_forcing and t < seq_len - 1:
                current_joints = joints[:, t + 1]
            else:
                current_joints = joint_pred
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
        
        return outputs
    
    def get_model_info(self):
        info = super().get_model_info()
        info['ablation_type'] = 'no_attention'
        info['has_task_guided_attention'] = False
        return info


# ==================== 3. Late Fusion ====================
class LateFusionVariant(nn.Module):
    """消融3: 晚期融合（在LSTM后才融合视觉和关节）
    
    视觉和关节分别通过各自的LSTM，最后才融合
    测试: Early fusion vs Late fusion
    """
    
    def __init__(self, visual_encoder_type='baseline', **kwargs):
        super().__init__()
        
        self.visual_encoder_type = visual_encoder_type
        self.lstm_layers = kwargs.get('lstm_layers', 2)
        self.lstm_hidden_size = kwargs.get('lstm_hidden_size', 512)
        self.joint_dim = kwargs.get('joint_dim', 7)
        
        # 视觉编码器
        if visual_encoder_type == 'baseline':
            self.visual_encoder = MultiScaleVisualEncoder(
                cnn_layers=kwargs.get('cnn_layers', 3),
                cnn_channels=kwargs.get('cnn_channels', [64, 128, 256]),
                input_channels=kwargs.get('input_channels', 3)
            )
        elif visual_encoder_type in ['alexnet_pretrain', 'alexnet_no_pretrain']:
            self.visual_encoder = AlexNetMultiScaleEncoder(
                input_channels=kwargs.get('input_channels', 3),
                use_pretrain=(visual_encoder_type == 'alexnet_pretrain')
            )
        
        # 具身编码器
        self.embodiment_encoder = EmbodimentEncoder(
            joint_dim=kwargs.get('joint_dim', 7),
            hidden_dim=kwargs.get('feature_dim', 256)
        )
        
        # 🔥 两个独立的LSTM（各占一半hidden size）
        visual_dim = self.visual_encoder.total_feature_dim
        joint_dim = kwargs.get('feature_dim', 256)
        
        self.visual_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=self.lstm_hidden_size // 2,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=kwargs.get('dropout', 0.1) if self.lstm_layers > 1 else 0
        )
        
        self.joint_lstm = nn.LSTM(
            input_size=joint_dim,
            hidden_size=self.lstm_hidden_size // 2,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=kwargs.get('dropout', 0.1) if self.lstm_layers > 1 else 0
        )
        
        # 解码器（接收融合后的特征）
        self.motion_decoder = MotionDecoder(
            input_dim=self.lstm_hidden_size,
            hidden_dim=kwargs.get('feature_dim', 256),
            joint_dim=kwargs.get('joint_dim', 7)
        )
        
        self.counting_decoder = CountingDecoder(
            input_dim=self.lstm_hidden_size,
            hidden_dim=kwargs.get('feature_dim', 256)
        )
        
        # 用于记录（为了与原始模型接口一致）
        self.lstm_hidden_states = []
        self.attention_weights_history = []
    
    def init_lstm_hidden(self, batch_size, device):
        """初始化LSTM hidden state（注意size是减半的）"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size // 2, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size // 2, device=device)
        return (h0, c0)
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 提取所有帧的视觉和关节特征
        visual_features = []
        joint_features = []
        
        for t in range(seq_len):
            multi_scale_visual, _ = self.visual_encoder(images[:, t])
            visual_features.append(multi_scale_visual)
            
            joint_feat = self.embodiment_encoder(joints[:, t])
            joint_features.append(joint_feat)
        
        visual_seq = torch.stack(visual_features, dim=1)  # [B, T, D_v]
        joint_seq = torch.stack(joint_features, dim=1)    # [B, T, D_j]
        
        # 🔥 分别通过LSTM
        visual_hidden = self.init_lstm_hidden(batch_size, device)
        joint_hidden = self.init_lstm_hidden(batch_size, device)
        
        visual_lstm_out, _ = self.visual_lstm(visual_seq, visual_hidden)
        joint_lstm_out, _ = self.joint_lstm(joint_seq, joint_hidden)
        
        # 🔥 在LSTM后融合
        fused_lstm_out = torch.cat([visual_lstm_out, joint_lstm_out], dim=-1)
        
        # 解码
        count_predictions = []
        joint_predictions = []
        
        for t in range(seq_len):
            count_pred = self.counting_decoder(fused_lstm_out[:, t])
            joint_pred = self.motion_decoder(fused_lstm_out[:, t])
            
            count_predictions.append(count_pred)
            joint_predictions.append(joint_pred)
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        return outputs
    
    def get_model_info(self):
        return {
            'model_type': 'LateFusionVariant',
            'visual_encoder_type': self.visual_encoder_type,
            'ablation_type': 'late_fusion',
            'fusion_strategy': 'late',
            'has_internal_model': True,
            'has_early_fusion': False,
            'tasks': ['counting', 'motion_prediction']
        }


# ==================== 模型创建工厂函数 ====================
def create_ablation_model(config, ablation_type):
    """
    创建消融模型的工厂函数
    
    Args:
        config: 模型配置字典
        ablation_type: 'full_model', 'no_forward_model', 'no_attention', 'late_fusion'
    """
    from Model_alexnet_embodiment import create_model
    
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    
    visual_encoder_type = config['model_type']
    model_config['visual_encoder_type'] = visual_encoder_type
    
    if ablation_type == 'full_model':
        # 使用原始完整模型
        return create_model(config, model_type=visual_encoder_type)
    
    elif ablation_type == 'no_forward_model':
        model = NoForwardModelVariant(**model_config)
        print(f"创建No Forward Model消融模型 (基于{visual_encoder_type})")
        return model
    
    elif ablation_type == 'no_attention':
        model = NoAttentionVariant(**model_config)
        print(f"创建No Attention消融模型 (基于{visual_encoder_type})")
        return model
    
    elif ablation_type == 'late_fusion':
        model = LateFusionVariant(**model_config)
        print(f"创建Late Fusion消融模型 (基于{visual_encoder_type})")
        return model
    
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


if __name__ == "__main__":
    """测试各个消融模型"""
    print("=== 测试消融模型变体 ===\n")
    
    config = {
        'model_type': 'baseline',
        'image_mode': 'rgb',
        'model_config': {
            'cnn_layers': 3,
            'cnn_channels': [64, 128, 256],
            'lstm_layers': 2,
            'lstm_hidden_size': 512,
            'feature_dim': 256,
            'joint_dim': 7,
            'dropout': 0.1,
            'use_fovea_bias': True
        }
    }
    
    device = torch.device('cpu')
    batch_size = 2
    seq_len = 11
    
    sequence_data = {
        'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
        'joints': torch.randn(batch_size, seq_len, 7, device=device),
        'timestamps': torch.randn(batch_size, seq_len, device=device),
        'labels': torch.randint(0, 11, (batch_size, seq_len), device=device)
    }
    
    ablation_types = ['full_model', 'no_forward_model', 'no_attention', 'late_fusion']
    
    for ablation in ablation_types:
        print(f"\n{'='*60}")
        print(f"测试: {ablation}")
        print('='*60)
        
        model = create_ablation_model(config, ablation).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数数量: {total_params:,}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(sequence_data)
        
        print(f"输出:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print(f"模型信息: {model.get_model_info()}")
        
        del model
        torch.cuda.empty_cache()
    
    print("\n=== 测试完成 ===")