import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class VisualEncoder(nn.Module):
    """可配置的CNN视觉编码器，支持1-3层"""
    
    def __init__(self, cnn_layers=3, cnn_channels=[64, 128, 256], input_channels=1, feature_dim=256):
        super().__init__()
        
        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels[:cnn_layers]
        self.feature_hooks = {}
        
        # 构建CNN层
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(self.cnn_channels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # 计算展平后的特征维度
        spatial_size = 224 // (2 ** cnn_layers)
        self.flattened_size = self.cnn_channels[-1] * spatial_size * spatial_size
        
        # 用于获取空间特征的适配层
        self.spatial_adapter = nn.Conv2d(self.cnn_channels[-1], feature_dim, 1)
        
        # 全局池化用于获得全局特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_adapter = nn.Linear(self.cnn_channels[-1], feature_dim)
        
    def forward(self, x):
        """
        返回两种特征：
        - spatial_features: [B, feature_dim, H, W] 用于attention
        - global_features: [B, feature_dim] 用于融合
        """
        features = self.cnn(x)
        
        # 空间特征（保持空间维度）
        spatial_features = self.spatial_adapter(features)
        
        # 全局特征
        global_pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        global_features = self.global_adapter(global_pooled)
        
        return spatial_features, global_features
    
    def register_hooks(self):
        """注册feature hooks用于可视化"""
        for i, layer in enumerate(self.cnn):
            if isinstance(layer, nn.ReLU):
                def hook_fn(name):
                    def fn(module, input, output):
                        self.feature_hooks[name] = output.detach()
                    return fn
                layer.register_forward_hook(hook_fn(f'layer_{i}'))
    
    def get_feature_maps(self):
        """获取保存的特征图"""
        return self.feature_hooks


class EmbodimentEncoder(nn.Module):
    """具身编码器"""
    
    def __init__(self, joint_dim=8, hidden_dim=256):
        super().__init__()
        
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, joint_positions):
        """
        joint_positions: [batch, 8] - 当前的关节位置
        """
        return self.joint_encoder(joint_positions)


class MultiModalFusion(nn.Module):
    """多模态融合模块，使用Cross-Modal Attention"""
    
    def __init__(self, feature_dim=256, attention_heads=8, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        self.head_dim = feature_dim // attention_heads
        
        # Cross-Modal Attention layers
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Feature integration
        self.integration = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, visual_spatial, visual_global, embodiment_features):
        """
        visual_spatial: [B, feature_dim, H, W]
        visual_global: [B, feature_dim]
        embodiment_features: [B, feature_dim]
        """
        batch_size = embodiment_features.shape[0]
        
        # 准备spatial visual features for attention
        H, W = visual_spatial.shape[-2:]
        visual_spatial_flat = visual_spatial.flatten(-2).transpose(-1, -2)  # [B, HW, feature_dim]
        
        # Cross-Modal Attention: embodiment queries visual
        Q = self.query_proj(embodiment_features.unsqueeze(1))  # [B, 1, feature_dim]
        K = self.key_proj(visual_spatial_flat)  # [B, HW, feature_dim]
        V = self.value_proj(visual_spatial_flat)  # [B, HW, feature_dim]
        
        # Multi-head attention
        Q = Q.view(batch_size, 1, self.attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, H*W, self.attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, H*W, self.attention_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        attended_visual = torch.matmul(attention_weights, V)  # [B, heads, 1, head_dim]
        attended_visual = attended_visual.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim).squeeze(1)  # [B, feature_dim]
        
        attended_visual = self.output_proj(attended_visual)
        
        # Feature integration
        fused_features = self.integration(
            torch.cat([visual_global, embodiment_features], dim=-1)
        )
        
        # 结合attention结果和全局特征
        final_features = fused_features + attended_visual
        
        return final_features, attention_weights.squeeze(2)  # 返回attention权重用于可视化


class CountingDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=11):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)  # 输出11个类别的logits
        )
        
    def forward(self, x):
        return self.decoder(x)  # [batch, seq_len, 11]


class MotionDecoder(nn.Module):
    """动作解码器"""
    
    def __init__(self, input_dim=256, hidden_dim=128, joint_dim=8):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim)
        )
        
    def forward(self, x):
        """输出下一帧的关节位置预测"""
        return self.decoder(x)


class EmbodiedCountingModel(nn.Module):
    """具身计数模型"""
    
    def __init__(self, 
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 attention_heads=8,
                 joint_dim=8,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.joint_dim = joint_dim
        
        # 各模块初始化
        self.visual_encoder = VisualEncoder(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            feature_dim=feature_dim
        )
        
        self.embodiment_encoder = EmbodimentEncoder(
            joint_dim=joint_dim,
            hidden_dim=feature_dim
        )
        
        self.fusion = MultiModalFusion(
            feature_dim=feature_dim,
            attention_heads=attention_heads,
            dropout=dropout
        )
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.counting_decoder = CountingDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim
        )
        
        self.motion_decoder = MotionDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            joint_dim=joint_dim
        )
        
        # 用于存储可视化数据
        self.lstm_hidden_states = []
        self.attention_weights_history = []
        
        # 冻结标志
        self.frozen_modules = set()
        
    def init_lstm_hidden(self, batch_size, device):
        """初始化LSTM隐状态"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, 
                images, 
                initial_joints, 
                target_joints=None, 
                use_teacher_forcing=True,
                return_attention=False):
        """
        Args:
            images: [batch, seq_len, 1, 224, 224]
            initial_joints: [batch, 8] - 初始关节位置
            target_joints: [batch, seq_len, 8] - 目标关节位置（训练时使用）
            use_teacher_forcing: 是否使用teacher forcing
            return_attention: 是否返回attention权重
        """
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 初始化LSTM隐状态
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # 存储输出
        count_predictions = []
        joint_predictions = []
        
        # 清空可视化数据
        self.lstm_hidden_states.clear()
        self.attention_weights_history.clear()
        
        # 当前关节位置（初始为initial_joints）
        current_joints = initial_joints
        
        for t in range(seq_len):
            # 1. 编码当前帧的视觉信息
            visual_spatial, visual_global = self.visual_encoder(images[:, t])
            
            # 2. 编码当前关节位置
            embodiment_features = self.embodiment_encoder(current_joints)
            
            # 3. 多模态融合
            fused_features, attention_weights = self.fusion(
                visual_spatial, visual_global, embodiment_features
            )
            
            # 4. LSTM处理
            lstm_output, lstm_hidden = self.lstm(
                fused_features.unsqueeze(1), lstm_hidden
            )
            lstm_output = lstm_output.squeeze(1)
            
            # 保存LSTM隐状态（用于可视化）
            self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
            self.attention_weights_history.append(attention_weights.detach().clone())
            
            # 5. 预测count和下一步关节位置
            count_pred = self.counting_decoder(lstm_output)
            joint_pred = self.motion_decoder(lstm_output)
            
            count_predictions.append(count_pred)
            joint_predictions.append(joint_pred)
            
            # 6. 更新当前关节位置（用于下一时刻）
            if use_teacher_forcing and target_joints is not None and t < seq_len - 1:
                # 训练时使用真实值
                current_joints = target_joints[:, t]
            else:
                # 推理时使用预测值
                current_joints = joint_pred
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
            
        return outputs
    
    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        modules_map = {
            'visual': self.visual_encoder,
            'embodiment': self.embodiment_encoder,
            'fusion': self.fusion,
            'lstm': self.lstm,
            'counting': self.counting_decoder,
            'motion': self.motion_decoder
        }
        
        if module_name in modules_map:
            module = modules_map[module_name]
            for param in module.parameters():
                param.requires_grad = False
            self.frozen_modules.add(module_name)
        else:
            raise ValueError(f"Module {module_name} not found. Available: {list(modules_map.keys())}")
    
    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        modules_map = {
            'visual': self.visual_encoder,
            'embodiment': self.embodiment_encoder,
            'fusion': self.fusion,
            'lstm': self.lstm,
            'counting': self.counting_decoder,
            'motion': self.motion_decoder
        }
        
        if module_name in modules_map:
            module = modules_map[module_name]
            for param in module.parameters():
                param.requires_grad = True
            self.frozen_modules.discard(module_name)
        else:
            raise ValueError(f"Module {module_name} not found. Available: {list(modules_map.keys())}")
    
    def set_training_mode(self, mode='both'):
        """设置训练模式"""
        if mode == 'counting_only':
            self.freeze_module('motion')
        elif mode == 'motion_only':
            self.freeze_module('counting')
        elif mode == 'both':
            self.unfreeze_module('counting')
            self.unfreeze_module('motion')
        else:
            raise ValueError(f"Unknown training mode: {mode}. Available: 'both', 'counting_only', 'motion_only'")
    
    def register_hooks(self):
        """注册所有hooks用于可视化"""
        self.visual_encoder.register_hooks()
    
    def get_visualization_data(self):
        """获取可视化数据"""
        return {
            'visual_features': self.visual_encoder.get_feature_maps(),
            'lstm_hidden_states': self.lstm_hidden_states,
            'attention_weights': self.attention_weights_history
        }