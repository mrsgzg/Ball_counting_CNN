import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class MultiScaleVisualEncoder(nn.Module):
    """多尺度视觉编码器"""
    
    def __init__(self, cnn_layers=3, cnn_channels=[64, 128, 256], input_channels=3):
        super().__init__()
        
        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels[:cnn_layers]
        
        # 构建CNN层
        layers = []
        in_channels = input_channels
        
        self.conv_blocks = nn.ModuleList()
        for i, out_channels in enumerate(self.cnn_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        # 全局池化层
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(cnn_layers)
        ])
        
        # 计算特征维度
        self.feature_dims = self.cnn_channels
        self.total_feature_dim = sum(self.feature_dims)
        
    def forward(self, x):
        """
        Returns:
            multi_scale_features: [batch, total_feature_dim] 多尺度特征拼接
            spatial_features: [batch, channels[-1], H, W] 最后一层的空间特征图
        """
        multi_scale_feats = []
        
        current = x
        for i, (conv_block, pool) in enumerate(zip(self.conv_blocks, self.global_pools)):
            current = conv_block(current)
            
            # 提取当前层级的全局特征
            global_feat = pool(current).flatten(1)  # [batch, channels]
            multi_scale_feats.append(global_feat)
            
            # 保存最后一层的空间特征
            if i == len(self.conv_blocks) - 1:
                spatial_features = current
        
        # 拼接所有尺度的特征
        multi_scale_features = torch.cat(multi_scale_feats, dim=1)
        
        return multi_scale_features, spatial_features


class EarlyFusionBlock(nn.Module):
    """早期融合块"""
    
    def __init__(self, visual_dim, joint_dim, output_dim, dropout=0.1):
        super().__init__()
        
        # 特征投影
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.joint_proj = nn.Linear(joint_dim, output_dim)
        
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, visual_feat, joint_feat):
        # 投影到相同维度
        vis_proj = self.visual_proj(visual_feat)
        joint_proj = self.joint_proj(joint_feat)
        
        # 拼接并融合
        concatenated = torch.cat([vis_proj, joint_proj], dim=1)
        fused = self.fusion_mlp(concatenated)
        
        return fused


class ResidualMultiModalFusion(nn.Module):
    """残差多模态融合模块"""
    
    def __init__(self, visual_total_dim, joint_dim, hidden_dim=256, num_fusion_layers=3):
        super().__init__()
        
        self.visual_total_dim = visual_total_dim
        self.num_fusion_layers = num_fusion_layers
        
        # 计算每层的视觉特征维度（逐层递增）
        self.visual_layer_dims = []
        base_dim = visual_total_dim // 4
        for i in range(num_fusion_layers):
            if i == 0:
                self.visual_layer_dims.append(base_dim)
            elif i == 1:
                self.visual_layer_dims.append(base_dim * 2)
            else:
                self.visual_layer_dims.append(visual_total_dim)
        
        # 对应的隐藏层维度
        self.hidden_layer_dims = []
        base_hidden = hidden_dim // 4
        for i in range(num_fusion_layers):
            if i == 0:
                self.hidden_layer_dims.append(base_hidden)
            elif i == 1:
                self.hidden_layer_dims.append(base_hidden * 2)
            else:
                self.hidden_layer_dims.append(hidden_dim)
        
        # 早期融合层
        self.fusion_layers = nn.ModuleList([
            EarlyFusionBlock(self.visual_layer_dims[i], joint_dim, self.hidden_layer_dims[i])
            for i in range(num_fusion_layers)
        ])
        
        # 残差投影层
        self.visual_shortcuts = nn.ModuleList([
            nn.Linear(self.visual_layer_dims[i], self.hidden_layer_dims[i])
            for i in range(num_fusion_layers)
        ])
        
        self.joint_shortcuts = nn.ModuleList([
            nn.Linear(joint_dim, self.hidden_layer_dims[i])
            for i in range(num_fusion_layers)
        ])
        
        # 最终输出维度
        self.output_dim = sum(self.hidden_layer_dims)
        
    def forward(self, visual_features, joint_features):
        """
        Args:
            visual_features: [batch, visual_total_dim]
            joint_features: [batch, joint_dim]
        Returns:
            fused_features: [batch, output_dim]
        """
        fused_outputs = []
        
        for i in range(self.num_fusion_layers):
            # 选择当前层级的视觉特征
            start_idx = 0
            for j in range(i):
                start_idx += self.visual_layer_dims[j]
            
            if i < len(self.visual_layer_dims) - 1:
                end_idx = start_idx + self.visual_layer_dims[i]
                vis_feat = visual_features[:, start_idx:end_idx]
            else:
                # 最后一层使用所有特征
                vis_feat = visual_features
            
            # 早期融合
            fused = self.fusion_layers[i](vis_feat, joint_features)
            
            # 残差连接
            vis_residual = self.visual_shortcuts[i](vis_feat)
            joint_residual = self.joint_shortcuts[i](joint_features)
            
            # 三路残差相加
            output = fused + vis_residual + joint_residual
            fused_outputs.append(output)
        
        # 拼接所有层级的输出
        final_fused = torch.cat(fused_outputs, dim=1)
        
        return final_fused


class TaskGuidedSpatialAttention(nn.Module):
    """任务引导的空间注意力"""
    
    def __init__(self, query_dim, key_dim, use_fovea_bias=True, fovea_sigma=0.3):
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.use_fovea_bias = use_fovea_bias
        self.fovea_sigma = fovea_sigma
        
        # Query投影
        self.query_proj = nn.Linear(query_dim, key_dim)
        
        # 可选的温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def _create_fovea_bias(self, H, W, device):
        """创建中央偏置（类似黄斑区）"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 高斯分布，中央权重高
        distance_sq = x**2 + y**2
        fovea_bias = torch.exp(-distance_sq / (2 * self.fovea_sigma**2))
        
        return fovea_bias * 0.5  # 控制偏置强度
        
    def forward(self, query, spatial_features):
        """
        Args:
            query: [batch, query_dim] - LSTM隐状态或融合特征
            spatial_features: [batch, key_dim, H, W] - CNN空间特征图
        Returns:
            attended_features: [batch, key_dim]
            attention_weights: [batch, H, W]
        """
        batch_size, key_dim, H, W = spatial_features.shape
        
        # 投影query
        q = self.query_proj(query)  # [batch, key_dim]
        q = q.unsqueeze(-1).unsqueeze(-1)  # [batch, key_dim, 1, 1]
        
        # 计算注意力分数
        attention_scores = (spatial_features * q).sum(dim=1) / self.temperature  # [batch, H, W]
        
        # 添加fovea偏置
        if self.use_fovea_bias:
            fovea_bias = self._create_fovea_bias(H, W, spatial_features.device)
            attention_scores = attention_scores + fovea_bias.unsqueeze(0)
        
        # 归一化得到注意力权重
        attention_weights = F.softmax(attention_scores.view(batch_size, -1), dim=1)
        attention_weights_2d = attention_weights.view(batch_size, H, W)
        
        # 加权求和得到attended特征
        attention_weights_expanded = attention_weights.unsqueeze(1)  # [batch, 1, H*W]
        spatial_flat = spatial_features.view(batch_size, key_dim, -1)  # [batch, key_dim, H*W]
        
        attended_features = torch.bmm(spatial_flat, attention_weights_expanded.transpose(1, 2))
        attended_features = attended_features.squeeze(-1)  # [batch, key_dim]
        
        return attended_features, attention_weights_2d


class EmbodimentEncoder(nn.Module):
    """具身编码器 - 处理7个关节"""
    
    def __init__(self, joint_dim=7, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, joint_positions):
        """
        Args:
            joint_positions: [batch, 7] - 7个关节位置
        Returns:
            joint_features: [batch, hidden_dim]
        """
        return self.encoder(joint_positions)


class CountingDecoder(nn.Module):
    """计数解码器 - Inverse Model组件"""
    
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=11):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.decoder(x)


class MotionDecoder(nn.Module):
    """运动解码器 - Forward Model组件"""
    
    def __init__(self, input_dim=256, hidden_dim=128, joint_dim=7):
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
        return self.decoder(x)


class EnhancedEmbodiedCountingModel(nn.Module):
    """
    增强的具身计数模型
    - 集成Internal Model思想
    - 早期融合 + 残差结构  
    - Task-guided spatial attention
    """
    
    def __init__(self, 
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 joint_dim=7,
                 input_channels=3,
                 dropout=0.1,
                 use_fovea_bias=True,
                 **kwargs):
        super().__init__()
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.joint_dim = joint_dim
        self.use_fovea_bias = use_fovea_bias
        
        # 多尺度视觉编码器
        self.visual_encoder = MultiScaleVisualEncoder(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            input_channels=input_channels
        )
        
        # 具身编码器
        self.embodiment_encoder = EmbodimentEncoder(
            joint_dim=joint_dim,
            hidden_dim=feature_dim
        )
        
        # 残差多模态融合
        self.residual_fusion = ResidualMultiModalFusion(
            visual_total_dim=self.visual_encoder.total_feature_dim,
            joint_dim=feature_dim,
            hidden_dim=feature_dim
        )
        
        # Task-guided空间注意力
        self.task_guided_attention = TaskGuidedSpatialAttention(
            query_dim=self.residual_fusion.output_dim,
            key_dim=cnn_channels[-1],
            use_fovea_bias=use_fovea_bias
        )
        
        # LSTM时序建模
        total_input_dim = self.residual_fusion.output_dim + cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Forward Model: 运动预测
        self.motion_decoder = MotionDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            joint_dim=joint_dim
        )
        
        # Inverse Model: 计数预测
        self.counting_decoder = CountingDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim
        )
        
        # 存储可视化数据
        self.lstm_hidden_states = []
        self.attention_weights_history = []
        
    def init_lstm_hidden(self, batch_size, device):
        """初始化LSTM隐状态"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        """
        Forward pass with Internal Model architecture
        
        Args:
            sequence_data: dict with 'images', 'joints', 'timestamps', 'labels'
            use_teacher_forcing: bool for training
            return_attention: bool for visualization
        """
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 初始化
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # 存储输出
        count_predictions = []
        joint_predictions = []
        
        # 清空可视化数据
        self.lstm_hidden_states.clear()
        self.attention_weights_history.clear()
        
        # 当前关节位置
        current_joints = joints[:, 0]
        
        for t in range(seq_len):
            # 1. 多尺度视觉特征提取
            multi_scale_visual, spatial_visual = self.visual_encoder(images[:, t])
            
            # 2. 具身特征编码
            embodiment_features = self.embodiment_encoder(current_joints)
            
            # 3. 早期融合 + 残差连接
            fused_features = self.residual_fusion(multi_scale_visual, embodiment_features)
            
            # 4. Task-guided spatial attention
            attended_features, attention_weights = self.task_guided_attention(
                query=fused_features,
                spatial_features=spatial_visual
            )
            
            # 5. 结合融合特征和注意力特征
            combined_features = torch.cat([fused_features, attended_features], dim=1)
            
            # 6. LSTM时序建模
            lstm_output, lstm_hidden = self.lstm(
                combined_features.unsqueeze(1), lstm_hidden
            )
            lstm_output = lstm_output.squeeze(1)
            
            # 保存可视化数据
            self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
            self.attention_weights_history.append(attention_weights.detach().clone())
            
            # 7. Forward Model: 预测下一步关节位置
            joint_pred = self.motion_decoder(lstm_output)
            joint_predictions.append(joint_pred)
            
            # 8. Inverse Model: 基于任务状态预测计数
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
            
            # 9. 更新当前关节位置
            if use_teacher_forcing and t < seq_len - 1:
                current_joints = joints[:, t + 1]
            else:
                current_joints = joint_pred
        
        # 输出结果
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
            
        return outputs
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'EnhancedEmbodiedCountingModel',
            'has_internal_model': True,
            'has_early_fusion': True,
            'has_residual_connections': True,
            'has_task_guided_attention': True,
            'has_fovea_bias': self.use_fovea_bias,
            'tasks': ['counting', 'motion_prediction']
        }