import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import torchvision.models as models


class MultiScaleVisualEncoder(nn.Module):
    """原始的多尺度视觉编码器 - 简单三层CNN"""
    
    def __init__(self, cnn_layers=3, cnn_channels=[64, 128, 256], input_channels=3):
        super().__init__()
        
        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels[:cnn_layers]
        
        # 构建CNN层
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        
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


class AlexNetMultiScaleEncoder(nn.Module):
    """基于AlexNet的多尺度视觉编码器"""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # 加载AlexNet
        if use_pretrain:
            self.alexnet = models.alexnet(pretrained=True)
        else:
            self.alexnet = models.alexnet(pretrained=False)
        
        # 提取AlexNet的特征层
        self.features = self.alexnet.features
        
        # 如果输入不是3通道，修改第一层
        if input_channels != 3:
            original_conv1 = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=11, stride=4, padding=2
            )
            # 如果是预训练模型且输入通道为1，平均RGB权重
            if use_pretrain and input_channels == 1:
                with torch.no_grad():
                    self.features[0].weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        # AlexNet特征提取点：conv1(64), conv3(192), conv5(256)
        self.extract_indices = [2, 5, 12]  # ReLU层后
        self.feature_dims = [64, 192, 256]
        self.total_feature_dim = sum(self.feature_dims)
        
        # 全局池化层
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(len(self.feature_dims))
        ])
        
    def forward(self, x):
        """
        Returns:
            multi_scale_features: [batch, total_feature_dim] 多尺度特征拼接
            spatial_features: [batch, 256, H, W] 最后一层的空间特征图
        """
        multi_scale_feats = []
        
        current = x
        extract_idx = 0
        
        for i, layer in enumerate(self.features):
            current = layer(current)
            
            if extract_idx < len(self.extract_indices) and i == self.extract_indices[extract_idx]:
                global_feat = self.global_pools[extract_idx](current).flatten(1)
                multi_scale_feats.append(global_feat)
                extract_idx += 1
                
                if extract_idx == len(self.extract_indices):
                    spatial_features = current
        
        multi_scale_features = torch.cat(multi_scale_feats, dim=1)
        
        return multi_scale_features, spatial_features


class EarlyFusionBlock(nn.Module):
    """早期融合块"""
    
    def __init__(self, visual_dim, joint_dim, output_dim, dropout=0.1):
        super().__init__()
        
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.joint_proj = nn.Linear(joint_dim, output_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, visual_feat, joint_feat):
        vis_proj = self.visual_proj(visual_feat)
        joint_proj = self.joint_proj(joint_feat)
        
        concatenated = torch.cat([vis_proj, joint_proj], dim=1)
        fused = self.fusion_mlp(concatenated)
        
        return fused


class ResidualMultiModalFusion(nn.Module):
    """残差多模态融合模块"""
    
    def __init__(self, visual_total_dim, joint_dim, hidden_dim=256, num_fusion_layers=3):
        super().__init__()
        
        self.visual_total_dim = visual_total_dim
        self.num_fusion_layers = num_fusion_layers
        
        # 计算每层的视觉特征维度
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
        
        self.output_dim = sum(self.hidden_layer_dims)
        
    def forward(self, visual_features, joint_features):
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
                vis_feat = visual_features
            
            # 早期融合
            fused = self.fusion_layers[i](vis_feat, joint_features)
            
            # 残差连接
            vis_residual = self.visual_shortcuts[i](vis_feat)
            joint_residual = self.joint_shortcuts[i](joint_features)
            
            # 三路残差相加
            output = fused + vis_residual + joint_residual
            fused_outputs.append(output)
        
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
        
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def _create_fovea_bias(self, H, W, device):
        """创建中央偏置（类似黄斑区）"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        distance_sq = x**2 + y**2
        fovea_bias = torch.exp(-distance_sq / (2 * self.fovea_sigma**2))
        
        return fovea_bias * 0.5
        
    def forward(self, query, spatial_features):
        batch_size, key_dim, H, W = spatial_features.shape
        
        q = self.query_proj(query)
        q = q.unsqueeze(-1).unsqueeze(-1)
        
        attention_scores = (spatial_features * q).sum(dim=1) / self.temperature
        
        if self.use_fovea_bias:
            fovea_bias = self._create_fovea_bias(H, W, spatial_features.device)
            attention_scores = attention_scores + fovea_bias.unsqueeze(0)
        
        attention_weights = F.softmax(attention_scores.view(batch_size, -1), dim=1)
        attention_weights_2d = attention_weights.view(batch_size, H, W)
        
        attention_weights_expanded = attention_weights.unsqueeze(1)
        spatial_flat = spatial_features.view(batch_size, key_dim, -1)
        
        attended_features = torch.bmm(spatial_flat, attention_weights_expanded.transpose(1, 2))
        attended_features = attended_features.squeeze(-1)
        
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
        return self.encoder(joint_positions)


class CountingDecoder(nn.Module):
    """计数解码器"""
    
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
    """运动解码器"""
    
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


class UniversalEmbodiedCountingModel(nn.Module):
    """通用具身计数模型 - 支持三种视觉编码器"""
    
    def __init__(self, 
                 visual_encoder_type='baseline',  # 'baseline', 'alexnet_pretrain', 'alexnet_no_pretrain'
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
        self.visual_encoder_type = visual_encoder_type
        
        # 选择视觉编码器
        if visual_encoder_type == 'baseline':
            self.visual_encoder = MultiScaleVisualEncoder(
                cnn_layers=cnn_layers,
                cnn_channels=cnn_channels,
                input_channels=input_channels
            )
            spatial_channel_dim = cnn_channels[-1]
        elif visual_encoder_type == 'alexnet_pretrain':
            self.visual_encoder = AlexNetMultiScaleEncoder(
                input_channels=input_channels,
                use_pretrain=True
            )
            spatial_channel_dim = 256
        elif visual_encoder_type == 'alexnet_no_pretrain':
            self.visual_encoder = AlexNetMultiScaleEncoder(
                input_channels=input_channels,
                use_pretrain=False
            )
            spatial_channel_dim = 256
        else:
            raise ValueError(f"Unsupported visual_encoder_type: {visual_encoder_type}")
        
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
            key_dim=spatial_channel_dim,
            use_fovea_bias=use_fovea_bias
        )
        
        # LSTM时序建模
        total_input_dim = self.residual_fusion.output_dim + spatial_channel_dim
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
        
        print(f"UniversalEmbodiedCountingModel 初始化:")
        print(f"  视觉编码器: {visual_encoder_type}")
        print(f"  总特征维度: {self.visual_encoder.total_feature_dim}")
        print(f"  融合输出维度: {self.residual_fusion.output_dim}")
        print(f"  LSTM输入维度: {total_input_dim}")
        
    def init_lstm_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
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
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
            
        return outputs
    
    def get_model_info(self):
        return {
            'model_type': 'UniversalEmbodiedCountingModel',
            'visual_encoder_type': self.visual_encoder_type,
            'has_internal_model': True,
            'has_early_fusion': True,
            'has_residual_connections': True,
            'has_task_guided_attention': True,
            'has_fovea_bias': self.use_fovea_bias,
            'tasks': ['counting', 'motion_prediction']
        }


def create_model(config, model_type='baseline'):
    """
    创建具身计数模型的工厂函数
    
    Args:
        config: 模型配置字典
        model_type: 'baseline', 'alexnet_pretrain', 'alexnet_no_pretrain'
    """
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    model_config['visual_encoder_type'] = model_type
    
    model = UniversalEmbodiedCountingModel(**model_config)
    
    type_names = {
        'baseline': 'Baseline (3-layer CNN)',
        'alexnet_pretrain': 'AlexNet (Pretrained)',
        'alexnet_no_pretrain': 'AlexNet (No Pretrain)'
    }
    
    print(f"创建{type_names[model_type]}具身计数模型")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== 通用具身计数模型测试 ===")
    
    config = {
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
    print()
    device = torch.device('cpu')
    
    model_types = ['baseline', 'alexnet_no_pretrain', 'alexnet_pretrain']
    
    for model_type in model_types:
        print(f"\n=== 测试 {model_type} ===")
        
        model = create_model(config, model_type=model_type).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")
        
        batch_size = 4
        seq_len = 11
        sequence_data = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
            'joints': torch.randn(batch_size, seq_len, 7, device=device),
            'timestamps': torch.randn(batch_size, seq_len, device=device),
            'labels': torch.randint(0, 11, (batch_size, seq_len), device=device)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(sequence_data, return_attention=True)
        
        print(f"输出形状:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print(f"模型信息: {model.get_model_info()}")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    print("\n=== 测试完成 ===")