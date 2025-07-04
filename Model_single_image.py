import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class VisualEncoderSingle(nn.Module):
    """单图像CNN视觉编码器 - 复用具身模型的CNN结构"""
    
    def __init__(self, cnn_layers=3, cnn_channels=[64, 128, 256], input_channels=3):
        super().__init__()
        
        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels[:cnn_layers]
        self.feature_hooks = {}
        
        # 构建CNN层 - 与具身模型相同的结构
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
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, 224, 224]
        Returns:
            features: [batch, cnn_channels[-1]]
        """
        features = self.cnn(x)  # [batch, cnn_channels[-1], H, W]
        
        # 全局平均池化
        pooled_features = self.global_pool(features).squeeze(-1).squeeze(-1)  # [batch, cnn_channels[-1]]
        
        return pooled_features
    
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


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, feature_dim, attention_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        self.head_dim = feature_dim // attention_heads
        
        assert feature_dim % attention_heads == 0, "feature_dim must be divisible by attention_heads"
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, feature_dim, H, W]
        Returns:
            attended_x: [batch, feature_dim, H, W]
            attention_weights: [batch, attention_heads, HW, HW]
        """
        batch_size, feature_dim, H, W = x.shape
        
        # Reshape to [batch, HW, feature_dim]
        x_flat = x.view(batch_size, feature_dim, H*W).transpose(1, 2)
        
        # Multi-head attention
        Q = self.query_proj(x_flat)  # [batch, HW, feature_dim]
        K = self.key_proj(x_flat)    # [batch, HW, feature_dim]
        V = self.value_proj(x_flat)  # [batch, HW, feature_dim]
        
        # Reshape for multi-head
        Q = Q.view(batch_size, H*W, self.attention_heads, self.head_dim).transpose(1, 2)  # [batch, heads, HW, head_dim]
        K = K.view(batch_size, H*W, self.attention_heads, self.head_dim).transpose(1, 2)  # [batch, heads, HW, head_dim]
        V = V.view(batch_size, H*W, self.attention_heads, self.head_dim).transpose(1, 2)  # [batch, heads, HW, head_dim]
        
        # Attention computation
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [batch, heads, HW, HW]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)  # [batch, heads, HW, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, H*W, feature_dim)  # [batch, HW, feature_dim]
        
        # Output projection
        attended = self.output_proj(attended)  # [batch, HW, feature_dim]
        
        # Reshape back to [batch, feature_dim, H, W]
        attended = attended.transpose(1, 2).view(batch_size, feature_dim, H, W)
        
        # Residual connection
        output = x + attended
        
        return output, attention_weights


class SingleImageClassifier(nn.Module):
    """单图像分类模型 - 带空间注意力的CNN分类器"""
    
    def __init__(self, 
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 input_channels=3,
                 num_classes=11,
                 hidden_dim=256,
                 attention_heads=4,
                 use_attention=True,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels[:cnn_layers]
        self.use_attention = use_attention
        
        # 视觉编码器 - 复用具身模型的CNN结构
        self.visual_encoder = VisualEncoderSingle(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            input_channels=input_channels
        )
        
        # 空间注意力机制
        if self.use_attention:
            self.spatial_attention = SpatialAttention(
                feature_dim=self.cnn_channels[-1], 
                attention_heads=attention_heads
            )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        feature_dim = self.cnn_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 用于存储可视化数据
        self.last_features = None
        self.last_attention_weights = None
        
        print(f"SingleImageClassifier初始化:")
        print(f"  CNN层数: {cnn_layers}")
        print(f"  CNN通道: {cnn_channels}")
        print(f"  输入通道: {input_channels}")
        print(f"  输出类别: {num_classes}")
        print(f"  特征维度: {feature_dim}")
        print(f"  隐藏维度: {hidden_dim}")
        print(f"  使用注意力: {use_attention}")
        if use_attention:
            print(f"  注意力头数: {attention_heads}")
    
    def forward(self, images, return_features=False, return_attention=False):
        """
        前向传播
        
        Args:
            images: [batch, channels, 224, 224]
            return_features: 是否返回中间特征
            return_attention: 是否返回注意力权重
            
        Returns:
            logits: [batch, num_classes]
            features: [batch, feature_dim] (如果return_features=True)
            attention_weights: [batch, heads, HW, HW] (如果return_attention=True且使用attention)
        """
        # 获取CNN特征图
        cnn_features = self.visual_encoder.cnn(images)  # [batch, cnn_channels[-1], H, W]
        
        # 应用空间注意力
        if self.use_attention:
            attended_features, attention_weights = self.spatial_attention(cnn_features)
            self.last_attention_weights = attention_weights.detach()
        else:
            attended_features = cnn_features
            attention_weights = None
            self.last_attention_weights = None
        
        # 全局池化得到特征向量
        pooled_features = self.global_pool(attended_features).squeeze(-1).squeeze(-1)  # [batch, feature_dim]
        
        # 保存特征用于可视化
        self.last_features = pooled_features.detach()
        
        # 分类
        logits = self.classifier(pooled_features)  # [batch, num_classes]
        
        # 根据需要返回额外信息
        results = [logits]
        if return_features:
            results.append(pooled_features)
        if return_attention and self.use_attention:
            results.append(attention_weights)
        
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
    
    def predict(self, images):
        """预测函数"""
        with torch.no_grad():
            logits = self.forward(images)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': predictions
        }
    
    def get_features(self, images):
        """提取特征"""
        with torch.no_grad():
            if self.use_attention:
                _, features = self.forward(images, return_features=True)
            else:
                features = self.visual_encoder(images)
        return features
    
    def get_attention_maps(self, images):
        """获取注意力图"""
        if not self.use_attention:
            print("模型未使用注意力机制")
            return None
            
        with torch.no_grad():
            if self.use_attention:
                _, _, attention_weights = self.forward(images, return_features=True, return_attention=True)
                return attention_weights
            else:
                return None
    
    def register_hooks(self):
        """注册所有hooks用于可视化"""
        self.visual_encoder.register_hooks()
    
    def get_visualization_data(self):
        """获取可视化数据"""
        viz_data = {
            'visual_features': self.visual_encoder.get_feature_maps(),
            'last_features': self.last_features
        }
        
        if self.use_attention and self.last_attention_weights is not None:
            viz_data['attention_weights'] = self.last_attention_weights
            
        return viz_data
    
    def freeze_backbone(self):
        """冻结CNN backbone"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        print("CNN backbone已冻结")
    
    def unfreeze_backbone(self):
        """解冻CNN backbone"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = True
        print("CNN backbone已解冻")
    
    def freeze_classifier(self):
        """冻结分类器"""
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("分类器已冻结")
    
    def unfreeze_classifier(self):
        """解冻分类器"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("分类器已解冻")


def create_single_image_model(config):
    """创建单图像分类模型的工厂函数"""
    
    # 确定输入通道数
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    # 提取模型配置
    model_config = config.get('model_config', {})
    
    # 基础模型参数
    model_params = {
        'cnn_layers': model_config.get('cnn_layers', 3),
        'cnn_channels': model_config.get('cnn_channels', [64, 128, 256]),
        'input_channels': input_channels,
        'num_classes': config.get('num_classes', 11),
        'hidden_dim': model_config.get('feature_dim', 256),
        'attention_heads': model_config.get('attention_heads', 4),
        'use_attention': config.get('use_attention', True),
        'dropout': model_config.get('dropout', 0.1)
    }
    
    model = SingleImageClassifier(**model_params)
    print("创建带注意力机制的单图像分类模型")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== 单图像分类模型测试 ===")
    
    # 测试配置
    config = {
        'image_mode': 'rgb',
        'num_classes': 10,
        'model_config': {
            'cnn_layers': 3,
            'cnn_channels': [64, 128, 256],
            'feature_dim': 256,
            'dropout': 0.1
        },
        'use_regularization': False
    }
    
    # 创建模型
    model = create_single_image_model(config)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # RGB测试
    print(f"\n=== RGB模式测试 ===")
    batch_size = 8
    rgb_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    with torch.no_grad():
        rgb_output = model(rgb_input)
        rgb_pred = model.predict(rgb_input)
    
    print(f"RGB输入形状: {rgb_input.shape}")
    print(f"RGB输出形状: {rgb_output.shape}")
    print(f"RGB预测形状: {rgb_pred['predictions'].shape}")
    print(f"预测范围: [{rgb_pred['predictions'].min()}, {rgb_pred['predictions'].max()}]")
    