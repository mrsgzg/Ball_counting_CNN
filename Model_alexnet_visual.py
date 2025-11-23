import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import torchvision.models as models


class AlexNetVisualOnlyModel(nn.Module):
    """纯视觉AlexNet模型 - 用于单图像分类对比实验"""
    
    def __init__(self, 
                 num_classes=11,
                 input_channels=3,
                 use_pretrain=True,
                 feature_dim=256,
                 dropout=0.5):
        """
        Args:
            num_classes: 输出类别数（球数1-10，共11类包括0）
            input_channels: 输入通道数 (3 for RGB, 1 for grayscale)
            use_pretrain: 是否使用预训练权重
            feature_dim: 分类头隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_pretrain = use_pretrain
        self.feature_dim = feature_dim
        
        # 加载AlexNet
        if use_pretrain:
            print("加载预训练AlexNet权重...")
            self.alexnet = models.alexnet(pretrained=True)
        else:
            print("使用随机初始化的AlexNet...")
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
                print("调整第一层卷积以适应灰度输入...")
                with torch.no_grad():
                    self.features[0].weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        # AlexNet的分类器部分需要调整
        # 原始AlexNet avgpool后是256*6*6=9216维
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 自定义分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 初始化新添加的层
        self._initialize_weights()
        
        print(f"AlexNetVisualOnlyModel 初始化完成:")
        print(f"  输入通道数: {input_channels}")
        print(f"  输出类别数: {num_classes}")
        print(f"  使用预训练: {use_pretrain}")
        print(f"  特征维度: {feature_dim}")
        print(f"  Dropout率: {dropout}")
        
    def _initialize_weights(self):
        """初始化分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # 特征提取
        features = self.features(x)
        
        # 池化
        features = self.avgpool(features)
        
        # 展平
        features = torch.flatten(features, 1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def get_model_info(self):
        """返回模型信息"""
        return {
            'model_type': 'AlexNetVisualOnly',
            'use_pretrain': self.use_pretrain,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }


def create_visual_model(config, use_pretrain=True):
    """
    创建纯视觉分类模型的工厂函数
    
    Args:
        config: 模型配置字典
        use_pretrain: 是否使用预训练权重
    
    Returns:
        model: AlexNetVisualOnlyModel实例
    """
    # 获取图像模式以确定输入通道数
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    # 获取模型配置参数
    model_config = config.get('model_config', {})
    
    model = AlexNetVisualOnlyModel(
        num_classes=config.get('num_classes', 11),
        input_channels=input_channels,
        use_pretrain=use_pretrain,
        feature_dim=model_config.get('feature_dim', 256),
        dropout=model_config.get('dropout', 0.5)
    )
    
    model_type = 'AlexNet (Pretrained)' if use_pretrain else 'AlexNet (No Pretrain)'
    print(f"\n创建{model_type}纯视觉模型")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== AlexNet纯视觉模型测试 ===")
    
    # 配置
    config = {
        'image_mode': 'rgb',
        'num_classes': 11,
        'model_config': {
            'feature_dim': 256,
            'dropout': 0.5
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试两种模型
    print("\n--- 测试预训练模型 ---")
    model_pretrain = create_visual_model(config, use_pretrain=True).to(device)
    
    print("\n--- 测试非预训练模型 ---")
    model_no_pretrain = create_visual_model(config, use_pretrain=False).to(device)
    
    # 测试输入
    print("\n--- 测试前向传播 ---")
    batch_size = 16
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 测试预训练模型
    model_pretrain.eval()
    with torch.no_grad():
        output_pretrain = model_pretrain(test_input)
    print(f"预训练模型输出形状: {output_pretrain.shape}")
    
    # 测试非预训练模型
    model_no_pretrain.eval()
    with torch.no_grad():
        output_no_pretrain = model_no_pretrain(test_input)
    print(f"非预训练模型输出形状: {output_no_pretrain.shape}")
    
    # 测试灰度图像输入
    print("\n--- 测试灰度图像输入 ---")
    config_gray = {
        'image_mode': 'grayscale',
        'num_classes': 11,
        'model_config': {
            'feature_dim': 256,
            'dropout': 0.5
        }
    }
    
    model_gray = create_visual_model(config_gray, use_pretrain=True).to(device)
    test_input_gray = torch.randn(batch_size, 1, 224, 224).to(device)
    
    model_gray.eval()
    with torch.no_grad():
        output_gray = model_gray(test_input_gray)
    print(f"灰度模型输出形状: {output_gray.shape}")
    
    print("\n=== 测试完成 ===")