import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

# 导入现有模型组件
from Model_embodiment import (
    VisualEncoder, 
    EmbodimentEncoder, 
    MultiModalFusion, 
    CountingDecoder,
    EmbodiedCountingModel
)


class EmbodiedCountingOnlyModel(EmbodiedCountingModel):
    """
    消融实验模型1: 纯计数具身模型
    - 保留完整的视觉+具身处理流程
    - 移除关节预测任务，只做计数
    - 基于现有具身模型，移除MotionDecoder
    """
    
    def __init__(self, 
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 attention_heads=8,
                 joint_dim=7,
                 input_channels=3,
                 dropout=0.1,
                 **kwargs):
        """
        初始化纯计数具身模型
        参数与原始模型相同，但不创建MotionDecoder
        """
        # 调用父类构造函数，但我们需要重写以移除motion_decoder
        nn.Module.__init__(self)  # 跳过EmbodiedCountingModel的__init__
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.joint_dim = joint_dim
        
        # 初始化所有模块（除了motion_decoder）
        self.visual_encoder = VisualEncoder(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            input_channels=input_channels,
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
        
        # 注意：不创建motion_decoder
        # self.motion_decoder = None
        
        # 用于存储可视化数据
        self.lstm_hidden_states = []
        self.attention_weights_history = []
        
        # 冻结标志
        self.frozen_modules = set()
        
        print("EmbodiedCountingOnlyModel 初始化完成 - 只进行计数任务")
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        """
        前向传播 - 只输出计数结果
        
        Args:
            sequence_data: dict包含:
                - 'images': [batch, seq_len, channels, 224, 224]
                - 'joints': [batch, seq_len, 7] 
                - 'timestamps': [batch, seq_len]
                - 'labels': [batch, seq_len]
            use_teacher_forcing: 是否使用teacher forcing
            return_attention: 是否返回attention权重
        
        Returns:
            dict with:
                - 'counts': [batch, seq_len, 11] - 计数预测
                - 'attention_weights': [batch, seq_len, ...] (如果return_attention=True)
        """
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 初始化LSTM隐状态
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # 存储输出
        count_predictions = []
        
        # 清空可视化数据
        self.lstm_hidden_states.clear()
        self.attention_weights_history.clear()
        
        # 当前关节位置（初始为第一帧的关节位置）
        current_joints = joints[:, 0]  # [batch, 7]
        
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
            
            # 5. 预测count（不预测关节位置）
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
            
            # 6. 更新当前关节位置（用于下一时刻）
            if use_teacher_forcing and t < seq_len - 1:
                # 训练时使用真实值
                current_joints = joints[:, t + 1]
            else:
                # 推理时保持当前关节位置不变（因为没有预测）
                # 或者可以使用简单的启发式方法
                current_joints = current_joints  # 保持不变
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),    # [batch, seq_len, 11]
            # 不返回joints预测
        }
        
        if return_attention:
            outputs['attention_weights'] = torch.stack(self.attention_weights_history, dim=1)
            
        return outputs
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'EmbodiedCountingOnly',
            'has_motion_decoder': False,
            'has_embodiment': True,
            'tasks': ['counting']
        }


class VisualOnlyCountingModel(nn.Module):
    """
    消融实验模型2: 纯视觉计数模型
    - 移除具身编码器和多模态融合
    - 只使用视觉信息进行计数
    - 基于现有具身模型的视觉部分
    """
    
    def __init__(self, 
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 input_channels=3,
                 dropout=0.1,
                 **kwargs):
        """
        初始化纯视觉计数模型
        """
        super().__init__()
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # 只初始化视觉相关模块
        self.visual_encoder = VisualEncoder(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            input_channels=input_channels,
            feature_dim=feature_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,  # 直接使用视觉特征
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.counting_decoder = CountingDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim
        )
        
        # 用于存储可视化数据
        self.lstm_hidden_states = []
        
        # 冻结标志
        self.frozen_modules = set()
        
        print("VisualOnlyCountingModel 初始化完成 - 只使用视觉信息进行计数")
    
    def init_lstm_hidden(self, batch_size, device):
        """初始化LSTM隐状态"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_attention=False):
        """
        前向传播 - 只使用视觉信息
        
        Args:
            sequence_data: dict包含:
                - 'images': [batch, seq_len, channels, 224, 224]
                - 'joints': [batch, seq_len, 7] - 忽略
                - 'timestamps': [batch, seq_len] - 忽略
                - 'labels': [batch, seq_len]
            use_teacher_forcing: 忽略（没有关节预测）
            return_attention: 忽略（没有attention机制）
        
        Returns:
            dict with:
                - 'counts': [batch, seq_len, 11] - 计数预测
        """
        images = sequence_data['images']
        # 忽略joints和timestamps
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 初始化LSTM隐状态
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # 存储输出
        count_predictions = []
        
        # 清空可视化数据
        self.lstm_hidden_states.clear()
        
        for t in range(seq_len):
            # 1. 编码当前帧的视觉信息
            visual_spatial, visual_global = self.visual_encoder(images[:, t])
            
            # 2. 直接使用视觉全局特征（不进行多模态融合）
            visual_features = visual_global  # [batch, feature_dim]
            
            # 3. LSTM处理
            lstm_output, lstm_hidden = self.lstm(
                visual_features.unsqueeze(1), lstm_hidden
            )
            lstm_output = lstm_output.squeeze(1)
            
            # 保存LSTM隐状态（用于可视化）
            self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
            
            # 4. 预测count
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
        
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),    # [batch, seq_len, 11]
            # 不返回joints预测
        }
        
        return outputs
    
    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        modules_map = {
            'visual': self.visual_encoder,
            'lstm': self.lstm,
            'counting': self.counting_decoder
        }
        
        if module_name in modules_map:
            module = modules_map[module_name]
            for param in module.parameters():
                param.requires_grad = False
            self.frozen_modules.add(module_name)
            print(f"模块 '{module_name}' 已冻结")
        else:
            raise ValueError(f"Module {module_name} not found. Available: {list(modules_map.keys())}")
    
    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        modules_map = {
            'visual': self.visual_encoder,
            'lstm': self.lstm,
            'counting': self.counting_decoder
        }
        
        if module_name in modules_map:
            module = modules_map[module_name]
            for param in module.parameters():
                param.requires_grad = True
            self.frozen_modules.discard(module_name)
            print(f"模块 '{module_name}' 已解冻")
        else:
            raise ValueError(f"Module {module_name} not found. Available: {list(modules_map.keys())}")
    
    def register_hooks(self):
        """注册所有hooks用于可视化"""
        self.visual_encoder.register_hooks()
    
    def get_visualization_data(self):
        """获取可视化数据"""
        return {
            'visual_features': self.visual_encoder.get_feature_maps(),
            'lstm_hidden_states': self.lstm_hidden_states,
            # 没有attention权重
        }
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'VisualOnlyCountingModel',
            'has_motion_decoder': False,
            'has_embodiment': False,
            'tasks': ['counting']
        }


def create_ablation_model(model_type, config):
    """
    创建消融实验模型的工厂函数
    
    Args:
        model_type: 'counting_only' 或 'visual_only'
        config: 模型配置字典
    
    Returns:
        消融模型实例
    """
    # 确定图像模式
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    # 获取模型配置
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    
    if model_type == 'counting_only':
        model = EmbodiedCountingOnlyModel(**model_config)
        print("创建纯计数具身模型 (有具身信息，无关节预测)")
    elif model_type == 'visual_only':
        model = VisualOnlyCountingModel(**model_config)
        print("创建纯视觉计数模型 (无具身信息，无关节预测)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Available: 'counting_only', 'visual_only'")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== 消融实验模型测试 ===")
    
    # 测试配置
    config = {
        'image_mode': 'rgb',
        'model_config': {
            'cnn_layers': 3,
            'cnn_channels': [64, 128, 256],
            'lstm_layers': 2,
            'lstm_hidden_size': 512,
            'feature_dim': 256,
            'attention_heads': 1,
            'joint_dim': 7,
            'dropout': 0.1
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 16
    seq_len = 11
    sequence_data = {
        'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
        'joints': torch.randn(batch_size, seq_len, 7, device=device),
        'timestamps': torch.randn(batch_size, seq_len, device=device),
        'labels': torch.randint(0, 11, (batch_size, seq_len), device=device)
    }
    
    print(f"测试数据形状:")
    for key, value in sequence_data.items():
        print(f"  {key}: {value.shape}")
    
    # 测试纯计数具身模型
    print(f"\n=== 测试纯计数具身模型 ===")
    counting_only_model = create_ablation_model('counting_only', config).to(device)
    print(counting_only_model)
    # 模型信息
    counting_params = sum(p.numel() for p in counting_only_model.parameters())
    print(f"纯计数模型参数数量: {counting_params:,}")
    print(f"模型信息: {counting_only_model.get_model_info()}")
    
    # 前向传播测试
    counting_only_model.eval()
    with torch.no_grad():
        counting_output = counting_only_model(sequence_data)
    
    print(f"输出形状:")
    for key, value in counting_output.items():
        print(f"  {key}: {value.shape}")
    
    # 测试纯视觉模型
    print(f"\n=== 测试纯视觉模型 ===")
    visual_only_model = create_ablation_model('visual_only', config).to(device)
    print(visual_only_model)
    # 模型信息
    visual_params = sum(p.numel() for p in visual_only_model.parameters())
    print(f"纯视觉模型参数数量: {visual_params:,}")
    print(f"模型信息: {visual_only_model.get_model_info()}")
    
    # 前向传播测试
    visual_only_model.eval()
    with torch.no_grad():
        visual_output = visual_only_model(sequence_data)
    
    print(f"输出形状:")
    for key, value in visual_output.items():
        print(f"  {key}: {value.shape}")
    
    # 比较原始模型参数数量
    print(f"\n=== 模型参数对比 ===")
    original_model = EmbodiedCountingModel(**config['model_config']).to(device)
    original_params = sum(p.numel() for p in original_model.parameters())
    
    print(f"原始完整模型参数数量: {original_params:,}")
    print(f"纯计数模型参数数量: {counting_params:,} ({counting_params/original_params*100:.1f}%)")
    print(f"纯视觉模型参数数量: {visual_params:,} ({visual_params/original_params*100:.1f}%)")
    
    print(f"\n=== 测试完成 ===")