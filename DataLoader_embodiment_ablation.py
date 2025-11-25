"""
消融实验数据加载器
提供数据打乱功能用于Shuffled Batch和Shuffled Temporal消融实验
"""

import torch
from torch.utils.data import DataLoader


class ShuffledBatchWrapper:
    """
    Shuffled Batch消融: 打乱样本间的视觉-关节配对
    
    在batch维度随机打乱joints，破坏视觉-运动的语义对应关系
    保留: 时序顺序
    破坏: 场景配对
    """
    
    def __init__(self, dataloader, seed=None):
        self.dataloader = dataloader
        self.seed = seed
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
    
    def __iter__(self):
        for batch in self.dataloader:
            # 🔥 兼容两种batch格式
            if 'sequence_data' in batch:
                # 嵌套格式
                images = batch['sequence_data']['images']
                joints = batch['sequence_data']['joints']
                labels = batch['sequence_data']['labels']
                timestamps = batch['sequence_data'].get('timestamps', None)
            else:
                # 扁平格式
                images = batch['images']
                joints = batch['joints']
                labels = batch['labels']
                timestamps = batch.get('timestamps', None)
            
            # 获取batch size
            batch_size = images.shape[0]
            
            # 生成随机排列
            perm = torch.randperm(batch_size, generator=self.rng)
            
            # 🔥 返回扁平格式（与原始DataLoader一致）
            shuffled_batch = {
                'images': images,          # 保持原样
                'joints': joints[perm],    # 🔥 打乱
                'labels': labels
            }
            if timestamps is not None:
                shuffled_batch['timestamps'] = timestamps
            
            yield shuffled_batch
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def dataset(self):
        return self.dataloader.dataset


class ShuffledTemporalWrapper:
    """
    Shuffled Temporal消融: 打乱每个样本内部的时序
    
    在时序维度随机打乱joints，破坏时序同步
    保留: 样本身份（同一场景）
    破坏: 时序对应
    """
    
    def __init__(self, dataloader, seed=None):
        self.dataloader = dataloader
        self.seed = seed
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
    
    def __iter__(self):
        for batch in self.dataloader:
            # 🔥 兼容两种batch格式
            if 'sequence_data' in batch:
                # 嵌套格式
                images = batch['sequence_data']['images']
                joints = batch['sequence_data']['joints']
                labels = batch['sequence_data']['labels']
                timestamps = batch['sequence_data'].get('timestamps', None)
            else:
                # 扁平格式
                images = batch['images']
                joints = batch['joints']
                labels = batch['labels']
                timestamps = batch.get('timestamps', None)
            
            batch_size, seq_len = images.shape[:2]
            
            # 每个样本独立打乱时序
            shuffled_joints = []
            for i in range(batch_size):
                # 为每个样本生成独立的随机排列
                perm = torch.randperm(seq_len, generator=self.rng)
                shuffled_joints.append(joints[i, perm])
            
            # 🔥 返回扁平格式（与原始DataLoader一致）
            shuffled_batch = {
                'images': images,  # 保持原样
                'joints': torch.stack(shuffled_joints, dim=0),  # 🔥 时序打乱
                'labels': labels
            }
            if timestamps is not None:
                shuffled_batch['timestamps'] = timestamps
            
            yield shuffled_batch
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def dataset(self):
        return self.dataloader.dataset


def wrap_dataloader(dataloader, ablation_type, seed=None):
    """
    根据消融类型包装dataloader
    
    Args:
        dataloader: 原始dataloader
        ablation_type: 消融类型
        seed: 随机种子
    
    Returns:
        包装后的dataloader或原始dataloader
    """
    if ablation_type == 'shuffled_batch':
        return ShuffledBatchWrapper(dataloader, seed=seed)
    elif ablation_type == 'shuffled_temporal':
        return ShuffledTemporalWrapper(dataloader, seed=seed)
    else:
        # 其他消融不需要修改数据
        return dataloader


if __name__ == "__main__":
    """测试数据包装器"""
    print("=== 测试数据包装器 ===\n")
    
    # 创建模拟数据
    class DummyDataset:
        def __init__(self, num_samples=10):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'sequence_data': {
                    'images': torch.randn(11, 3, 224, 224),
                    'joints': torch.randn(11, 7),
                    'timestamps': torch.randn(11),
                    'labels': torch.randint(0, 11, (11,))
                }
            }
    
    dataset = DummyDataset(num_samples=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 测试原始dataloader
    print("原始DataLoader:")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"  Batch {i}: images shape = {batch['sequence_data']['images'].shape}")
            print(f"  Batch {i}: joints shape = {batch['sequence_data']['joints'].shape}")
            print(f"  Batch {i}: joints[0,0,:3] = {batch['sequence_data']['joints'][0,0,:3]}")
            break
    
    # 测试Shuffled Batch
    print("\nShuffled Batch Wrapper:")
    shuffled_batch_loader = ShuffledBatchWrapper(dataloader, seed=42)
    for i, batch in enumerate(shuffled_batch_loader):
        if i == 0:
            # 🔥 Wrapper输出是扁平格式
            print(f"  Batch {i}: images shape = {batch['images'].shape}")
            print(f"  Batch {i}: joints shape = {batch['joints'].shape}")
            print(f"  Batch {i}: joints[0,0,:3] = {batch['joints'][0,0,:3]}")
            print("  (注意: joints已在batch维度打乱)")
            break
    
    # 测试Shuffled Temporal
    print("\nShuffled Temporal Wrapper:")
    shuffled_temporal_loader = ShuffledTemporalWrapper(dataloader, seed=42)
    for i, batch in enumerate(shuffled_temporal_loader):
        if i == 0:
            # 🔥 Wrapper输出是扁平格式
            print(f"  Batch {i}: images shape = {batch['images'].shape}")
            print(f"  Batch {i}: joints shape = {batch['joints'].shape}")
            print(f"  Batch {i}: joints[0,:3,0] = {batch['joints'][0,:3,0]}")
            print("  (注意: joints已在时序维度打乱)")
            break
    
    # 测试wrap_dataloader函数
    print("\n测试wrap_dataloader函数:")
    for ablation in ['full_model', 'shuffled_batch', 'shuffled_temporal']:
        wrapped = wrap_dataloader(dataloader, ablation, seed=42)
        print(f"  {ablation}: {type(wrapped).__name__}")
    
    print("\n=== 测试完成 ===")