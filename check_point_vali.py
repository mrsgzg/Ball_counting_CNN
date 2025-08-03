"""
Checkpoint验证工具
用于检查PyTorch checkpoint文件是否完整和可用
"""

import torch
import os
import sys
import argparse
from datetime import datetime

def validate_checkpoint(checkpoint_path, verbose=True):
    """验证checkpoint文件是否完整"""
    if verbose:
        print(f"验证checkpoint: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        if verbose:
            print(f"❌ 文件不存在: {checkpoint_path}")
        return False, "文件不存在"
    
    # 检查文件大小
    file_size = os.path.getsize(checkpoint_path)
    if verbose:
        print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1024:  # 小于1KB通常是有问题的
        error_msg = f"文件大小异常小 ({file_size} bytes)，可能损坏"
        if verbose:
            print(f"⚠️ {error_msg}")
        return False, error_msg
    
    # 尝试加载checkpoint
    try:
        if verbose:
            print("🔄 尝试加载checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if verbose:
            print("✅ checkpoint加载成功")
        
        # 检查必要的键
        required_keys = ['model_state_dict', 'config']
        missing_keys = []
        
        for key in required_keys:
            if key not in checkpoint:
                missing_keys.append(key)
        
        if missing_keys:
            error_msg = f"缺少必要的键: {missing_keys}"
            if verbose:
                print(f"⚠️ {error_msg}")
            return False, error_msg
        
        # 检查模型状态字典
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict) or len(model_state) == 0:
            error_msg = "model_state_dict无效或为空"
            if verbose:
                print(f"❌ {error_msg}")
            return False, error_msg
        
        if verbose:
            print(f"✅ 模型参数数量: {len(model_state)}")
        
        # 检查配置
        config = checkpoint['config']
        if not isinstance(config, dict):
            error_msg = "config无效"
            if verbose:
                print(f"❌ {error_msg}")
            return False, error_msg
        
        if verbose:
            print(f"✅ 配置信息: {list(config.keys())}")
        
        # 显示一些关键信息
        if verbose:
            print(f"📊 Checkpoint信息:")
            print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  - 图像模式: {config.get('image_mode', 'N/A')}")
            print(f"  - 模型类型: {config.get('model_config', {}).get('cnn_layers', 'N/A')} 层CNN")
            print(f"  - 最佳验证准确率: {checkpoint.get('best_val_accuracy', 'N/A')}")
        
        return True, "checkpoint验证成功"
        
    except Exception as e:
        error_msg = f"加载checkpoint失败: {e}"
        if verbose:
            print(f"❌ {error_msg}")
            print(f"错误类型: {type(e).__name__}")
        return False, error_msg


def find_valid_checkpoints(directory, pattern="*.pth"):
    """在目录中查找有效的checkpoint文件"""
    import glob
    
    print(f"🔍 在目录中查找checkpoint文件: {directory}")
    
    # 查找所有.pth文件
    checkpoint_files = glob.glob(os.path.join(directory, pattern))
    
    if not checkpoint_files:
        print(f"❌ 未找到任何checkpoint文件")
        return []
    
    print(f"📁 找到 {len(checkpoint_files)} 个潜在checkpoint文件")
    
    valid_checkpoints = []
    
    for checkpoint_path in checkpoint_files:
        print(f"\n检查: {os.path.basename(checkpoint_path)}")
        is_valid, message = validate_checkpoint(checkpoint_path, verbose=False)
        
        if is_valid:
            print(f"✅ 有效")
            valid_checkpoints.append(checkpoint_path)
        else:
            print(f"❌ 无效: {message}")
    
    return valid_checkpoints


def repair_checkpoint_suggestions(checkpoint_path):
    """提供修复checkpoint的建议"""
    print(f"\n🔧 修复建议 for {checkpoint_path}:")
    
    # 检查文件是否损坏
    try:
        with open(checkpoint_path, 'rb') as f:
            # 尝试读取前几个字节
            header = f.read(10)
            if len(header) < 10:
                print("1. 文件太小，可能是保存时中断")
                print("   建议：重新训练模型或使用backup")
                return
    except Exception as e:
        print(f"1. 文件读取错误: {e}")
        print("   建议：检查文件权限或磁盘空间")
        return
    
    # 其他建议
    print("2. 如果是训练刚开始就保存的checkpoint，可能模型还未充分训练")
    print("   建议：使用训练更多epoch后保存的checkpoint")
    
    print("3. 检查是否有其他可用的checkpoint文件")
    directory = os.path.dirname(checkpoint_path)
    if directory:
        valid_checkpoints = find_valid_checkpoints(directory)
        if valid_checkpoints:
            print(f"   发现 {len(valid_checkpoints)} 个有效checkpoint:")
            for cp in valid_checkpoints:
                print(f"   - {cp}")
    
    print("4. 如果所有checkpoint都损坏，需要重新训练模型")


def main():
    parser = argparse.ArgumentParser(description='验证PyTorch checkpoint文件')
    parser.add_argument('checkpoint', help='checkpoint文件路径')
    parser.add_argument('--repair', action='store_true', 
                       help='显示修复建议')
    parser.add_argument('--find-valid', action='store_true',
                       help='在同一目录中查找有效的checkpoint')
    
    args = parser.parse_args()
    
    print("🔍 Checkpoint验证工具")
    print("=" * 50)
    
    # 验证主checkpoint
    is_valid, message = validate_checkpoint(args.checkpoint)
    
    if not is_valid:
        print(f"\n❌ 验证失败: {message}")
        
        if args.repair:
            repair_checkpoint_suggestions(args.checkpoint)
        
        if args.find_valid:
            directory = os.path.dirname(args.checkpoint)
            if directory:
                valid_checkpoints = find_valid_checkpoints(directory)
                if valid_checkpoints:
                    print(f"\n✅ 建议使用以下有效checkpoint:")
                    for cp in valid_checkpoints:
                        print(f"   {cp}")
    else:
        print(f"\n✅ 验证成功: {message}")
        print("\n该checkpoint可以正常使用进行分析")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔍 Checkpoint验证工具")
        print("=" * 50)
        print("用法:")
        print("  python validate_checkpoint.py <checkpoint_path>")
        print("  python validate_checkpoint.py <checkpoint_path> --repair")
        print("  python validate_checkpoint.py <checkpoint_path> --find-valid")
        print()
        print("示例:")
        print("  python validate_checkpoint.py ./model.pth")
        print("  python validate_checkpoint.py ./model.pth --repair --find-valid")
    else:
        main()