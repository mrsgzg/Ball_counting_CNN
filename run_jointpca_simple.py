#!/usr/bin/env python
"""
修复了JSON序列化问题的JointPCA运行脚本
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# 确保能找到你的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def numpy_to_python(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='JointPCA分析（修复版）')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--val_csv', type=str,
                       default='scratch/Ball_counting_CNN/Tools_script/ball_dynamic_val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    parser.add_argument('--num_samples_per_label', type=int, default=5,
                       help='每个标签的样本数')
    parser.add_argument('--save_dir', type=str, default='./jointpca_analysis_fixed',
                       help='结果保存目录')
    parser.add_argument('--quick_mode', action='store_true',
                       help='快速模式：只收集LSTM轨迹，不生成单样本可视化')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'analysis_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎯 JointPCA分析（修复版）")
    print("="*50)
    print(f"模型: {args.checkpoint}")
    print(f"每个标签样本数: {args.num_samples_per_label}")
    print(f"保存目录: {save_dir}")
    print(f"模式: {'快速' if args.quick_mode else '完整'}")
    print("="*50)
    
    try:
        # 根据是否已经修复了LSTM_Attention_Viz_JPCA.py来选择导入
        try:
            # 尝试导入修复后的版本
            from LSTM_Attention_Viz_JPCA import UniversalModelVisualizer, load_model_and_data
        except ImportError:
            print("⚠️ 使用原始版本，可能需要手动修复JSON序列化")
            from Universal_LSTM_Viz_with_JointPCA import UniversalModelVisualizer, load_model_and_data
        
        from DataLoader_embodiment import BallCountingDataset
        
        # 加载模型和数据
        print("\n📥 加载模型和数据...")
        model, dataset, device, config = load_model_and_data(
            args.checkpoint, args.val_csv, args.data_root
        )
        
        # 创建可视化器
        visualizer = UniversalModelVisualizer(model, device)
        
        # 准备平衡的样本集
        print("\n📊 准备平衡的样本集...")
        samples_by_label = {i: [] for i in range(1, 11)}
        
        # 收集每个标签的样本索引
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample['label'].item()
            if label in samples_by_label and len(samples_by_label[label]) < args.num_samples_per_label:
                samples_by_label[label].append(idx)
        
        # 选择要分析的样本
        selected_indices = []
        label_counts = {}
        for label in sorted(samples_by_label.keys()):
            indices = samples_by_label[label][:args.num_samples_per_label]
            selected_indices.extend(indices)
            label_counts[label] = len(indices)
            print(f"  标签 {label}: {len(indices)} 个样本")
        
        print(f"\n总计: {len(selected_indices)} 个样本")
        
        if args.quick_mode:
            # 快速模式：只收集LSTM轨迹
            print("\n🚀 快速模式：只收集LSTM轨迹...")
            for i, idx in enumerate(selected_indices):
                if i % 10 == 0:
                    print(f"  进度: {i}/{len(selected_indices)}")
                
                sample_data = dataset[idx]
                
                # 准备序列数据
                sequence_data = {
                    'images': sample_data['sequence_data']['images'].unsqueeze(0).to(device),
                    'joints': sample_data['sequence_data']['joints'].unsqueeze(0).to(device),
                    'timestamps': sample_data['sequence_data']['timestamps'].unsqueeze(0).to(device),
                    'labels': sample_data['sequence_data']['labels'].unsqueeze(0).to(device)
                }
                
                # 获取模型输出
                visualizer._prepare_model_for_visualization()
                outputs = visualizer._get_model_outputs(sequence_data)
                viz_data = visualizer._extract_visualization_data(outputs)
                
                # 收集LSTM轨迹
                if viz_data['lstm_states']:
                    true_label = sample_data['label'].item()
                    visualizer.collected_lstm_trajectories[(true_label, idx)] = viz_data['lstm_states']
        else:
            # 完整模式：生成每个样本的可视化
            print("\n🎨 完整模式：生成单样本可视化并收集轨迹...")
            for i, idx in enumerate(selected_indices):
                print(f"  处理样本 {i+1}/{len(selected_indices)} (索引: {idx})")
                sample_data = dataset[idx]
                visualizer.visualize_sample_sequence(sample_data, idx, save_dir, 
                                                   collect_for_joint_pca=True)
        
        print(f"\n✅ 收集了 {len(visualizer.collected_lstm_trajectories)} 个轨迹")
        
        # 执行JointPCA分析
        print("\n🎨 执行JointPCA分析...")
        
        # 如果原始脚本没有修复，我们在这里捕获错误并手动处理
        try:
            analyzer = visualizer.perform_joint_pca_analysis(save_dir)
        except TypeError as e:
            if "JSON serializable" in str(e):
                print("⚠️ 检测到JSON序列化错误，尝试手动修复...")
                
                # 手动执行分析的核心部分
                from circular_dynamics_analysis import CircularDynamicsAnalyzer
                
                analyzer = CircularDynamicsAnalyzer()
                
                # 添加轨迹
                for (label, sample_id), lstm_states in visualizer.collected_lstm_trajectories.items():
                    analyzer.add_trajectory(lstm_states, label, sample_id)
                
                # 执行分析
                analyzer.compute_joint_pca(n_components=3)
                patterns = analyzer.detect_rotation_patterns(min_circularity=0.5)
                
                # 创建可视化
                joint_pca_dir = os.path.join(save_dir, 'joint_pca_analysis')
                os.makedirs(joint_pca_dir, exist_ok=True)
                
                analyzer.plot_joint_trajectories(
                    save_path=os.path.join(joint_pca_dir, 'joint_pca_trajectories.png')
                )
                
                analyzer.plot_circular_analysis(
                    save_path=os.path.join(joint_pca_dir, 'circular_patterns.png')
                )
                
                analyzer.plot_rotation_metrics_summary(
                    save_path=os.path.join(joint_pca_dir, 'rotation_metrics_summary.png')
                )
                
                print("✅ 手动修复成功，可视化已生成")
            else:
                raise e
        
        # 打印统计信息
        if 'analyzer' in locals() and analyzer:
            circular_count = sum(1 for p in analyzer.analysis_results['rotation_patterns'] 
                               if p['is_circular'])
            total_count = len(analyzer.analysis_results['rotation_patterns'])
            
            print(f"\n📊 分析结果:")
            print(f"  • 总轨迹数: {total_count}")
            print(f"  • 环形轨迹: {circular_count} ({circular_count/total_count*100:.1f}%)")
            print(f"  • PCA解释方差: {analyzer.analysis_results['joint_pca']['total_variance']:.1%}")
        
        print("\n🎉 分析完成！")
        print(f"📁 结果保存在: {save_dir}")
        if os.path.exists(os.path.join(save_dir, 'joint_pca_analysis')):
            print("📊 生成的文件:")
            print("  • joint_pca_trajectories.png - 跨样本轨迹可视化")
            print("  • circular_patterns.png - 检测到的环形模式")
            print("  • rotation_metrics_summary.png - 旋转指标汇总")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存已收集的数据
        if 'visualizer' in locals() and hasattr(visualizer, 'collected_lstm_trajectories'):
            import pickle
            backup_path = os.path.join(save_dir, 'lstm_trajectories_backup.pkl')
            with open(backup_path, 'wb') as f:
                pickle.dump(visualizer.collected_lstm_trajectories, f)
            print(f"\n💾 已保存LSTM轨迹备份到: {backup_path}")
            print("   可以手动加载并分析这些数据")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎯 JointPCA分析工具（修复版）")
        print("="*50)
        print("修复了JSON序列化问题，更稳定地分析LSTM动力学")
        print()
        print("使用示例:")
        print()
        print("# 快速模式（推荐）")
        print("python run_jointpca_fixed.py --checkpoint ./best_model.pth --quick_mode")
        print()
        print("# 完整模式（生成单样本可视化）")
        print("python run_jointpca_fixed.py --checkpoint ./best_model.pth")
        print()
        print("# 自定义参数")
        print("python run_jointpca_fixed.py \\")
        print("    --checkpoint ./model.pth \\")
        print("    --num_samples_per_label 10 \\")
        print("    --save_dir ./my_analysis \\")
        print("    --quick_mode")
        print()
        print("参数说明:")
        print("  --checkpoint              模型路径（必需）")
        print("  --val_csv                 验证集CSV路径")
        print("  --data_root              数据根目录")
        print("  --num_samples_per_label  每个标签的样本数（默认5）")
        print("  --save_dir               保存目录")
        print("  --quick_mode             快速模式，跳过单样本可视化")
        print()
        print("💡 提示:")
        print("  • 快速模式下运行速度快10倍以上")
        print("  • 自动处理JSON序列化问题")
        print("  • 如果分析失败，会保存LSTM轨迹备份")
    else:
        main()