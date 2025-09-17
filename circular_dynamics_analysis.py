"""
环形动力学分析工具
用于分析LSTM隐状态的旋转型attractor结构
支持jointPCA和环形模式检测
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import defaultdict

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CircularDynamicsAnalyzer:
    """环形动力学分析器"""
    
    def __init__(self):
        self.joint_pca = None
        self.trajectories_by_label = defaultdict(list)
        self.analysis_results = {}
        
    def add_trajectory(self, lstm_states, label, sample_id):
        """添加一个样本的LSTM轨迹"""
        # 将LSTM状态转换为numpy数组
        if isinstance(lstm_states, list):
            trajectory = torch.stack(lstm_states).cpu().numpy()
        else:
            trajectory = lstm_states
            
        if len(trajectory.shape) > 2:
            trajectory = trajectory.squeeze(1)
            
        self.trajectories_by_label[label].append({
            'trajectory': trajectory,
            'sample_id': sample_id
        })
    
    def compute_joint_pca(self, n_components=3, labels_to_analyze=None):
        """计算跨样本的jointPCA"""
        if labels_to_analyze is None:
            labels_to_analyze = list(self.trajectories_by_label.keys())
        
        # 收集所有轨迹
        all_trajectories = []
        trajectory_info = []
        
        for label in labels_to_analyze:
            for traj_data in self.trajectories_by_label[label]:
                all_trajectories.append(traj_data['trajectory'])
                trajectory_info.append({
                    'label': label,
                    'sample_id': traj_data['sample_id'],
                    'length': len(traj_data['trajectory'])
                })
        
        # 合并所有轨迹点
        all_points = np.vstack(all_trajectories)
        
        # 执行PCA
        self.joint_pca = PCA(n_components=n_components)
        all_points_pca = self.joint_pca.fit_transform(all_points)
        
        # 重新分割轨迹
        trajectories_pca = []
        start_idx = 0
        for info in trajectory_info:
            end_idx = start_idx + info['length']
            trajectories_pca.append(all_points_pca[start_idx:end_idx])
            start_idx = end_idx
            
        # 保存结果
        self.analysis_results['joint_pca'] = {
            'trajectories_pca': trajectories_pca,
            'trajectory_info': trajectory_info,
            'explained_variance_ratio': self.joint_pca.explained_variance_ratio_,
            'total_variance': self.joint_pca.explained_variance_ratio_.sum()
        }
        
        return trajectories_pca, trajectory_info
    
    def analyze_circularity(self, trajectory_2d):
        """分析2D轨迹的环形特性"""
        # 计算质心
        center = np.mean(trajectory_2d, axis=0)
        
        # 转换到极坐标
        centered = trajectory_2d - center
        radii = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        
        # 展开角度（处理2π跳变）
        angles_unwrapped = np.unwrap(angles)
        
        # 计算环形指标
        metrics = {}
        
        # 1. 角速度
        angular_velocities = np.diff(angles_unwrapped)
        metrics['mean_angular_velocity'] = np.mean(angular_velocities)
        metrics['std_angular_velocity'] = np.std(angular_velocities)
        metrics['angular_consistency'] = 1.0 - (metrics['std_angular_velocity'] / 
                                               (np.abs(metrics['mean_angular_velocity']) + 1e-8))
        
        # 2. 半径稳定性
        metrics['mean_radius'] = np.mean(radii)
        metrics['std_radius'] = np.std(radii)
        metrics['radius_stability'] = 1.0 - (metrics['std_radius'] / 
                                            (metrics['mean_radius'] + 1e-8))
        
        # 3. 旋转方向（顺时针-1，逆时针+1）
        metrics['rotation_direction'] = np.sign(metrics['mean_angular_velocity'])
        
        # 4. 完成的旋转圈数
        total_rotation = angles_unwrapped[-1] - angles_unwrapped[0]
        metrics['total_rotations'] = total_rotation / (2 * np.pi)
        
        # 5. 环形度评分（综合指标）
        metrics['circularity_score'] = (metrics['angular_consistency'] + 
                                       metrics['radius_stability']) / 2
        
        # 6. 轨迹闭合度
        start_end_distance = np.linalg.norm(trajectory_2d[0] - trajectory_2d[-1])
        metrics['closure_ratio'] = 1.0 - (start_end_distance / 
                                         (metrics['mean_radius'] * 2 + 1e-8))
        
        return metrics, radii, angles_unwrapped, center
    
    def detect_rotation_patterns(self, min_circularity=0.5):
        """检测所有轨迹中的旋转模式"""
        if 'joint_pca' not in self.analysis_results:
            print("请先运行compute_joint_pca()")
            return None
        
        trajectories_pca = self.analysis_results['joint_pca']['trajectories_pca']
        trajectory_info = self.analysis_results['joint_pca']['trajectory_info']
        
        rotation_patterns = []
        
        for i, (traj_pca, info) in enumerate(zip(trajectories_pca, trajectory_info)):
            # 只分析PC1-PC2平面
            traj_2d = traj_pca[:, :2]
            
            # 分析环形特性
            metrics, radii, angles, center = self.analyze_circularity(traj_2d)
            
            pattern = {
                'label': info['label'],
                'sample_id': info['sample_id'],
                'metrics': metrics,
                'trajectory_2d': traj_2d,
                'radii': radii,
                'angles': angles,
                'center': center
            }
            
            # 判断是否为显著的环形模式
            if metrics['circularity_score'] >= min_circularity:
                pattern['is_circular'] = True
            else:
                pattern['is_circular'] = False
                
            rotation_patterns.append(pattern)
        
        self.analysis_results['rotation_patterns'] = rotation_patterns
        return rotation_patterns
    
    def plot_joint_trajectories(self, labels_to_plot=None, save_path=None):
        """绘制jointPCA轨迹"""
        if 'joint_pca' not in self.analysis_results:
            print("请先运行compute_joint_pca()")
            return
            
        trajectories_pca = self.analysis_results['joint_pca']['trajectories_pca']
        trajectory_info = self.analysis_results['joint_pca']['trajectory_info']
        explained_var = self.analysis_results['joint_pca']['explained_variance_ratio']
        
        if labels_to_plot is None:
            labels_to_plot = list(set(info['label'] for info in trajectory_info))
        
        # 创建图形
        fig = plt.figure(figsize=(18, 6))
        
        # 1. PC1-PC2平面
        ax1 = fig.add_subplot(131)
        self._plot_2d_trajectories(ax1, trajectories_pca, trajectory_info, 
                                  labels_to_plot, dims=(0, 1), explained_var=explained_var)
        
        # 2. PC1-PC3平面
        ax2 = fig.add_subplot(132)
        self._plot_2d_trajectories(ax2, trajectories_pca, trajectory_info, 
                                  labels_to_plot, dims=(0, 2), explained_var=explained_var)
        
        # 3. 3D图
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_3d_trajectories(ax3, trajectories_pca, trajectory_info, 
                                  labels_to_plot, explained_var=explained_var)
        
        plt.suptitle('Joint PCA Trajectories Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_circular_analysis(self, min_circularity=0.5, save_path=None):
        """绘制环形分析结果"""
        if 'rotation_patterns' not in self.analysis_results:
            patterns = self.detect_rotation_patterns(min_circularity)
        else:
            patterns = self.analysis_results['rotation_patterns']
        
        # 筛选环形轨迹
        circular_patterns = [p for p in patterns if p['is_circular']]
        
        if not circular_patterns:
            print("没有检测到显著的环形模式")
            return
        
        # 创建图形
        n_patterns = min(6, len(circular_patterns))  # 最多显示6个
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_patterns):
            pattern = circular_patterns[i]
            ax = axes[i]
            
            # 绘制轨迹
            traj = pattern['trajectory_2d']
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6, linewidth=2)
            
            # 用颜色表示时间
            scatter = ax.scatter(traj[:, 0], traj[:, 1], 
                               c=range(len(traj)), cmap='viridis', 
                               s=30, alpha=0.8)
            
            # 标记起点和终点
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=100, 
                      marker='o', edgecolor='black', linewidth=2, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100, 
                      marker='*', edgecolor='black', linewidth=2, label='End')
            
            # 绘制中心
            center = pattern['center']
            ax.scatter(center[0], center[1], color='black', s=50, 
                      marker='+', linewidth=2)
            
            # 添加信息
            metrics = pattern['metrics']
            info_text = f"Label: {pattern['label']}\n"
            info_text += f"Circularity: {metrics['circularity_score']:.2f}\n"
            info_text += f"Rotations: {metrics['total_rotations']:.1f}\n"
            info_text += f"Direction: {'CCW' if metrics['rotation_direction'] > 0 else 'CW'}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            ax.set_aspect('equal')
            ax.set_title(f"Sample {pattern['sample_id']}")
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_patterns, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Circular Patterns (Circularity > {min_circularity})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_rotation_metrics_summary(self, save_path=None):
        """绘制旋转指标汇总"""
        if 'rotation_patterns' not in self.analysis_results:
            patterns = self.detect_rotation_patterns()
        else:
            patterns = self.analysis_results['rotation_patterns']
        
        # 按标签分组统计
        metrics_by_label = defaultdict(list)
        for pattern in patterns:
            label = pattern['label']
            metrics = pattern['metrics']
            metrics_by_label[label].append(metrics)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 环形度分布
        ax1 = axes[0, 0]
        # 准备数据用于boxplot
        circularity_data = []
        labels_for_plot = []
        
        for label in sorted(metrics_by_label.keys()):
            scores = [m['circularity_score'] for m in metrics_by_label[label]]
            if scores:  # 只添加非空的数据
                circularity_data.append(scores)
                labels_for_plot.append(label)
        
        if circularity_data:
            ax1.boxplot(circularity_data, labels=labels_for_plot)
        ax1.set_xlabel('Count Label')
        ax1.set_ylabel('Circularity Score')
        ax1.set_title('Circularity Score by Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. 旋转圈数
        ax2 = axes[0, 1]
        for label in sorted(metrics_by_label.keys()):
            rotations = [m['total_rotations'] for m in metrics_by_label[label]]
            if rotations:  # 只绘制非空数据
                ax2.scatter([label] * len(rotations), rotations, alpha=0.6, s=50)
        
        ax2.set_xlabel('Count Label')
        ax2.set_ylabel('Total Rotations')
        ax2.set_title('Number of Rotations by Count')
        ax2.grid(True, alpha=0.3)
        
        # 设置x轴范围以包含所有标签
        if metrics_by_label:
            ax2.set_xlim(min(metrics_by_label.keys()) - 0.5, 
                        max(metrics_by_label.keys()) + 0.5)
        
        # 3. 角速度一致性
        ax3 = axes[1, 0]
        mean_consistencies = []
        std_consistencies = []
        labels_sorted = sorted(metrics_by_label.keys())
        
        for label in labels_sorted:
            consistencies = [m['angular_consistency'] for m in metrics_by_label[label]]
            mean_consistencies.append(np.mean(consistencies))
            std_consistencies.append(np.std(consistencies))
        
        ax3.errorbar(labels_sorted, mean_consistencies, yerr=std_consistencies,
                    marker='o', capsize=5, linewidth=2, markersize=8)
        ax3.set_xlabel('Count Label')
        ax3.set_ylabel('Angular Consistency')
        ax3.set_title('Angular Velocity Consistency by Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. 旋转方向统计
        ax4 = axes[1, 1]
        ccw_counts = []
        cw_counts = []
        labels_sorted = sorted(metrics_by_label.keys())
        
        for label in labels_sorted:
            directions = [m['rotation_direction'] for m in metrics_by_label[label]]
            ccw_counts.append(sum(d > 0 for d in directions))
            cw_counts.append(sum(d < 0 for d in directions))
        
        if labels_sorted:  # 只在有数据时绘制
            x = np.arange(len(labels_sorted))
            width = 0.35
            
            ax4.bar(x - width/2, ccw_counts, width, label='CCW', alpha=0.8)
            ax4.bar(x + width/2, cw_counts, width, label='CW', alpha=0.8)
            ax4.set_xlabel('Count Label')
            ax4.set_ylabel('Number of Samples')
            ax4.set_title('Rotation Direction Distribution')
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels_sorted)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Rotation Metrics Summary', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_2d_trajectories(self, ax, trajectories_pca, trajectory_info, 
                             labels_to_plot, dims=(0, 1), explained_var=None):
        """辅助函数：绘制2D轨迹"""
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for traj, info in zip(trajectories_pca, trajectory_info):
            if info['label'] not in labels_to_plot:
                continue
                
            label = info['label']
            color = colors[label - 1]  # 假设标签从1开始
            
            # 绘制轨迹
            ax.plot(traj[:, dims[0]], traj[:, dims[1]], 
                   color=color, alpha=0.5, linewidth=1)
            
            # 标记起点
            ax.scatter(traj[0, dims[0]], traj[0, dims[1]], 
                      color=color, s=50, marker='o', edgecolor='black', linewidth=1)
        
        # 设置标签
        if explained_var is not None:
            ax.set_xlabel(f'PC{dims[0]+1} ({explained_var[dims[0]]:.1%})')
            ax.set_ylabel(f'PC{dims[1]+1} ({explained_var[dims[1]]:.1%})')
        else:
            ax.set_xlabel(f'PC{dims[0]+1}')
            ax.set_ylabel(f'PC{dims[1]+1}')
            
        ax.set_title(f'PC{dims[0]+1} vs PC{dims[1]+1}')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        handles = []
        for label in sorted(set(info['label'] for info in trajectory_info)):
            if label in labels_to_plot:
                handles.append(plt.Line2D([0], [0], color=colors[label-1], 
                                        linewidth=2, label=f'Count {label}'))
        ax.legend(handles=handles, loc='best')
    
    def _plot_3d_trajectories(self, ax, trajectories_pca, trajectory_info, 
                             labels_to_plot, explained_var=None):
        """辅助函数：绘制3D轨迹"""
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for traj, info in zip(trajectories_pca, trajectory_info):
            if info['label'] not in labels_to_plot:
                continue
                
            label = info['label']
            color = colors[label - 1]
            
            # 绘制轨迹
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=color, alpha=0.5, linewidth=1)
            
            # 标记起点
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                      color=color, s=50, marker='o', edgecolor='black', linewidth=1)
        
        # 设置标签
        if explained_var is not None:
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
            ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})')
        else:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
        ax.set_title('3D Joint PCA')
        ax.view_init(elev=20, azim=45)


# 便捷函数
def analyze_rotation_in_trajectories(lstm_states_dict, save_dir=None):
    """
    便捷函数：分析多个样本的旋转模式
    
    Args:
        lstm_states_dict: {(label, sample_id): lstm_states}
        save_dir: 保存目录
    
    Returns:
        analyzer: 包含所有分析结果的分析器对象
    """
    analyzer = CircularDynamicsAnalyzer()
    
    # 添加所有轨迹
    for (label, sample_id), lstm_states in lstm_states_dict.items():
        analyzer.add_trajectory(lstm_states, label, sample_id)
    
    # 执行分析
    print("执行jointPCA...")
    analyzer.compute_joint_pca(n_components=3)
    
    print("检测旋转模式...")
    patterns = analyzer.detect_rotation_patterns(min_circularity=0.5)
    
    # 统计结果
    circular_count = sum(1 for p in patterns if p['is_circular'])
    print(f"检测到 {circular_count}/{len(patterns)} 个环形轨迹")
    
    # 生成可视化
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        analyzer.plot_joint_trajectories(
            save_path=os.path.join(save_dir, 'joint_pca_trajectories.png')
        )
        
        analyzer.plot_circular_analysis(
            save_path=os.path.join(save_dir, 'circular_patterns.png')
        )
        
        analyzer.plot_rotation_metrics_summary(
            save_path=os.path.join(save_dir, 'rotation_metrics_summary.png')
        )
    
    return analyzer