#!/bin/bash --login
#SBATCH -p gpuV              # v100 GPU 
#SBATCH -G 1                 # 请求 1 个 GPU
#SBATCH -n 4                 # 请求 4 个 CPU 核心
#SBATCH -t 6:00:00               # 最长运行时间 1 天
#SBATCH --mem=10G            # 请求 10GB 内存

# 显示作业基本信息
echo "=== 作业信息 ==="
echo "节点: $SLURMD_NODENAME"
echo "GPU: $SLURM_GPUS (ID: $CUDA_VISIBLE_DEVICES)"
echo "CPU核心: $SLURM_NTASKS"
echo "内存: $SLURM_MEM_PER_NODE MB"
echo "开始时间: $(date)"

# 激活conda环境
echo "=== 激活环境 ==="
#source /opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate cgtest

# 直接使用环境Python路径（确保可靠）
#PYTHON_PATH="/mnt/iusers01/fatpou01/compsci01/k09562zs/.conda/envs/cgtest/bin/python"

# 验证环境
echo "Python版本: $($PYTHON_PATH --version)"
echo "当前环境: $CONDA_DEFAULT_ENV"
# 运行程序
echo "=== 开始训练 ==="
cd /mnt/iusers01/fatpou01/compsci01/k09562zs
python scratch/Ball_counting_CNN/Main_embodiment_ablation.py \
--model_type 'visual_only' \
--data_root '/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection' \
--train_csv 'scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_20.csv' \
--val_csv 'scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv' \
--batch_size 16 \
--learning_rate 1e-4 \
--weight_decay 1e-5 \
--total_epochs 350 \
--attention_heads 1 \
--image_mode 'rgb' \
--scheduler_type 'none' \
--save_dir ./scratch/Ball_counting_CNN/Final_result/Embodiment_Visual_only/20data/check_points \
--log_dir ./scratch/Ball_counting_CNN/Final_result/Embodiment_Visual_only/20data/logs \
--save_every 2 \
--num_workers 4

echo "=== 完成 ==="
echo "结束时间: $(date)"
