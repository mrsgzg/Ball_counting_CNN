import subprocess
import time

# 外层循环控制 n_poch 从 1 到 10
for n_poch in range(2, 11):
    dir_name = f"results_{n_poch}"
    command = f"python main.py --data_dir /home/embody_data/raw --model_type simple --visualize --epochs {n_poch} --save_dir {dir_name}"
    
    # 内层循环执行每个 n_poch 的 10 次重复
    for i in range(10):
        print(f"Epoch {n_poch}, Execution {i+1}/10 starting...")
        
        # Execute the command
        process = subprocess.run(command, shell=True, cwd="/home/Ball_counting_CNN")
        
        # Print the completion status
        if process.returncode == 0:
            print(f"Epoch {n_poch}, Execution {i+1}/10 completed successfully.")
        else:
            print(f"Epoch {n_poch}, Execution {i+1}/10 failed with return code {process.returncode}.")
        
        # Add a small delay between executions (optional)
        if i < 9:  # Don't sleep after the last iteration
            time.sleep(1)

    print(f"All executions for epoch {n_poch} completed.")

print("All epochs and executions completed.")