import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def process_and_truncate_trajectory():
    """
    加载、重塑、截取并保存新的参考轨迹。
    """
    # --- 1. 加载 MAT 文件 ---
    try:
        mat_data = scipy.io.loadmat('Ref.mat')
        print("成功加载 'Ref.mat' 文件。")
    except FileNotFoundError:
        print("错误: 未找到 'Ref.mat' 文件。请确保此脚本和 .mat 文件在同一个目录下。")
        return

    # --- 2. 提取并重塑数据 ---
    # 兼容 'Ref' 或 'X_ref' 两种可能的键名
    if 'Ref' in mat_data:
        ref_vector_stacked = mat_data['Ref']
    elif 'X_ref' in mat_data:
        ref_vector_stacked = mat_data['X_ref']
    else:
        print("错误: 在 .mat 文件中未能找到 'Ref' 或 'X_ref' 数据。")
        return

    # 将数据展平为一维，然后重塑为 (5150, 2) 的形状
    try:
        original_ref_trajectory = ref_vector_stacked.flatten().reshape(-1, 2)
        print(f"原始轨迹已成功重塑为 shape: {original_ref_trajectory.shape}")
    except ValueError:
        print(f"错误: 无法将数据重塑为 (n, 2) 的形状。请检查数据总长度 {len(ref_vector_stacked.flatten())} 是否为偶数。")
        return

    # --- 3. 提取第 2500 个数据点之后的数据 ---
    start_index = 2300
    if original_ref_trajectory.shape[0] <= start_index:
        print(f"错误: 起始索引 {start_index} 超出了数据总行数 {original_ref_trajectory.shape[0]}。")
        return

    new_ref_trajectory = original_ref_trajectory[start_index:, :]
    print(f"已提取新的轨迹数据，shape 为: {new_ref_trajectory.shape}")

    # --- 4. 保存为新的 MAT 文件 ---
    new_mat_filename = 'Ref_truncated.mat'
    scipy.io.savemat(new_mat_filename, {'Ref': new_ref_trajectory})
    print(f"新的参考轨迹已保存到 '{new_mat_filename}' 文件中。")

    # --- 5. 可视化对比 ---
    ts = 0.02  # 假设采样时间为 0.02s
    original_time = np.arange(original_ref_trajectory.shape[0]) * ts
    new_time = np.arange(new_ref_trajectory.shape[0]) * ts + start_index * ts

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    ax1.plot(original_time, original_ref_trajectory[:, 0], label='Original q3_ref')
    ax1.plot(original_time, original_ref_trajectory[:, 1], label='Original q4_ref')
    ax1.axvline(x=start_index * ts, color='r', linestyle='--', label=f'截取点 t={start_index * ts:.1f}s')
    ax1.set_title('完整的原始参考轨迹')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('角度 (rad)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(new_time, new_ref_trajectory[:, 0], label='New q3_ref')
    ax2.plot(new_time, new_ref_trajectory[:, 1], label='New q4_ref')
    ax2.set_title('截取后的新参考轨迹 (从第2500个点开始)')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('角度 (rad)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    process_and_truncate_trajectory()