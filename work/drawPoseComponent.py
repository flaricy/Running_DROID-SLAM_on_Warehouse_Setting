'''
    This script is used to plot the some components in poses of a camera trajectory,
    which may be helpful to indicate the correspondense between the xyz in pose and 
    the xyz in world coordinate.

    Please modify `traj_est.npy` file path in the 37-th line if you need.
'''

import numpy as np
import matplotlib.pyplot as plt

# 假设 poses 文件已加载，形状为 (N, 6) 或 (N, 7)
# 每一行 pose 包含 [x, y, z, rx, ry, rz] 或 [x, y, z, qw, qx, qy, qz]
def load_poses(poses_file_path):
    return np.load(poses_file_path)

def plot_poses(poses, indices=[0, 2]):
    """
    绘制指定位姿分量的变化曲线
    :param poses: 位姿数据，形状 (N, 6) 或 (N, 7)
    :param indices: 要绘制的分量索引，例如 [0, 2]
    """
    # 提取指定分量的曲线
    time_steps = np.arange(poses.shape[0])  # 时间步
    for idx in indices:
        plt.plot(time_steps, poses[:, idx], label=f"Pose[{idx}]")
    
    # 图形修饰
    plt.title("Pose Components Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# 示例使用
poses_file_path = "reconstructions/warehouse/traj_est.npy"  # 修改为实际位姿文件路径 # 替换为实际路径
poses = load_poses(poses_file_path)

# 绘制 pose[0], pose[2]
plot_poses(poses, indices=[0, 2])