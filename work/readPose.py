'''
    This script print poses from `reconstructions/warehouse/traj_est.npy`.
    You can prescribe whether print 7-tuples (x, y, z, qx, qy, qz, qw) or 4x4 matrix,
    in the 77-th line.
    You can also precribe files to analyze in the 66-th line;
'''

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def load_poses(file_path):
    """
    加载保存的位姿数据，假定每行是一个 7 元组 (x, y, z, qx, qy, qz, qw)
    :param file_path: 位姿文件路径 (txt 或 npy)
    :return: numpy 数组 (N, 7)
    """
    if file_path.endswith('.npy'):
        poses = np.load(file_path)
    elif file_path.endswith('.txt'):
        poses = []
        with open(file_path, 'r') as f:
            for line in f:
                poses.append(list(map(float, line.strip().split())))
        poses = np.array(poses)
    else:
        raise ValueError("Unsupported file format. Please use .npy or .txt")
    
    return poses[:10]


def pose_to_matrix(pose):
    """
    将 7 元组 (x, y, z, qx, qy, qz, qw) 转换为 4x4 的齐次变换矩阵
    :param pose: 长度为 7 的数组
    :return: 4x4 的齐次变换矩阵
    """
    translation = pose[:3]  # 平移向量 (x, y, z)
    quaternion = pose[3:]   # 四元数 (qx, qy, qz, qw)
    rotation_matrix = R.from_quat(quaternion).as_matrix()  # 四元数转旋转矩阵
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix



def display_poses(poses, as_matrix=False, step=1):
    """
    打印位姿数据
    :param poses: 位姿数组 (N, 7)
    :param as_matrix: 是否以 4x4 矩阵形式显示
    :param step: 每隔几帧打印一帧
    """
    print(f"Total number of poses: {len(poses)}")
    for i in range(0, len(poses), step):
        print(f"\nPose {i}:")
        if as_matrix:
            matrix = pose_to_matrix(poses[i])
            print(matrix)
        else:
            print(poses[i])

def main():
    # 指定位姿文件路径
    file_path = "reconstructions/warehouse/traj_est.npy" # 修改为实际路径

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return

    # 读取位姿文件
    poses = load_poses(file_path)

    # 查看和打印位姿
    display_poses(poses, as_matrix=False, step=1)  # 每隔 1 帧显示

if __name__ == "__main__":
    main()