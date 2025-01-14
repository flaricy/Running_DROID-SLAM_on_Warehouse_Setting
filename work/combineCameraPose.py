'''
    This script marks camera trajectory in the original point cloud. The trajectory is by 
    default colored red. 
    Please ensure that the original ply file exists, which should be put at 
    `work/<setting>/point_clouds/all_frames.ply`, where setting is what you prescribed in 
    commandline arguments.
'''

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import argparse
import os 

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

def load_poses(file_path):
    """
    加载位姿文件，假定每行是一个 7 元组 (x, y, z, qx, qy, qz, qw)
    :param file_path: 位姿文件路径 (.npy 文件)
    :return: numpy 数组 (N, 7)
    """
    poses = np.load(file_path)
    return poses

def add_camera_positions_to_point_cloud(ply_file_path, poses_file_path, output_path):
    """
    在点云中添加相机位置，并保存更新后的点云
    :param ply_file_path: 点云文件路径 (.ply)
    :param poses_file_path: 相机位姿文件路径 (.npy)
    :param output_path: 保存结果的点云文件路径
    """
    # 加载点云
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    if not point_cloud:
        raise FileNotFoundError(f"Point cloud file not found: {ply_file_path}")

    # 加载相机位姿
    poses = load_poses(poses_file_path)

    # 提取相机位置 (x, y, z)
    camera_positions = poses[:, :3]

    # 创建相机位置点云
    camera_points = o3d.geometry.PointCloud()
    camera_points.points = o3d.utility.Vector3dVector(camera_positions)

    # 设置相机点为红色
    num_camera_points = camera_positions.shape[0]
    camera_points.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (num_camera_points, 1)))  # 红色

    # 合并原始点云和相机位置点云
    combined_cloud = point_cloud + camera_points

    # 保存合并后的点云
    o3d.io.write_point_cloud(output_path, combined_cloud)
    print(f"Combined point cloud with camera positions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", help="which setting to load", default="warehouse")
    args = parser.parse_args()
    
    setting = args.setting
    # 文件路径
    ply_file_path = f"work/{setting}/point_clouds/all_frames.ply"  # 修改为实际点云文件路径
    poses_file_path = f"reconstructions/{setting}/traj_est.npy"  # 修改为实际位姿文件路径
    output_path = f"work/{setting}/point_clouds/combined.ply"  # 输出图片目录
    
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 在点云中添加相机位置
    add_camera_positions_to_point_cloud(ply_file_path, poses_file_path, output_path)