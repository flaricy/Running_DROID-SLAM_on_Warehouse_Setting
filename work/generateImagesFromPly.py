'''
    This script output rendered frames in each predicted camera pose from the reconstructed point 
    cloud. Now, this file only supports `warehouse` setting, where there are specific image height
    and width, camera intrinsic matrix. Please be careful if you want to modify this file.

    The output images are saved in `work/screen_capture/warehouse`.

    Please make sure your open3d version is 0.16.0
'''

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os

def load_camera_intrinsics(file_path):
    """
    加载相机内参文件，包含 8 个数。
    :param file_path: 内参文件路径
    :return: K (3x3 内参矩阵), distortion_coeffs (畸变参数，长度为 4)
    """
    # 加载数据
    data = np.loadtxt(file_path)
    if len(data) != 8:
        raise ValueError("Camera intrinsic file must contain 8 values")

    # 提取内参和畸变参数
    fx, fy, cx, cy = data[:4]
    distortion_coeffs = data[4:]

    # 构造内参矩阵 K
    K = np.array([
        #[fx,  0, cx], 
        [fx,  0, 421],
       # [ 0, fy, cy],
        [0, fy, 246.5],
        [ 0,  0,  1]
    ])
    return K, distortion_coeffs, 421, 246.5#cx, cy


def set_camera_intrinsics(ctr, K, image_size):
    """
    设置相机内参到 Open3D 控制器
    :param ctr: Open3D ViewControl
    :param K: 相机内参矩阵 (3x3)
    :param image_size: 图像大小 (width, height)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    width, height = image_size
    print("info:", cx * 2, cy * 2, width, height)

    # 创建相机内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # 获取当前相机参数并更新内参
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    camera_params.intrinsic = intrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    

def pose_to_matrix(pose):
    """
    将 7 元组 (x, y, z, qx, qy, qz, qw) 转换为 4x4 的齐次变换矩阵
    :param pose: 长度为 7 的数组 (全局坐标系)
    :return: 4x4 的齐次变换矩阵
    """
    # 平移向量 (全局坐标系)
    translation = pose[:3]  

    # 四元数 (qx, qy, qz, qw)
    quaternion = pose[3:]

    # 计算旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    rotation_matrix = np.dot(rotation_matrix, np.diag([-1, 1, 1]))  # 反转绕 x 轴

    # 将全局坐标的变换矩阵反转，以适应局部坐标系
    rotation_matrix = rotation_matrix.T  # 旋转矩阵的转置
    translation = -np.dot(rotation_matrix, translation)  # 平移向量反转

    # 构建 4x4 的齐次变换矩阵
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def load_poses(file_path):
    """
    加载位姿文件，假定每行是一个 7 元组 (x, y, z, qx, qy, qz, qw)
    :param file_path: 位姿文件路径 (npy 文件)
    :return: numpy 数组 (N, 7)
    """
    poses = np.load(file_path)
    return poses


def render_point_cloud_from_poses(ply_file_path, poses_file_path, intrinsics_file_path, output_dir, image_size=(843, 494)):
    """
    渲染点云并根据给定的相机位姿生成图片
    :param ply_file_path: 点云文件路径 (.ply)
    :param poses_file_path: 位姿文件路径 (.npy)
    :param output_dir: 图片输出目录
    :param image_size: 图片大小 (width, height)
    """
    # 加载点云
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    if not point_cloud:
        raise FileNotFoundError(f"Point cloud file not found: {ply_file_path}")

    # 加载位姿
    poses = load_poses(poses_file_path)

    # 加载相机内参
    K, distortion_coeffs, _, _ = load_camera_intrinsics(intrinsics_file_path)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_size[0], height=image_size[1])
    vis.add_geometry(point_cloud)

    render_option = vis.get_render_option()
    render_option.point_size = 3  # 设置点的渲染大小

    # 获取视图控制器
    ctr = vis.get_view_control()

    # 设置相机内参
    set_camera_intrinsics(ctr, K, image_size)

    # 渲染每一帧
    for i, pose in enumerate(poses):
        # 转换位姿为 4x4 矩阵
        # print(f"{i}: pose = {pose}")
        extrinsic = pose_to_matrix(pose)

        # 获取相机参数并设置外参
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params)

        # 渲染当前帧并保存图片
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        print(f"Saved frame {i} to {output_path}")

    vis.destroy_window()
    print(f"All frames have been saved to {output_dir}")


if __name__ == "__main__":
    # 文件路径
    ply_file_path = "work/warehouse/point_clouds/all_frames.ply"  # 修改为实际点云文件路径
    poses_file_path = "reconstructions/warehouse/traj_est.npy"  # 修改为实际位姿文件路径
    output_dir = "work/screen_capture/rendered_frames/"  # 输出图片目录
    intrinsic_file_path = "calib/warehouse.txt"

    # camera intrinsics
    K, distortion_coeffs, height, width = load_camera_intrinsics(intrinsic_file_path)
    print("Camera Intrinsic Matrix (K):\n", K)
    print("Distortion Coefficients:", distortion_coeffs)

    # 渲染点云并生成图片
    render_point_cloud_from_poses(ply_file_path, poses_file_path, intrinsic_file_path, output_dir)