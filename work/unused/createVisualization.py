import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.spatial.transform import Rotation as R
import open3d.visualization.rendering as rendering

def create_camera_trajectory(poses):
    # 将四元数转换为 4x4 的位姿矩阵
    trajectory = []
    for pose in poses:
        t = pose[:3]
        q = pose[3:]
        r = R.from_quat(q).as_matrix()
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = r
        camera_pose[:3, 3] = t
        trajectory.append(camera_pose)
    return trajectory


def load_point_cloud_and_poses(ply_file, poses_file):
    # 加载点云
    pcd = o3d.io.read_point_cloud(ply_file)
    # 加载相机轨迹
    poses = np.load(poses_file)  # 假设为 (N, 7)，格式 [x, y, z, qx, qy, qz, qw]
    return pcd, poses

# def render_frames(pcd, camera_trajectory, output_folder):
#     # 创建输出文件夹
#     os.makedirs(output_folder, exist_ok=True)
    
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=512, height=288, visible=False)  # 创建窗口
#     vis.add_geometry(pcd)  # 添加点云

#     for i, pose in enumerate(camera_trajectory):
#         ctr = vis.get_view_control()
#         ctr.convert_from_pinhole_camera_parameters(o3d.camera.PinholeCameraParameters())
#         ctr.camera.localize()  # 相机根据位姿设置
#         vis.poll_events()
#         vis.update_renderer()

#         # 保存帧
#         image_path = os.path.join(output_folder, f"frame_{i:04d}.png")
#         vis.capture_screen_image(image_path)
#         print(f"Saved frame {i} to {image_path}")

#     vis.destroy_window()

def render_frames_offscreen(pcd, camera_trajectory, output_folder, width=512, height=288):
    """
    使用 Open3D 的离屏渲染器生成点云视频帧。

    Args:
        pcd (open3d.geometry.PointCloud): 点云数据。
        camera_trajectory (list): 相机轨迹（4x4 位姿矩阵列表）。
        output_folder (str): 保存渲染帧的路径。
        width (int): 图像宽度。
        height (int): 图像高度。
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 创建离屏渲染器
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.add_geometry("point_cloud", pcd, rendering.MaterialRecord())

    # 设置相机参数（FOV 和其他参数）
    fov = 60.0  # 设置视场角（单位：度）
    aspect_ratio = width / height
    near_plane = 0.1
    far_plane = 1000.0

    for i, pose in enumerate(camera_trajectory):
        # 设置相机的透视投影
        renderer.scene.camera.set_projection(
            fov, aspect_ratio, near_plane, far_plane, rendering.Camera.FovType.Vertical
        )
        
        # 转换参数为 (3, 1) 的形状并确保数据类型为 float32
        center = np.array(pose[:3, 3]).reshape(3, 1).astype(np.float32)  # 目标点
        eye = np.array(pose[:3, 3] - pose[:3, 2] * 5).reshape(3, 1).astype(np.float32)  # 相机位置
        up = np.array(pose[:3, 1]).reshape(3, 1).astype(np.float32)  # 相机上方向

        # 设置相机视角
        renderer.scene.camera.look_at(center, eye, up)

        # 渲染图像并保存
        image = renderer.render_to_image()
        image_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        o3d.io.write_image(image_path, image)
        print(f"Saved frame {i} to {image_path}")

    print("Rendering complete.")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()
    reconstruction_path = args.reconstruction_path

    # 路径
    ply_file = f"reconstructions/{reconstruction_path}/point_cloud.ply"
    poses_file = f"reconstructions/{reconstruction_path}/poses.npy"

    pcd, poses = load_point_cloud_and_poses(ply_file, poses_file)
    print("Loaded point cloud and poses.")

    # pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 下采样点云
    # print("Downsampled point cloud.")

    camera_trajectory = create_camera_trajectory(poses)
    print("Created camera trajectory.")

    # 保存帧
    output_folder = "work/abandonedFactory/frames"
    #render_frames(pcd, camera_trajectory, output_folder)
    render_frames_offscreen(pcd, camera_trajectory, output_folder, width=512, height=288)