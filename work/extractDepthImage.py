'''
    This script is used to get a glimpse of depth images and RGB images.
    Results will be saved in `work/<reconstruction_path>/depth (or color)/`, 
    where reconstruction_path is what you input in the command line.
'''
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R
import cv2 
import os

def intrinsics_to_matrix(intrinsics):
    """将 [fx, fy, cx, cy] 转换为 3x3 内参矩阵"""
    #print(intrinsics)
    fx, fy, cx, cy = intrinsics
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def pose_to_matrix(pose):
    """将 [x, y, z, qx, qy, qz, qw] 转换为 4x4 位姿矩阵"""
    t = pose[:3]  # 平移
    q = pose[3:]  # 四元数
    r = R.from_quat(q).as_matrix()  # 旋转矩阵
    matrix = np.eye(4)
    matrix[:3, :3] = r
    matrix[:3, 3] = t
    return matrix

def save_depth_image(disps, images, stride = 2, output_dir=None):

    if output_dir == None: 
        output_dir = "work/warehouse"
    else : 
        output_dir = os.path.join("work", output_dir)

    for idx in range(len(disps[:100:stride])):
        i = stride * idx
        print("Processing frame {}".format(i))
        # 1. 获取当前帧数据
        depth = 1.0 / np.clip(disps[i], 1e-5, None)  # 将视差转为深度
        color = images[i].transpose(1, 2, 0) / 255.0  # 归一化图像色彩
        # print(color.shape) # (384, 512, 3)
        # print(depth)

        multiplier = 200.0 
        depth = np.clip(depth, 0, 255 / multiplier)
        depth_image = (depth * multiplier).astype(np.uint8)  # 转换为 8 位灰度图
        color_image_bgr = (color * 255).astype(np.uint8)[:, :, ::-1]  # 转换为 BGR 格式以适配 OpenCV

        # 保存深度图
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        depth_image_path = os.path.join(output_dir, "depth", f"depth_{i:04d}.png")
        cv2.imwrite(depth_image_path, depth_image)
        print(f"Saved depth map to {depth_image_path}")

        # 保存颜色图
        os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
        color_image_path = os.path.join(output_dir, "color", f"color_{i:04d}.png")
        cv2.imwrite(color_image_path, color_image_bgr)
        print(f"Saved color image to {color_image_path}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    # 加载数据并调用函数
    reconstruction_path = args.reconstruction_path
    disps = np.load(f"reconstructions/{reconstruction_path}/disps.npy")
    print("Disps shape:", disps.shape)
    images = np.load(f"reconstructions/{reconstruction_path}/images.npy")

    # 保存点云到 PLY 文件
    save_depth_image(disps, images, stride=2, output_dir=reconstruction_path)
