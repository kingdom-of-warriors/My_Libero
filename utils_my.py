import cv2
from scipy.ndimage import rotate, gaussian_filter
import numpy as np
import os
from PIL import Image
import h5py

def create_motion_blur_kernel(kernel_size, angle, intensity):
    """
    创建运动模糊卷积核
    kernel_size: 卷积核大小
    angle: 运动方向角度
    intensity: 模糊强度
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    angle_rad = np.deg2rad(angle) # 将角度转换为弧度
    dx = intensity * np.cos(angle_rad) # 计算运动方向的向量
    dy = intensity * np.sin(angle_rad)
    
    for i in range(-center, center + 1): # 计算卷积核
        x = center + int(dx * i / center) 
        y = center + int(dy * i / center)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    return kernel / kernel.sum()

def add_camera_jitter_and_blur(image, max_translation=5, max_rotation=2, 
                             blur_kernel_size=15, blur_intensity=3):
    """
    模拟相机抖动和运动模糊效果，避免产生黑边
    image: 输入图像
    max_translation: 最大平移像素数
    max_rotation: 最大旋转角度(度)
    blur_kernel_size: 模糊卷积核大小
    blur_intensity: 模糊强度
    """
    dx = np.random.uniform(-max_translation, max_translation)
    dy = np.random.uniform(-max_translation, max_translation)
    angle = np.random.uniform(-max_rotation, max_rotation)
    translated = np.pad(image, ((max_translation, max_translation), (max_translation, max_translation), (0, 0)), mode='reflect')
    translated = np.roll(translated, int(dx) + max_translation, axis=1)
    translated = np.roll(translated, int(dy) + max_translation, axis=0)
    # 为了避免黑边，裁剪回原始大小
    translated = translated[max_translation:-max_translation, max_translation:-max_translation]
    padded = np.pad(translated, ((max_translation, max_translation), (max_translation, max_translation), (0, 0)), mode='reflect')
    rotated = rotate(padded, angle, reshape=False, mode='reflect')
    jittered = rotated[max_translation:-max_translation, max_translation:-max_translation]
    # 计算出运动角度
    angle = np.rad2deg(np.arctan2(dy, dx))
    blurred = np.zeros_like(jittered)
    kernel = create_motion_blur_kernel(blur_kernel_size, angle, blur_intensity)
    # 应用卷积模糊
    for channel in range(3):
        blurred[:,:,channel] = cv2.filter2D(jittered[:,:,channel], -1, kernel, borderType=cv2.BORDER_REFLECT)
    final_image = gaussian_filter(blurred, sigma=0.5)
    
    return final_image.astype(np.uint8)


def add_lighting_effect(image, light_position='top_left', intensity=0.5):
    """
    给图像添加光照效果
    image: numpy数组，输入图像
    light_position: 字符串，光源位置 ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    intensity: float, 光照强度 (0.0 到 1.0)
    """
    height, width = image.shape[:2]
    # 创建渐变遮罩
    y, x = np.ogrid[:height, :width]
    
    if light_position == 'top_left': center_y, center_x = 0, 0
    elif light_position == 'top_right': center_y, center_x = 0, width
    elif light_position == 'bottom_left': center_y, center_x = height, 0
    elif light_position == 'bottom_right': center_y, center_x = height, width
    else: raise ValueError("无效的光源位置")
    
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(height**2 + width**2)
    gradient = 1 - (distance / max_distance)
    gradient = gradient * intensity
    result = image.astype(float)
    for i in range(3): 
        result[:,:,i] = result[:,:,i] * (1 + gradient)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def extract_camera_images(hdf5_path, save_dir):
    """
    从 HDF5 文件中提取摄像机图像并保存为视频。

    Args:
        hdf5_path (str): HDF5 文件路径。
        save_dir (str): 保存视频的目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 打开 HDF5 文件
    with h5py.File(hdf5_path, "r") as f:
        # 遍历每个 episode
        for episode_name in f.keys():
            episode = f[episode_name]
            print(f"处理 {episode_name}...")
            
            # 遍历每个 demo
            for demo_name in episode.keys():                    
                print(f"  处理 {demo_name}...")
                demo = episode[demo_name]
                # 获取观察数据
                obs = demo["obs"]
                print(f"  观察数据键: {list(obs.keys())}")
                
                agentview = obs['agentview_rgb']
                print(f"  处理 agentview_rgb, 类型: {type(agentview)}")
                
                camera_data = agentview[:]
                video_path = os.path.join(save_dir, f"{episode_name}_{demo_name}_agentview.mp4")
                
                # 获取视频的帧率和大小 (假设所有帧大小相同)
                height, width, _ = camera_data[0].shape
                fps = 20  # 假设帧率为20，可以根据实际情况调整

                # 定义视频编码器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                for t in range(camera_data.shape[0]):
                    frame = camera_data[t][::-1].astype(np.uint8)
                    # 写入帧到视频
                    video_writer.write(frame)
                
                # 释放视频编写器
                video_writer.release()
                print(f"  保存了 {camera_data.shape[0]} 帧到 {video_path}")

    print("所有摄像机图像提取完成。")

