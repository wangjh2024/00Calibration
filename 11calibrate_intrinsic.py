import logging
import os
import random

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def save_camera_parameters(camera_matrix, dist_coeffs, filename):
    """
    将相机内参和所有五个畸变系数保存到 YAML 文件，格式化为指定的结构
    :param camera_matrix: 相机内参矩阵 (3x3)
    :param dist_coeffs: 畸变系数 (1x5 或 1x4)
    :param filename: 保存的 YAML 文件名
    """

    # 确保输入为 NumPy 数组
    if not isinstance(camera_matrix, np.ndarray) or not isinstance(dist_coeffs, np.ndarray):
        raise TypeError("相机内参和畸变系数应为 NumPy 数组")

    # 组织参数
    parameters = {
        "IntrinsicMatrix": camera_matrix.tolist(),
        "Distortion[k1,k2,k3,p1,p2]": dist_coeffs.tolist(),  # 前三个作为径向畸变系数
    }

    # 保存到 YAML 文件
    with open(filename, 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False, sort_keys=False)

    print(f"相机参数已保存到 '{filename}'")


def calibrate_camera_from_images(directory_path, board_size=(4, 3), board_scale=30):
    """
    使用棋盘格图像进行相机标定。
    """
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale / 1000  # 转换为米

    obj_points, img_points = [], []

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.png'))]
    total_images = len(image_files)

    if total_images == 0:
        logging.error("图像目录为空，没有可用的标定图像。")
        return None, None, None, None

    # 使用 tqdm 显示图像处理进度条
    for image_file in tqdm(image_files, desc="处理图像", unit="图像"):
        img_path = os.path.join(directory_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"无法读取图像 '{image_file}'")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            logging.warning(f"在图像 '{image_file}' 中未找到棋盘格角点。")

    if obj_points and img_points:
        logging.info("开始相机标定...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        if ret:
            logging.info("相机内参标定成功")
            return camera_matrix, dist_coeffs, rvecs, tvecs
        else:
            logging.error("相机内参标定失败")
    else:
        logging.error("没有足够的标定图像用于标定。")
    return None, None, None, None


def calibrate_camera_from_images(directory_path, board_size=(4, 3), board_scale=30, num_samples=80):
    """
    使用棋盘格图像进行相机标定。
    :param directory_path: 图像所在目录
    :param board_size: 棋盘格的行列数 (列数, 行数)
    :param board_scale: 棋盘格的尺度 (mm)
    :param num_samples: 采样图像的数量
    """
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale / 1000  # 转换为米

    obj_points, img_points = [], []

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.png'))]
    total_images = len(image_files)

    if total_images == 0:
        logging.error("图像目录为空，没有可用的标定图像。")
        return None, None, None, None

    # 进行降采样，随机选择 num_samples 张图像
    sampled_image_files = random.sample(image_files, min(num_samples, total_images))

    # 使用 tqdm 显示图像处理进度条
    for image_file in tqdm(sampled_image_files, desc="处理图像", unit="图像"):
        img_path = os.path.join(directory_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"无法读取图像 '{image_file}'")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            logging.warning(f"在图像 '{image_file}' 中未找到棋盘格角点。")

    if obj_points and img_points:
        logging.info("开始相机标定...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        if ret:
            logging.info("相机内参标定成功")
            return camera_matrix, dist_coeffs, rvecs, tvecs
        else:
            logging.error("相机内参标定失败")
    else:
        logging.error("没有足够的标定图像用于标定。")
    return None, None, None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设定基础目录路径
    base_directory = os.path.dirname(__file__)
    params_directory_path = os.path.join(base_directory, 'data')

    # 处理 images_cal_01 至 images_cal_09 九个文件夹
    for i in range(1, 10):
        index = f"{i:02d}"  # 生成索引 01, 02, ..., 09
        image_directory_path = os.path.join(base_directory, 'data', f'images_cal_{index}')

        # step 1 读数据并开始标定
        if not os.path.isdir(image_directory_path):
            logging.error(f"图像目录 '{image_directory_path}' 不存在。")
            continue

        camera_matrix, dist_coeffs, r_vecs, t_vecs = calibrate_camera_from_images(
            image_directory_path, num_samples=64
        )
        # step 2 数据输出
        if camera_matrix is not None and dist_coeffs is not None:
            output_yaml = os.path.join(params_directory_path, f'camera_params_{index}.yaml')
            save_camera_parameters(camera_matrix, dist_coeffs, output_yaml)
            logging.info(f"相机参数已成功保存到 '{output_yaml}\n'")
        else:
            logging.error(f"相机内参标定失败，无法保存标定结果 '{output_yaml}'")
