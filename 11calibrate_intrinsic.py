import logging
import os

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
    # 将 numpy 数组转换为 Python 列表

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
        # 在计算过程中添加进度条
        logging.info("开始相机标定...")
        # 注意：cv2.calibrateCamera 并不提供直接的进度条更新，因此我们可以在这里添加状态更新
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


def process_camera_params(input_file_path, output_file_path):
    """
    处理单个相机参数 YAML 文件，并将其转换为指定格式保存到新的文件中。

    参数:
    - input_file_path: str, 输入 YAML 文件的路径
    - output_file_path: str, 输出 YAML 文件的路径
    """
    try:
        # 加载 YAML 文件
        with open(input_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # 提取数据
        intrinsic_matrix = data.get('IntrinsicMatrix', [])
        distortion = data.get('Distortion', [])
        tangential_distortion = data.get('TangentialDistortion', [])

        # 将内参矩阵转换为第二种文件格式
        intrinsic_matrix_second_format = [
            [intrinsic_matrix[0][0], intrinsic_matrix[0][1], intrinsic_matrix[0][2]],
            [intrinsic_matrix[1][0], intrinsic_matrix[1][1], intrinsic_matrix[1][2]],
            [intrinsic_matrix[2][0], intrinsic_matrix[2][1], intrinsic_matrix[2][2]]
        ]

        # 创建新的格式数据
        new_format_data = {
            'IntrinsicMatrix': intrinsic_matrix_second_format,
            'RadialDistortion': distortion[0:2],
            'TangentialDistortion': distortion[3:5]
        }

        # 保存到新的 YAML 文件中
        with open(output_file_path, 'w') as file:
            yaml.dump(new_format_data, file, default_flow_style=False)

        print(f"处理完成。数据已保存到 {output_file_path}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    global output_yaml
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # step 1 初始化
    index = '01'  # 修改索引值
    base_directory = os.path.dirname(__file__)
    image_directory_path = os.path.join(base_directory, 'data', f'images_cal_{index}')
    params_directory_path = os.path.join(base_directory, 'data')

    # step 2 读数据并开始标定
    if not os.path.isdir(image_directory_path):
        logging.error(f"图像目录 '{image_directory_path}' 不存在。")
    camera_matrix, dist_coeffs, r_vecs, t_vecs = calibrate_camera_from_images(image_directory_path)

    # step 3 数据输出
    if camera_matrix is not None and dist_coeffs is not None:
        output_yaml = os.path.join(params_directory_path, f'camera_params_{index}.yaml')
        save_camera_parameters(camera_matrix, dist_coeffs, output_yaml)
        logging.info(f"相机参数已成功保存到 '{output_yaml}'")
    else:
        logging.error("相机内参标定失败，无法保存标定结果。")
