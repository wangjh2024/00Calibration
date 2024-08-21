import cv2
import numpy as np
import os
import yaml
from tqdm import tqdm
import logging


def save_camera_parameters(camera_matrix, dist_coeffs, r_vec, t_vec, filename):
    """
    将相机内参和畸变系数保存到 YAML 文件，格式化为指定的结构
    :param camera_matrix: 相机内参矩阵 (3x3)
    :param dist_coeffs: 畸变系数 (1x5 或 1x4)
    :param r_vec: 旋转向量 (1x3)
    :param t_vec: 平移向量 (1x3)
    :param filename: 保存的 YAML 文件名
    """
    parameters = {
        "IntrinsicMatrix": camera_matrix.tolist(),
        "RadialDistortion[k1,k2]": dist_coeffs[:2].tolist(),  # 取前两个作为径向畸变系数
        "TangentialDistortion[p1,p2]": dist_coeffs[2:4].tolist(),  # 取接下来的两个作为切向畸变系数
        "R": [r_vec.tolist() for r_vec in r_vecs],  # 多个旋转向量
        "T": [t_vec.tolist() for t_vec in t_vecs]   # 多个平移向量
    }
    with open(filename, 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)
    print(f"相机参数已保存到 '{filename}'")


def calibrate_camera_from_images(directory_path, board_size=(4, 3), board_scale=30):
    """
    使用棋盘格图像进行相机内参标定
    :param directory_path: 包含标定板图像的目录路径
    :param board_size: 棋盘格的尺寸 (内角点的数目) (cols, rows)
    :param board_scale: 每个棋盘格的边长，单位为毫米
    :return: 相机矩阵和畸变系数，旋转向量和转换向量
    """
    # 棋盘格角点的实际世界坐标 (以毫米为单位)
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale / 1000  # 转换为米

    obj_points = []  # 3D 点在世界坐标系中的坐标
    img_points = []  # 2D 点在图像平面中的坐标

    # 获取目录中的所有图像文件
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # 通过 tqdm 显示进度条
    for image_file in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(directory_path, image_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            logging.warning(f"在图像 '{image_file}' 中未找到棋盘格角点。")

    if obj_points and img_points:
        # 计算相机矩阵和畸变系数
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],
                                                                            None, None)
        if ret:
            logging.info("相机内参标定成功")
            logging.info(f"相机矩阵:\n{camera_matrix}")
            logging.info(f"畸变系数:\n{dist_coeffs}")
            return camera_matrix, dist_coeffs, rvecs, tvecs
        else:
            logging.error("相机内参标定失败")
    else:
        logging.error("没有足够的标定图像用于标定。")
    return None, None, None, None

def renumber_files(directory_path, prefix='file', extension='.jpg'):
    """
    对指定目录中的所有指定类型的文件进行重新编号
    :param directory_path: 目录路径
    :param prefix: 文件名前缀
    :param extension: 文件扩展名（例如 '.jpg'）
    """
    # 确保目录存在
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在。")
        return

    # 获取目录中所有指定类型的文件
    files = [f for f in os.listdir(directory_path) if
             os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(extension)]

    # 按文件名排序（可以按需更改排序标准）
    files.sort()

    # 遍历文件并重命名
    for index, file in enumerate(files, start=1):
        old_file_path = os.path.join(directory_path, file)
        new_file_name = f"{prefix}_{index:03}{extension}"  # 使用 3 位数字进行编号
        new_file_path = os.path.join(directory_path, new_file_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"已重命名 '{old_file_path}' 为 '{new_file_path}'")


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 相对路径设置
    base_directory = os.path.dirname(__file__)  # 当前脚本所在目录
    image_directory_path = os.path.join(base_directory, 'data', 'images_cal_01')
    params_directory_path = os.path.join(base_directory, 'data')

    #重命名
    # renumber_files(SAVE_PATH)

    if not os.path.isdir(image_directory_path):
        print(f"错误: 图像目录 '{image_directory_path}' 不存在。")
        return

    # 计算相机内参
    camera_matrix, dist_coeffs, r_vecs, t_vecs = calibrate_camera_from_images(image_directory_path)

    # 保存标定结果
    if camera_matrix is not None and dist_coeffs is not None:
        output_yaml = os.path.join(params_directory_path, 'camera_params_01.yaml')
        save_camera_parameters(camera_matrix, dist_coeffs, r_vecs, t_vecs, output_yaml)
        logging.info(f"相机参数已成功保存到 '{output_yaml}'")
    else:
        logging.error("相机内参标定失败，无法保存标定结果。")


if __name__ == "__main__":
    main()
