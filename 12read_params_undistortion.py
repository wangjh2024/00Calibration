import os

import cv2
import numpy as np
import yaml


def load_camera_parameters(file_path):
    """从 YAML 文件中加载相机参数"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # 提取内参矩阵和畸变系数
    K = np.array(data['IntrinsicMatrix'])
    distCoeffs = np.array(data['Distortion[k1,k2,k3,p1,p2]'])
    return K, distCoeffs


def on_mouse(event, x, y, flags, param):
    """处理鼠标点击事件：输出点击位置"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击位置: {(x, y)}")


def display_image_windows(original_image, undistorted_image):
    """创建窗口并显示图像"""
    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", on_mouse)
    cv2.namedWindow("Undistorted Image")
    cv2.setMouseCallback("Undistorted Image", on_mouse)

    # 显示图像
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Undistorted Image", undistorted_image)


if __name__ == "__main__":
    # 设定基础目录路径
    base_directory = os.path.dirname(__file__)
    params_directory_path = os.path.join(base_directory, 'data', 'params_files')

    # 构建 YAML 文件的完整路径
    camera_params_path = os.path.join(params_directory_path, 'camera_params_01.yaml')

    # 加载相机参数
    K, distCoeffs = load_camera_parameters(camera_params_path)

    # 打印内参矩阵和畸变系数
    print("内参矩阵:\n", K)
    print("畸变系数:\n", distCoeffs)

    # 定义图像文件路径
    image_path = os.path.join(base_directory, 'data', 'images_cal_01', '12.jpg')  # 根据实际文件名修改

    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像文件: {image_path}")
        exit()

    # 对图像进行畸变校正
    img_undistorted = cv2.undistort(image, K, distCoeffs)

    # 显示图像
    display_image_windows(image, img_undistorted)

    # 等待用户按键并关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
