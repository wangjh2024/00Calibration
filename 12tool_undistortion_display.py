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
    distCoeffs_0 = np.array(data['Distortion[k1,k2,k3,p1,p2]'])

    # 去掉 k3
    distCoeffs_1 = np.array([distCoeffs_0[0][0], distCoeffs_0[0][1], 0, distCoeffs_0[0][3], distCoeffs_0[0][4]])
    distCoeffs = np.array([distCoeffs_1])
    return K, distCoeffs


def on_mouse(event, x, y, flags, param):
    """处理鼠标点击事件：关闭窗口"""
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()


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

    # 等待直到所有窗口关闭
    while cv2.getWindowProperty("Original Image", cv2.WND_PROP_VISIBLE) >= 1 or \
            cv2.getWindowProperty("Undistorted Image", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)


if __name__ == "__main__":
    # 设定基础目录路径
    base_directory = os.path.dirname(__file__)

    # 定义 index 来选择相机参数文件和图像文件夹
    index = 5  # 可以修改为其他值，例如 2, 3, ...

    # 构建 YAML 文件的完整路径
    camera_params_filename = f'camera_params_0{index}.yaml'
    camera_params_path = os.path.join(base_directory, 'data', 'params_files', camera_params_filename)

    # 加载相机参数
    K, distCoeffs = load_camera_parameters(camera_params_path)

    # 打印内参矩阵和畸变系数
    print("内参矩阵:\n", K)
    print("畸变系数:\n", distCoeffs)

    # 定义图像文件夹路径
    images_directory_name = f'images_cal_0{index}'
    images_directory_path = os.path.join(base_directory, 'data', 'capture_images', images_directory_name)

    # 遍历文件夹中的全部图片
    for image_file in os.listdir(images_directory_path):
        # 构建每个图片的完整路径
        image_path = os.path.join(images_directory_path, image_file)

        # 读取图像
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue

        # 对图像进行畸变校正
        img_undistorted = cv2.undistort(image, K, distCoeffs)

        # 显示图像
        display_image_windows(image, img_undistorted)
