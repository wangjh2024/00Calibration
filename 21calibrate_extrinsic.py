import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def calculate_transformation(corners, board_size, board_scale):
    """
    计算相机相对于法兰盘的旋转和平移矩阵
    :param corners: 棋盘角点的像素坐标
    :param board_size: 棋盘格的大小 (列数, 行数)
    :param board_scale: 棋盘格的实际尺寸
    :return: 相机相对于法兰盘的旋转矩阵和平移向量
    """
    # 创建棋盘格的3D点坐标，假设棋盘在z=0的平面上
    obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

    # 使用PnP算法求解相机位姿 (旋转向量和位移向量)
    _, r_vec, t_vec = cv2.solvePnP(obj_points, corners, K, Distoreffs)

    # 将旋转向量转换为旋转矩阵 Rcb (标定板相对于相机的旋转矩阵)
    Rcb, _ = cv2.Rodrigues(r_vec)
    Tcb = t_vec.squeeze()  # 将平移向量 t_vec 转为1D数组

    # 计算相机相对于法兰盘的旋转矩阵 Rfc 和平移向量 Tfc
    Rfc = Rfb @ Rcb.T  # 相机相对于法兰盘的旋转矩阵
    Tfc = Rfb @ (-Rcb.T @ Tcb) + Tfb  # 相机相对于法兰盘的平移向量

    return Rfc, Tfc  # 返回旋转矩阵和平移向量


def display_corners(frame, corners, board_size):
    """
    显示检测到的角点并去除图像畸变
    :param frame: 当前帧图像
    :param corners: 棋盘角点的像素坐标
    :param board_size: 棋盘格的大小 (列数, 行数)
    """
    # 在图像上绘制棋盘角点
    cv2.drawChessboardCorners(frame, board_size, corners, True)

    # 将角点的坐标转换为整数并绘制圆
    center = tuple(map(int, corners[0].ravel()))  # 将角点坐标转换为整数元组
    cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 绘制红色圆圈

    # 去除图像畸变
    undistorted_frame = cv2.undistort(frame, K, Distoreffs)

    # 显示去畸变后的图像
    cv2.imshow("frame", undistorted_frame)


def load_camera_parameters(file_path):
    """从 YAML 文件中加载相机参数"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # 提取内参矩阵和畸变系数
    K = np.array(data['IntrinsicMatrix'])
    Distoreffs_0 = np.array(data['Distortion[k1,k2,k3,p1,p2]'])

    Distoreffs = np.array([Distoreffs_0[0][0], Distoreffs_0[0][1], 0, Distoreffs_0[0][3], Distoreffs_0[0][4]])

    return K, Distoreffs


if __name__ == "__main__":
    # 打开默认摄像头
    cap = cv2.VideoCapture(1)
    board_size = (4, 3)  # 棋盘格的大小
    board_scale = 24  # 棋盘格每个方格的边长

    base_directory = os.path.dirname(__file__)
    input_directory = os.path.join(base_directory, 'data', 'params_files', 'camera_params_08.yaml')
    K, Distoreffs = load_camera_parameters(input_directory)

    # 打印内参矩阵和畸变系数
    print("内参矩阵:\n", K)
    print("畸变系数:\n\n\n", Distoreffs)

    # 定义相机的内参矩阵 K 和畸变系数 D
    # K = np.array([[720.4387, 0, 319.0484],  # 焦距 fx 和主点 cx
    #               [0, 719.4163, 204.2231],  # 焦距 fy 和主点 cy
    #               [0, 0, 1]])  # 齐次坐标系的第三行
    # Distoreffs = np.array([-0.5124, 0.1789, 0.0118, -0.0016])  # 畸变系数 [k1, k2, p1, p2]

    # 标定板相对于法兰盘的平移向量 Tfb 和旋转矩阵 Rfb
    Tfb = np.array([-47, -57, 380])  # 标定板相对于法兰盘的平移向量
    Rfb = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()  # 欧拉角转旋转矩阵

    while True:
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 如果读取失败，跳过本次循环
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        flag, corners = cv2.findChessboardCorners(gray, board_size)  # 检测棋盘角点

        if flag:  # 如果检测到棋盘角点
            # 使用亚像素精度优化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

            # 显示角点并去畸变
            display_corners(frame, corners, board_size)

            # 计算相机相对于法兰盘的旋转和平移矩阵
            Rfc, Tfc = calculate_transformation(corners.squeeze(), board_size, board_scale)

            # 打印旋转矩阵对应的欧拉角（以度为单位）和平移向量
            print(f"Rotation (Euler angles): {Rotation.from_matrix(Rfc).as_euler('xyz', degrees=True)}")
            print(f"Translation: {Tfc}")

        # 如果按下 'q' 键，退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
