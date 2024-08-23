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
    obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

    _, r_vec, t_vec = cv2.solvePnP(obj_points, corners, K, Distoreffs)

    Rcb, _ = cv2.Rodrigues(r_vec)
    Tcb = t_vec.squeeze()

    Rfc = Rfb @ Rcb.T
    Tfc = Rfb @ (-Rcb.T @ Tcb) + Tfb

    return Rfc, Tfc


def display_corners(frame, corners, board_size):
    """
    显示检测到的角点并去除图像畸变
    :param frame: 当前帧图像
    :param corners: 棋盘角点的像素坐标
    :param board_size: 棋盘格的大小 (列数, 行数)
    """
    cv2.drawChessboardCorners(frame, board_size, corners, True)
    center = tuple(map(int, corners[0].ravel()))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    undistorted_frame = cv2.undistort(frame, K, Distoreffs)
    cv2.imshow("frame", undistorted_frame)
    cv2.waitKey(0)  # Wait for a key press to close the image window


def load_camera_parameters(file_path):
    """从 YAML 文件中加载相机参数"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    K = np.array(data['IntrinsicMatrix'])
    Distoreffs_0 = np.array(data['Distortion[k1,k2,k3,p1,p2]'])
    Distoreffs = np.array([Distoreffs_0[0][0], Distoreffs_0[0][1], 0, Distoreffs_0[0][3], Distoreffs_0[0][4]])

    return K, Distoreffs


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    board_size = (4, 3)
    board_scale = 24

    base_directory = os.path.dirname(__file__)
    input_directory = os.path.join(base_directory, 'data', 'params_files', 'camera_params_08.yaml')
    K, Distoreffs = load_camera_parameters(input_directory)

    print("内参矩阵:\n", K)
    print("畸变系数:\n", Distoreffs, '\n')

    Tfb = np.array([-47, -57, 380])
    Rfb = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

    ret, frame = cap.read()  # 从摄像头读取一帧图像
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag, corners = cv2.findChessboardCorners(gray, board_size)

        if flag:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            display_corners(frame, corners, board_size)
            Rfc, Tfc = calculate_transformation(corners.squeeze(), board_size, board_scale)
            print(f"Rotation (Euler angles): {Rotation.from_matrix(Rfc).as_euler('xyz', degrees=True)}")
            print(f"Translation: {Tfc}")
        else:
            print("未检测到棋盘角点。")

    cap.release()
    cv2.destroyAllWindows()
