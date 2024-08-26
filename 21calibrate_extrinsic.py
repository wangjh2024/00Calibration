import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def calculate_transformation(corners, board_size, board_scale):
    """
    计算相机相对于法兰盘的旋转和平移矩阵。
    :param corners: 棋盘角点的像素坐标。
    :param board_size: 棋盘格的大小 (列数, 行数)。
    :param board_scale: 棋盘格的实际尺寸。
    :return: 相机相对于法兰盘的旋转矩阵和平移向量。
    """
    # 生成棋盘格的实际坐标点 (obj_points)，假设棋盘的每个角点都是在z=0平面上
    obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

    # 使用 solvePnP 方法计算旋转向量 (r_vec) 和平移向量 (t_vec)
    _, r_vec, t_vec = cv2.solvePnP(obj_points, corners, K, Distoreffs)

    # 将旋转向量转换为旋转矩阵
    Rcb, _ = cv2.Rodrigues(r_vec)
    Tcb = t_vec.squeeze()

    # 计算相机相对于法兰盘的旋转矩阵 (Rfc) 和平移向量 (Tfc)
    Rfc = Rfb @ Rcb.T
    Tfc = Rfb @ (-Rcb.T @ Tcb) + Tfb

    return Rfc, Tfc


def display_corners(frame, corners, board_size):
    """
    显示检测到的角点并去除图像畸变。
    :param frame: 当前帧图像。
    :param corners: 棋盘角点的像素坐标。
    :param board_size: 棋盘格的大小 (列数, 行数)。
    """
    # 在图像上绘制棋盘格角点
    cv2.drawChessboardCorners(frame, board_size, corners, True)

    # 绘制角点的中心位置
    center = tuple(map(int, corners[0].ravel()))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # 去除图像畸变
    undistorted_frame = cv2.undistort(frame, K, Distoreffs)
    cv2.imshow("frame", undistorted_frame)
    cv2.waitKey(500)


def load_camera_parameters(file_path):
    """
    从 YAML 文件中加载相机参数。
    :param file_path: YAML 文件路径。
    :return: 相机内参矩阵 K 和畸变系数 Distoreffs。
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"读取 YAML 文件出错: {e}")

    # 读取相机内参矩阵 K
    K = np.array(data['IntrinsicMatrix'])

    # 读取畸变系数，并将其转换为 cv2.undistort() 所需的格式
    Distoreffs_0 = np.array(data['Distortion[k1,k2,k3,p1,p2]'])
    Distoreffs = np.array([Distoreffs_0[0][0], Distoreffs_0[0][1], 0, Distoreffs_0[0][3], Distoreffs_0[0][4]])

    return K, Distoreffs


def save_parameters(file_path, K, Distoreffs, Rfc_avg, Tfc_avg):
    """
    保存相机参数和计算得到的平均旋转矩阵和平移向量到 YAML 文件中。
    :param file_path: 保存的 YAML 文件路径。
    :param K: 相机内参矩阵。
    :param Distoreffs: 畸变系数。
    :param Rfc_avg: 平均旋转矩阵。
    :param Tfc_avg: 平均平移向量。
    """
    # 将旋转矩阵转换为欧拉角以便保存
    Rfc_avg_euler = Rotation.from_matrix(Rfc_avg).as_euler('xyz', degrees=True)

    # 按照指定顺序创建字典
    parameters = {
        'IntrinsicMatrix_sign': [
            ['fx', '0', 'x0'],
            ['0', 'fy', 'y0'],
            ['0', '0', '1']
        ],
        'IntrinsicMatrix': K.tolist(),  # 将内参矩阵转换为列表
        'RadialDistortion[k1, k2]': Distoreffs[0:2].tolist(),  # 径向畸变系数 (k1, k2)
        'TangentialDistortion[p1, p2]': Distoreffs[3:5].tolist(),  # 切向畸变系数 (p1, p2)
        'Y_RotationEuler': Rfc_avg_euler.tolist(),  # 平均旋转矩阵的欧拉角
        'Y_TranslationVector': Tfc_avg.tolist()  # 平均平移向量
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 保存参数到 YAML 文件
    try:
        with open(file_path, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)
        print(f"Parameters saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving parameters: {e}")


if __name__ == "__main__":
    # 手动设置序号，取值范围01-09
    sequence_number = "08"  # 你可以将其更改为"01", "02", ..., "09"

    board_size = (4, 3)  # 棋盘格的大小 (列数, 行数)
    board_scale = 24  # 棋盘格的实际尺寸 (毫米)

    base_directory = os.path.dirname(__file__)  # 当前脚本所在目录
    input_directory = os.path.join(base_directory, 'data', 'params_files', f'camera_params_{sequence_number}.yaml')
    save_directory = os.path.join(base_directory, 'data', 'saved_images', f'saved_images_{sequence_number}')
    os.makedirs(save_directory, exist_ok=True)

    # 从文件中加载相机参数
    K, Distoreffs = load_camera_parameters(input_directory)

    print("内参矩阵:\n", K)
    print("畸变系数:\n", Distoreffs, '\n')

    # 定义法兰盘到相机的变换（假设已知）
    Tfb = np.array([-24, -36, 495])  # 法兰盘到相机的平移向量
    Rfb = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()  # 法兰盘到相机的旋转矩阵（单位矩阵）

    # 初始化旋转矩阵和平移向量的累积和
    Rfc_sum = np.zeros((3, 3))
    Tfc_sum = np.zeros(3)
    valid_count = 0  # 有效图像计数

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    try:
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flag, corners = cv2.findChessboardCorners(gray, board_size)

                if flag:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
                    display_corners(frame, corners, board_size)

                    image_path = os.path.join(save_directory, f'image_{i + 1}.png')
                    cv2.imwrite(image_path, frame)

                    # 计算相机相对于法兰盘的旋转矩阵和平移向量
                    Rfc, Tfc = calculate_transformation(corners.squeeze(), board_size, board_scale)
                    Rfc_sum += Rfc
                    Tfc_sum += Tfc
                    valid_count += 1

                    print(f"R (frame {i + 1}): {Rotation.from_matrix(Rfc).as_euler('xyz', degrees=True)}")
                    print(f"T (frame {i + 1}): {Tfc}")
                else:
                    print(f"未检测到棋盘角点 (frame {i + 1})。")

        if valid_count > 0:
            # 计算平均旋转矩阵和平移向量
            Rfc_avg = Rfc_sum / valid_count
            Tfc_avg = Tfc_sum / valid_count

            print("\n平均 R:", Rotation.from_matrix(Rfc_avg).as_euler('xyz', degrees=True))
            print("平均 T:", Tfc_avg)

            # 保存参数
            output_directory = os.path.join(base_directory, 'data', 'params_files_output',
                                            f'camera_params_{sequence_number}.yaml')
            save_parameters(output_directory, K, Distoreffs, Rfc_avg, Tfc_avg)
        else:
            print("未成功检测到足够的棋盘角点，无法计算平均 R 和 T。")
    finally:
        cap.release()  # 释放摄像头资源
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
