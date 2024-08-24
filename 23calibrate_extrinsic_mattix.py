import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# 相机内参和畸变系数
K = np.array([[720.4387, 0, 319.0484],  # 焦距 fx 和主点 cx
              [0, 719.4163, 204.2231],  # 焦距 fy 和主点 cy
              [0, 0, 1]])               # 齐次坐标系的第三行
D = np.array([-0.5124, 0.1789, 0.0118, -0.0016])  # 畸变系数 [k1, k2, p1, p2]

# 标定板相对于法兰盘的平移向量 Tfb 和旋转矩阵 Rfb
translation_vector_flange_to_board = np.array([-47, -57, 380])  # 法兰盘相对于标定板的平移向量
euler_angles_flange_to_board = [0, 0, 0]  # 法兰盘相对于标定板的欧拉角（单位：度）
rotation_matrix_flange_to_board = Rotation.from_euler('xyz', euler_angles_flange_to_board, degrees=True).as_matrix()  # 欧拉角转旋转矩阵

def create_rotation_matrix(angles_degrees):
    """
    创建旋转矩阵
    :param angles_degrees: 欧拉角列表，单位为度
    :return: 3x3 旋转矩阵
    """
    rotation = Rotation.from_euler('xyz', angles_degrees, degrees=True)
    return rotation.as_matrix()

def calculate_transformation(corners, board_size, board_scale):
    """
    计算相机相对于法兰盘的旋转矩阵和平移向量
    :param corners: 棋盘角点的像素坐标
    :param board_size: 棋盘格的大小 (列数, 行数)
    :param board_scale: 棋盘格的实际尺寸
    :return: 相机相对于法兰盘的旋转矩阵和平移向量
    """
    obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

    _, r_vec, t_vec = cv2.solvePnP(obj_points, corners, K, D)

    # 将旋转向量转换为旋转矩阵 Rcb (标定板相对于相机的旋转矩阵)
    rotation_matrix_board_to_camera, _ = cv2.Rodrigues(r_vec)
    translation_vector_board_to_camera = t_vec.squeeze()

    # 计算相机相对于法兰盘的旋转和平移矩阵
    rotation_matrix_camera_to_flange = rotation_matrix_flange_to_board @ rotation_matrix_board_to_camera.T
    translation_vector_camera_to_flange = rotation_matrix_flange_to_board @ (-rotation_matrix_board_to_camera.T @ translation_vector_board_to_camera) + translation_vector_flange_to_board

    return rotation_matrix_camera_to_flange, translation_vector_camera_to_flange

def create_homogeneous_matrix(rotation_matrix, translation_vector):
    """
    创建齐次变换矩阵
    :param rotation_matrix: 3x3 旋转矩阵
    :param translation_vector: 3x1 平移向量
    :return: 4x4 齐次变换矩阵
    """
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix
    # 齐次变换矩阵的结构：
    # T_camera_to_flange =
    # [ R11  R12  R13  Tx ]
    # [ R21  R22  R23  Ty ]
    # [ R31  R32  R33  Tz ]
    # [ 0    0    0    1  ]
    #
    # 其中：
    # R11, R12, R13 是旋转矩阵的第一行
    # R21, R22, R23 是旋转矩阵的第二行
    # R31, R32, R33 是旋转矩阵的第三行
    # Tx, Ty, Tz 是平移向量的分量


def main():
    # 定义棋盘格的大小和边长
    board_size = (4, 3)  # 棋盘格的大小 (列数, 行数)
    board_scale = 30  # 棋盘格每个方格的边长

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break

        # 棋盘角点检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag, corners = cv2.findChessboardCorners(gray, board_size)

        if flag:
            # 亚像素角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

            # 计算相机相对于法兰盘的旋转矩阵和平移向量
            rotation_matrix_camera_to_flange, translation_vector_camera_to_flange = calculate_transformation(corners, board_size, board_scale)

            # 创建齐次变换矩阵
            homogeneous_matrix = create_homogeneous_matrix(rotation_matrix_camera_to_flange, translation_vector_camera_to_flange)

            # 输出齐次变换矩阵
            np.set_printoptions(precision=8, suppress=True) #设置输出精度
            print("齐次变换矩阵 T_camera_to_flange=\n{}".format(homogeneous_matrix))

            # 绘制角点
            cv2.drawChessboardCorners(frame, board_size, corners, flag)

        # 显示图像q
        cv2.imshow("frame", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
