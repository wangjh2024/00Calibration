import os
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

def calculate_transformation(corners, board_size, board_scale, K, Distoreffs):
    """
    计算相机相对于棋盘的旋转矩阵和平移向量。
    :param corners: 棋盘角点的像素坐标。
    :param board_size: 棋盘格的大小 (列数, 行数)。
    :param board_scale: 棋盘格的实际尺寸。
    :param K: 相机内参矩阵。
    :param Distoreffs: 畸变系数。
    :return: 相机相对于棋盘的旋转矩阵和平移向量。
    """
    obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

    _, r_vec, t_vec = cv2.solvePnP(obj_points, corners, K, Distoreffs)
    Rcb, _ = cv2.Rodrigues(r_vec)
    Tcb = t_vec.squeeze()

    Rfc = Rfb @ Rcb.T
    Tfc = Rfb @ (-Rcb.T @ Tcb) + Tfb

    return Rfc, Tfc, r_vec, t_vec

def calculate_reprojection_error(obj_points, corners, r_vec, t_vec, K, Distoreffs):
    """
    计算重投影误差，并输出为相对误差。
    :param obj_points: 3D 物体点。
    :param corners: 2D 图像点。
    :param r_vec: 旋转向量。
    :param t_vec: 平移向量。
    :param K: 相机内参矩阵。
    :param Distoreffs: 畸变系数。
    :return: 相对重投影误差。
    """
    img_points, _ = cv2.projectPoints(obj_points, r_vec, t_vec, K, Distoreffs)
    img_points = np.squeeze(img_points)
    corners = np.squeeze(corners)

    if corners.shape != img_points.shape:
        raise ValueError("corners 和 img_points 的形状不匹配")

    error = cv2.norm(corners, img_points, cv2.NORM_L2) / len(img_points)
    max_error = 10.0  # 最大允许误差，作为参考值
    relative_error = error / max_error

    return relative_error

def display_corners(frame, corners, board_size, K, Distoreffs):
    """
    显示检测到的角点并去除图像畸变。
    :param frame: 当前帧图像。
    :param corners: 棋盘角点的像素坐标。
    :param board_size: 棋盘格的大小 (列数, 行数)。
    :param K: 相机内参矩阵。
    :param Distoreffs: 畸变系数。
    """
    cv2.drawChessboardCorners(frame, board_size, corners, True)
    center = tuple(map(int, corners[0].ravel()))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

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

    K = np.array(data['IntrinsicMatrix'])
    Distoreffs_0 = np.array(data['Distortion[k1,k2,k3,p1,p2]'])
    Distoreffs = np.array([Distoreffs_0[0][0], Distoreffs_0[0][1], 0, Distoreffs_0[0][3], Distoreffs_0[0][4]])

    return K, Distoreffs

def save_parameters(file_path, K, Distoreffs, Rfc_avg, Tfc_avg, avg_reprojection_error):
    """
    保存相机参数和计算得到的平均旋转矩阵、平移向量以及平均相对重投影误差到 YAML 文件中。
    :param file_path: 保存的 YAML 文件路径。
    :param K: 相机内参矩阵。
    :param Distoreffs: 畸变系数。
    :param Rfc_avg: 平均旋转矩阵。
    :param Tfc_avg: 平均平移向量。
    :param avg_reprojection_error: 平均相对重投影误差。
    """
    Rfc_avg_euler = Rotation.from_matrix(Rfc_avg).as_euler('xyz', degrees=True)

    parameters = {
        'IntrinsicMatrix_sign': [
            ['fx', '0', 'x0'],
            ['0', 'fy', 'y0'],
            ['0', '0', '1']
        ],
        'IntrinsicMatrix': K.tolist(),
        'RadialDistortion[k1, k2]': Distoreffs[0:2].tolist(),
        'TangentialDistortion[p1, p2]': Distoreffs[3:5].tolist(),
        'Y_RotationEuler': Rfc_avg_euler.tolist(),
        'Y_TranslationVector': Tfc_avg.tolist(),
        'AverageReprojectionErrorRelative': float(avg_reprojection_error)
    }

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)
        print(f"参数已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存参数时出错: {e}")

if __name__ == "__main__":
    board_size = (4, 3)
    board_scale = 24

    Tfb = np.array([-24, -36, 448])
    Rfb = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

    base_directory = os.path.dirname(__file__)

    for sequence_number in range(2, 10):
        sequence_str = f"{sequence_number:02d}"
        input_directory = os.path.join(base_directory, 'data', 'saved_images', f'saved_images_{sequence_str}')
        save_directory = os.path.join(base_directory, 'data', 'params_files_output2',
                                      f'camera_params_{sequence_str}.yaml')

        camera_params_file = os.path.join(base_directory, 'data', 'params_files',
                                          f'camera_params_{sequence_str}.yaml')
        K, Distoreffs = load_camera_parameters(camera_params_file)

        print(f"\n处理文件夹 {sequence_str}")
        print("内参矩阵:\n", K)
        print("畸变系数:\n", Distoreffs, '\n')

        Rfc_sum = np.zeros((3, 3))
        Tfc_sum = np.zeros(3)
        reprojection_errors = []

        obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

        for i, image_file in enumerate([f for f in os.listdir(input_directory) if f.endswith('.png')]):
            image_path = os.path.join(input_directory, image_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flag, corners = cv2.findChessboardCorners(gray, board_size)

                if flag:
                    display_corners(frame, corners, board_size, K, Distoreffs)

                    Rfc, Tfc, r_vec, t_vec = calculate_transformation(corners, board_size, board_scale, K, Distoreffs)

                    reprojection_error = calculate_reprojection_error(obj_points, corners, r_vec, t_vec, K, Distoreffs)
                    reprojection_errors.append(reprojection_error)
                    print(f"图像 {i + 1} 的相对重投影误差: {reprojection_error:.4f}")

                    Rfc_sum += Rfc
                    Tfc_sum += Tfc
                else:
                    print(f"图像 {i + 1} 未检测到棋盘角点。")
            else:
                print(f"无法读取图像文件 (图像 {i + 1})。")

        if len(reprojection_errors) > 0:
            Rfc_avg = Rfc_sum / len(reprojection_errors)
            Tfc_avg = Tfc_sum / len(reprojection_errors)
            avg_reprojection_error = sum(reprojection_errors) / len(reprojection_errors)

            print("\n平均旋转矩阵 R:\n", Rfc_avg)
            print("平均平移向量 T:\n", Tfc_avg)
            print(f"平均相对重投影误差: {avg_reprojection_error:.4f}")

            save_parameters(save_directory, K, Distoreffs, Rfc_avg, Tfc_avg, avg_reprojection_error)
        else:
            print("未计算到有效的重投影误差。")
