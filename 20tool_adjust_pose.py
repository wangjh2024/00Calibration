import cv2
import numpy as np
from scipy.spatial.transform import Rotation

"""
IntrinsicMatrix:
720.4387         0                0
-0.7903       719.4163            0
319.0484      204.2231          1.0000
RadialDistortion[k1,k2]:
-0.5124    0.1789
TangentialDistortion[p1,p2]:
0.0118   -0.0016
"""
# 相机内参
fx = 720.4387
fy = 719.4163
cx = 319.0484
cy = 204.2231

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
D = np.array([-0.5124, 0.1789, 0.0118, -0.0016])

# 标定板相对于法兰盘的欧拉角和平移矩阵
euler = np.array([0, 0, 0])
Tfb = np.array([-47, -57, 380])
Rfb = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()


# 下标含义:a坐标系相对于b坐标系,表示为Xba
def check_euler_angle():
    angle = [15, 20, 25]
    Rx = Rotation.from_euler('xyz', np.array([angle[0], 0, 0]), degrees=True).as_matrix()
    Ry = Rotation.from_euler('xyz', np.array([0, angle[1], 0]), degrees=True).as_matrix()
    Rz = Rotation.from_euler('xyz', np.array([0, 0, angle[2]]), degrees=True).as_matrix()

    R = Rz.dot(Ry).dot(Rx)
    print("Rz.dot(Ry).dot(Rx)=\n{}".format(R))

    R = Rotation.from_euler('xyz', angle, degrees=True).as_matrix()
    print("Rotation.from_euler('xyz', angle, degrees=True).as_matrix()=\n{}".format(R))


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        # 角点检测
        board_size = (4, 3)
        board_scale = 24
        flag, corners = cv2.findChessboardCorners(frame, board_size)
        if not flag:
            continue
        # 亚像素角点检测
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 迭代次数30，精度0.001
        corners = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners, (3, 3), (-1, -1), criteria)

        # 绘制角点
        cv2.drawChessboardCorners(frame, board_size, corners, flag)
        zero_point = corners[0].squeeze()
        cv2.circle(frame, (int(zero_point[0]), int(zero_point[1])), 5, (0, 0, 255), -1)
        cv2.imshow("frame", cv2.undistort(frame, K, D, None, None))

        # 计算标定板的旋转矩阵和平移矩阵
        obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale
        _, r_vec, t_vec = cv2.solvePnP(obj_points, corners.squeeze(), K, D)
        # 计算重投影误差
        img_points, _ = cv2.projectPoints(obj_points, r_vec, t_vec, K, D)
        error = cv2.norm(corners, img_points, cv2.NORM_L2) / len(img_points)
        print("重投影误差：{:.4f}".format(error))
        # 计算标定板相对于相机坐标系的旋转矩阵
        Rcb, _ = cv2.Rodrigues(r_vec)
        # 计算标定板相对于相机坐标系的平移矩阵
        Tcb = t_vec.transpose().squeeze()
        # 计算相机坐标系相对于标定板坐标系的旋转矩阵
        Rbc = Rcb.transpose()
        # 计算相机坐标系相对于标定板坐标系的平移矩阵
        Tbc = -Rcb.transpose().dot(Tcb)
        # 计算相机坐标系相对于法兰q盘坐标系的旋转矩阵
        Rfc = Rfb.dot(Rbc)
        # 计算相机坐标系相对于法兰盘坐标系的平移矩阵
        Tfc = Rfb.dot(Tbc) + Tfb

        print("R：{}".format(Rotation.from_matrix(Rfc).as_euler('xyz', degrees=True)))
        print("T：{}".format(Tfc))

        if cv2.waitKey(1) == ord('q'):
            break
