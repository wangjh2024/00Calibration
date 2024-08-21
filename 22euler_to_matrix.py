import numpy as np
from scipy.spatial.transform import Rotation


def create_rotation_matrix(angles_degrees):
    """
    创建旋转矩阵
    :param angles_degrees: 欧拉角列表，单位为度
    :return: 3x3 旋转矩阵
    """
    rotation = Rotation.from_euler('xyz', angles_degrees, degrees=True)
    return rotation.as_matrix()


def transform_point(point, rotation_matrix, translation_vector):
    """
    对点进行旋转和平移变换
    :param point: 需要变换的点（numpy 数组）
    :param rotation_matrix: 3x3 旋转矩阵
    :param translation_vector: 平移向量（列表或 numpy 数组）
    :return: 变换后的点
    """
    return rotation_matrix.dot(point) + translation_vector


if __name__ == "__main__":
    # 定义平移向量（单位：单位）
    translation_vector = [58.39, -597.91, -2.08]

    # 定义欧拉角（单位：度）
    rotation_angles = [170.67, 179.32, 90.198]

    # 创建旋转矩阵
    rotation_matrix = create_rotation_matrix(rotation_angles)
    print("旋转矩阵 R=\n{}".format(rotation_matrix))

    # 定义需要变换的点（单位：单位）
    point = np.array([-874, 43, 1510])

    # 进行旋转和平移变换
    transformed_point = transform_point(point, rotation_matrix, translation_vector)

    print("变换后的点 p=\n{}".format(transformed_point))
