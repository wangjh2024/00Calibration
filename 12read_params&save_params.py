import os

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


def save_camera_parameters(file_path, K, distCoeffs):
    """将修改后的相机参数保存到新的 YAML 文件"""
    data = {
        'IntrinsicMatrix': K.tolist(),
        'Distortion[k1,k2,k3,p1,p2]': distCoeffs.tolist()
    }
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)


def process_and_save_files(input_directory, output_directory):
    """处理目录下的所有 YAML 文件并保存到新的目录"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.yaml'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            print(f"处理文件: {input_file_path}")

            try:
                # 加载相机参数
                K, distCoeffs = load_camera_parameters(input_file_path)

                # 打印内参矩阵和畸变系数
                print("内参矩阵:\n", K)
                print("畸变系数:\n", distCoeffs)

                # 确保 distCoeffs 包含至少 5 个元素
                if len(distCoeffs[0]) < 5:
                    raise ValueError(f"畸变系数数组长度不足: {len(distCoeffs)}，文件: {input_directory}")

                # 去掉 k3
                distCoeffs_1 = np.array([distCoeffs[0][0], distCoeffs[0][1], 0, distCoeffs[0][3], distCoeffs[0][4]])
                distCoeffs_new = np.array([distCoeffs_1])

                # 打印修改后的 distCoeffs
                print(f"修改后的畸变系数 (file: {output_directory}):", distCoeffs_new)

                # 保存修改后的参数
                save_camera_parameters(output_file_path, K, distCoeffs)

                print(f"保存修改后的参数到: {output_file_path}\n\n\n")
            except Exception as e:
                print(f"处理文件 {input_file_path} 时发生错误: {e}")


if __name__ == "__main__":
    # 设定基础目录路径
    base_directory = os.path.dirname(__file__)
    input_directory = os.path.join(base_directory, 'data', 'camera_params_files')
    output_directory = os.path.join(base_directory, 'data', 'camera_params_files_new')

    # 处理并保存所有 YAML 文件
    process_and_save_files(input_directory, output_directory)
