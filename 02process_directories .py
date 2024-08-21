import glob
import os

import cv2


def read_images_from_directory(directory_path, file_extension='jpg'):
    """从指定目录中读取图像文件"""
    image_files = glob.glob(os.path.join(directory_path, f'*.{file_extension}'))
    return image_files


def detect_chessboard_corners(image, chessboard_size):
    """检测棋盘格角点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    return ret, corners


def process_detection_results(image_file, ret, corners, successful_images, failed_images):
    """处理角点检测结果"""
    chessboard_size = (4, 3)  # 棋盘格的角点数目 (列数, 行数)

    if ret:
        successful_images.append(image_file)
        # 进一步处理成功的图像
        img = cv2.imread(image_file)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # 可选：显示图像和角点
        # cv2.imshow('Chessboard Corners', img)
        # cv2.waitKey(500)  # 显示角点500毫秒
    else:
        failed_images.append(image_file)

    # 输出结果
    # print(f"图像 {image_file} 检测结果: {'成功' if ret else '失败'}")


def process_and_remove_failed_images(image_file, ret):
    """根据角点检测结果处理图像，删除检测失败的图像"""
    if not ret:
        os.remove(image_file)
        print(f"已删除文件: {image_file}")


def rename_files(folder_path):
    """按顺序重命名文件夹中的所有文件"""
    files = sorted(os.listdir(folder_path))
    for index, filename in enumerate(files):
        old_file_path = os.path.join(folder_path, filename)
        new_file_name = f"{index + 1}.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)

        os.rename(old_file_path, new_file_path)

    num_files = len(files)
    print(f"已重命名: {folder_path} 全部{num_files}个文件\n")


if __name__ == "__main__":
    # 设定起始和结束索引，以及目录的基础路径
    start_index = 1
    end_index = 9
    base_path = "data"

    # 调用处理函数
    for i in range(start_index, end_index + 1):
        directory = os.path.join(base_path, f"images_cal_{i:02d}/")  # 生成目录路径
        if os.path.isdir(directory):  # 检查目录是否存在
            print(f"处理目录: {directory}")

            # 读取图像
            image_files = read_images_from_directory(directory)
            print(f"检测前全部图像数量: {len(image_files)}")

            # 检测角点并根据结果处理图像
            for image_file in image_files:
                img = cv2.imread(image_file)
                ret, corners = detect_chessboard_corners(img, (4, 3))
                process_and_remove_failed_images(image_file, ret)

            # 打印处理结果
            print(f"检测后全部图像数量:  {len(image_files)}")

            # 重命名文件
            rename_files(directory)
        else:
            print(f"目录不存在: {directory}")
