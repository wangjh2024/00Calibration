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
        # cv2.imshow('Chessboard Corners', img)
        # cv2.waitKey(500)  # 显示角点500毫秒
    else:
        failed_images.append(image_file)

    # 输出结果
    # print(f"图像 {image_file} 检测结果: {'成功' if ret else '失败'}")


def rename_files(folder_path):
    """按顺序重命名文件夹中的所有文件"""
    files = sorted(os.listdir(folder_path))
    for index, filename in enumerate(files):
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, f"file1_{index}.jpg")
        os.rename(old_file_path, new_file_path)
    print(f"文件夹 {folder_path} 文件已重命名")


if __name__ == "__main__":
    # 设定起始和结束索引，目录的基础路径，以及目录数量
    start_index = 1
    end_index = 9
    base_path = "data"

    # 处理所有目录
    for i in range(start_index, end_index + 1):
        directory = os.path.join(base_path, f"images_cal_{i:02d}/")  # 生成目录路径
        if os.path.isdir(directory):  # 检查目录是否存在
            print(f"处理目录: {directory}")

            # 读取图像
            image_files = read_images_from_directory(directory)
            successful_images = []
            failed_images = []

            # 检测角点并处理结果
            for image_file in image_files:
                img = cv2.imread(image_file)
                ret, corners = detect_chessboard_corners(img, (4, 3))
                process_detection_results(image_file, ret, corners, successful_images, failed_images)

            # 打印处理结果
            print(f"成功检测的图像: {successful_images}")
            print(f"失败检测的图像: {failed_images}")

            # 重命名文件
            rename_files(directory)
        else:
            print(f"目录不存在: {directory}")
