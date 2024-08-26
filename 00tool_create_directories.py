"""
    在指定的基础目录下，创建从 `images_cal_01` 到 `images_cal_09` 的文件夹。"""
import os


def create_image_directories(base_dir, start=1, end=9):
    """
    在指定的基础目录下，创建从 `images_cal_01` 到 `images_cal_09` 的文件夹。

    参数:
    - base_dir (str): 基础目录的路径。
    - start (int): 索引起始值（默认为1）。
    - end (int): 索引结束值（默认为9）。
    """
    for i in range(start, end + 1):
        index = f'{i:02}'  # 将数字格式化为两位数，例如 '01', '02', ..., '09'
        image_directory_path = os.path.join(base_dir, 'data', 'capture_images', f'images_cal_{index}')
        os.makedirs(image_directory_path, exist_ok=True)
        print(f"目录 {image_directory_path} 已创建。")


if __name__ == "__main__":
    # 获取当前脚本所在目录
    base_directory = os.path.dirname(__file__)

    # 调用函数创建文件夹
    create_image_directories(base_directory)
