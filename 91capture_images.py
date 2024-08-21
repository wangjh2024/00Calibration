import os
import cv2
import glob


# step 1 清理文件夹
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)  # 删除子目录，如果存在的话
    print(f"文件夹 {folder_path} 已清理")


# step 2 图像显示 采集数据
def initialize_camera(camera_index):

    """初始化摄像头并返回对象"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    return cap


def display_and_save(cap, save_path):
    """显示视频并处理保存操作"""
    num = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == ord('s'):
            filename = os.path.join(save_path, f"{num}.jpg")
            cv2.imwrite(filename, frame)
            num += 1
            print(f"保存图像: {filename}")
        elif k == ord('q'):
            break


def cleanup(cap):
    """释放资源"""
    cap.release()
    cv2.destroyAllWindows()


# step 3: 读取保存的图像并进行棋盘格角点检测
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


# step 4 重命名文件夹全部文件
def rename_files(folder_path):
    """按顺序重命名文件夹中的所有文件"""
    files = sorted(os.listdir(folder_path))
    for index, filename in enumerate(files):
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, f"file_{index}.jpg")
        os.rename(old_file_path, new_file_path)
    print(f"文件夹 {folder_path} 文件已重命名")


if __name__ == "__main__":
    # 图像保存路径
    SAVE_PATH = "data/images_cal_09/"
    # os.makedirs(SAVE_PATH, exist_ok=True)

    # step 1 清理文件夹
    # clear_folder(SAVE_PATH)

    # step 2 图像显示 采集数据
    camera_index = 0
    cap = initialize_camera(camera_index)
    if cap:
        display_and_save(cap, SAVE_PATH)
        cleanup(cap)

    # step 3: 读取保存的图像并进行棋盘格角点检测
    # image_files = read_images_from_directory(SAVE_PATH)
    # successful_images = []
    # failed_images = []
    # for image_file in image_files:
    #     img = cv2.imread(image_file)
    #     ret, corners = detect_chessboard_corners(img, chessboard_size)
    #     process_detection_results(image_file, ret, corners, successful_images, failed_images)
    # print(f"成功检测到角点的图像数量: {len(successful_images)}")
    # print(f"未检测到角点的图像数量: {len(failed_images)}")
    # # cv2.destroyAllWindows()


    # step 4 重命名文件夹全部文件
    rename_files(SAVE_PATH)
