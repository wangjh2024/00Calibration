import os
import cv2


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


def clear_folder(folder_path):
    """清理文件夹中的所有内容"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)  # 删除子目录，如果存在的话
    print(f"文件夹 {folder_path} 已清理")


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
    SAVE_PATH = "data/images_cal_01/"

    # step 1 清理文件夹
    # clear_folder(SAVE_PATH)

    # step 2 图像显示 采集数据
    # camera_index = 0
    # cap = initialize_camera(camera_index)
    # if cap:
    #     display_and_save(cap, SAVE_PATH)
    #     cleanup(cap)

    # step 3 清理并重命名文件夹
    # rename_files(SAVE_PATH)
