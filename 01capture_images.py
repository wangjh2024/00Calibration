"""
    在指定的基础目录下，创建从 `images_cal_01` 到 `images_cal_09` 的文件夹。"""
import os

import cv2


def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)  # 创建视频捕捉对象，打开指定索引的摄像头
    if not cap.isOpened():  # 检查摄像头是否成功打开
        print("无法打开摄像头")  # 如果未成功打开，打印错误信息
        return None  # 返回 None，表示无法打开摄像头
    return cap  # 返回摄像头对象


def display_and_save(cap, save_path):
    num = 0  # 初始化图像编号
    if not os.path.exists(save_path):  # 检查保存路径是否存在
        os.makedirs(save_path)  # 如果路径不存在，则创建目录
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取摄像头的帧率
    print(f"FPS: {fps}")  # 打印帧率信息

    while True:  # 无限循环，直到用户决定退出
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 检查是否成功读取帧
            print("无法读取帧")  # 如果无法读取帧，打印错误信息
            break  # 退出循环

        cv2.imshow("frame", frame)  # 显示当前帧图像在窗口中
        k = cv2.waitKey(1)  # 等待用户输入的键，1毫秒超时

        if k == ord('s'):  # 如果用户按下 's' 键
            filename = os.path.join(save_path, f"{num}.jpg")  # 构建图像保存的完整路径
            cv2.imwrite(filename, frame)  # 将当前帧图像保存为 JPEG 文件
            num += 1  # 图像编号递增
            print(f"保存图像: {filename}")  # 打印保存图像的文件名
        elif k == ord('q'):  # 如果用户按下 'q' 键
            break  # 退出循环


def cleanup(cap):
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口


if __name__ == "__main__":
    # 图像保存路径
    save_path = "data/images_cal_10/"  # 定义图像保存路径

    # 初始化摄像头
    camera_index = 0  # 定义摄像头索引
    cap = initialize_camera(camera_index)  # 调用函数初始化摄像头

    if cap:  # 如果摄像头成功初始化
        display_and_save(cap, save_path)  # 调用函数显示视频并处理图像保存
        cleanup(cap)  # 调用函数释放摄像头资源
