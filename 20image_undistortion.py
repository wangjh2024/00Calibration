import cv2
import numpy as np


def on_mouse(event, x, y, flags, param):
    """处理鼠标点击事件：输出点击位置"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击位置: {(x, y)}")


def display_image_windows(original_image, undistorted_image):
    """创建窗口并显示图像"""
    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", on_mouse)
    cv2.namedWindow("Undistorted Image")
    cv2.setMouseCallback("Undistorted Image", on_mouse)

    # 显示图像
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Undistorted Image", undistorted_image)


if __name__ == "__main__":

    # 图像文件路径
    image_path = r"data\images_cal_01\12.jpg"

    # 相机内参矩阵
    K = np.array([[720.4754, 0, 353.6338],
                  [0, 719.1394, 200.9174],
                  [0, 0, 1.0000]])

    # 畸变系数
    distCoeffs = np.array([-0.5154, 0.2236, 0.0252, -0.0086])

    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像文件: {image_path}")

    # 对图像进行畸变校正
    img_undistorted = cv2.undistort(image, K, distCoeffs)

    # 显示图像
    display_image_windows(image, img_undistorted)

    # 等待用户按键并关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
