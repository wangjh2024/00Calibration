import cv2

# 创建 VideoCapture 对象
video_sources = []
for i in range(10):  # 假设最多有 10 个摄像头
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        video_sources.append(i)
    cap.release()

print("可用的摄像头索引:", video_sources)
