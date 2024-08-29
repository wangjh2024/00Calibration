import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义棋盘格的大小和方格的实际尺寸
board_size = (4, 3)  # 棋盘格的大小 (列数, 行数)
board_scale = 24  # 棋盘格的实际尺寸 (毫米)

# 创建 3D 物体点的数组 (单位为毫米)
obj_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * board_scale

# 创建 3D 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 点
ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c='r', marker='o')

# 在每个点旁边标记其索引
for i in range(len(obj_points)):
    ax.text(obj_points[i, 0], obj_points[i, 1], obj_points[i, 2], f'{i}', color='black')

# 设置图形的标签
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('3D Object Points with Index Labels')

# 绘制棋盘格的网格线（可选）
for x in range(board_size[0]):
    ax.plot([x * board_scale] * board_size[1],
            np.arange(board_size[1]) * board_scale,
            [0] * board_size[1], 'b-', alpha=0.5)
for y in range(board_size[1]):
    ax.plot(np.arange(board_size[0]) * board_scale,
            [y * board_scale] * board_size[0],
            [0] * board_size[0], 'b-', alpha=0.5)

# 显示图形
plt.show()
