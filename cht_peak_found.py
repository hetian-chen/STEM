import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import os, sys

os.chdir(sys.path[0])

# 读取 dm4 文件
signal = hs.load('JEOL ADF_15MX_0483.dm4')
data = signal.data

# 对图像进行高斯滤波，减少噪声
smoothed_data = gaussian_filter(data, sigma=5)

# 归一化图像数据
normalized_data = (smoothed_data - np.min(smoothed_data)) / (np.max(smoothed_data) - np.min(smoothed_data))

# 使用自适应对比度增强
enhanced_data = rescale_intensity(normalized_data, in_range=(0, 1), out_range=(0, 1))

# 降低阈值以允许较暗的点被识别
coordinates = peak_local_max(enhanced_data, min_distance=5, threshold_abs=0.05)

# 设定亮度阈值来区分 Fe 和 Bi
threshold_brightness = 0.7  # 根据实际情况调整该值，较低亮度为 Fe，较高亮度为 Bi

# 存储 Fe 和 Bi 的位置
Fe_positions = []
Bi_positions = []

# 高斯拟合函数（二维高斯）
def gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xdata_tuple
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# 根据亮度值分类
for coord in coordinates:
    y, x = coord
    if enhanced_data[y, x] < threshold_brightness:  # Fe 比较暗
        Fe_positions.append((x, y))
    else:  # Bi 比较亮
        Bi_positions.append((x, y))

# 检查 Fe 和 Bi 的原子位置
print(f"Fe_positions count: {len(Fe_positions)}")
print(f"Bi_positions count: {len(Bi_positions)}")

# 转换为 NumPy 数组并确保是二维的
Fe_positions = np.array(Fe_positions)
Bi_positions = np.array(Bi_positions)

# 如果 Fe_positions 或 Bi_positions 为空，避免绘制
if Fe_positions.size > 0:
    Fe_positions = Fe_positions.reshape(-1, 2)  # 确保是二维数组
if Bi_positions.size > 0:
    Bi_positions = Bi_positions.reshape(-1, 2)  # 确保是二维数组

np.savetxt('Fe_positions.txt',Fe_positions)
np.savetxt('Bi_positions.txt',Bi_positions)
# 可视化结果
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.2, 0.2, 0.2, 0.6])
ax2 = fig.add_axes([0.45, 0.2, 0.2, 0.6])
ax3 = fig.add_axes([0.7, 0.2, 0.2, 0.6])

# 绘制 Fe 和 Bi 的位置
if Fe_positions.size > 0 and Bi_positions.size > 0:
    ax1.scatter(Fe_positions[:, 0], Fe_positions[:, 1], color='blue', s=10, label='Fe')
    ax1.scatter(Bi_positions[:, 0], Bi_positions[:, 1], color='red', s=10, label='Bi')

    ax2.scatter(Fe_positions[:, 0], Fe_positions[:, 1], color='blue', s=10, label='Fe')
    ax3.scatter(Bi_positions[:, 0], Bi_positions[:, 1], color='red', s=10, label='Bi')

ax1.imshow(enhanced_data, cmap='gray')
ax2.imshow(enhanced_data, cmap='gray', vmin=-2, vmax=-1)
ax3.imshow(enhanced_data, cmap='gray', vmin=-2, vmax=-1)
plt.savefig('peak_found.png',dpi=300)
plt.show()
