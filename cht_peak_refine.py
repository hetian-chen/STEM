import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import time
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

# 存储 Fe 和 Bi 的位置
Fe_positions = np.loadtxt('Fe_positions.txt')
Bi_positions = np.loadtxt('Bi_positions.txt')

# 高斯拟合函数（二维高斯）
def gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xdata_tuple
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# 高斯拟合修正位置函数
# 高斯拟合修正位置函数
area = 10
# 高斯拟合修正位置函数

def fit_and_correct(positions, image_data):
    corrected_positions = []
    fitted_gaussians = []
    # plt.ion()  # 开启交互模式
    num = 0
    # for pos in positions:
    print(f'{len(positions)} need to be done')
    for pos in positions:
        # fig = plt.figure(figsize=(12, 4))
        # ax1 = fig.add_subplot(131)
        # ax2 = fig.add_subplot(132)
        # ax3 = fig.add_subplot(133)
        x, y = pos
        
        # 强制将 y 和 x 转换为整数
        y, x = int(y), int(x)
        # print(pos)

        # 定义拟合的搜索区域，选择 7x7 的区域
        region = image_data[y-area:y+area+1, x-area:x+area+1]  # 取原子附近的 7x7 区域
        # ax1.contourf(region, cmap='viridis')
        
        if region.size == 0:
            continue
        
        # 创建 x 和 y 的网格，注意这里我们是相对于区域的坐标系
        xdata, ydata = np.meshgrid(np.arange(region.shape[1]), np.arange(region.shape[0]))
        
        # 初始参数估计
        amplitude = np.max(region)
        # 在拟合时需要调整坐标，保证 x0, y0 相对于整个图像
        xo, yo = area, area  # 这里是相对于截取的区域的中心坐标
        sigma_x = sigma_y = 12.0  # 初始估算
        offset = np.min(region)
        p0 = [amplitude, xo, yo, sigma_x, sigma_y, 0, offset]  # 初始猜测值
        
        # 尝试进行高斯拟合
        try:
            popt, pcov = curve_fit(gaussian_2d, (xdata, ydata), region.ravel(), p0=p0, method='trf', maxfev=50000,
                                   bounds=((0, 0, 0, 0.1, 0.1, 0, 0), (1, 2*area, 2*area, 100, 100, np.pi, 1)))
            # 拟合参数 popt 为 [amplitude, xo, yo, sigma_x, sigma_y, theta, offset]
            # 判断拟合质量，检查拟合的标准差（sigma_x, sigma_y）是否合理
            perr = np.sqrt(np.diag(pcov))
            # print(popt,perr)
            if np.abs(perr[1]/popt[1]) > 0.1 or np.abs(perr[2]/popt[2]) > 0.1:  # 如果 sigma_x 和 sigma_y 太大，说明拟合不准
                corrected_positions.append((x, y))  # 如果拟合效果不好，保留原位置
                fitted_gaussians.append(popt)  # 拟合失败
                print('f1')
            else:
                # 计算修正后的位置，将拟合结果的 xo, yo 转换回整个图像的坐标系
                corrected_positions.append((x + popt[1] - area, y + popt[2] - area))  # 使用拟合结果更新位置
                fitted_gaussians.append(popt)  # 存储拟合的参数
        except Exception as e:
            corrected_positions.append((x, y))  # 如果拟合失败，保留原位置
            fitted_gaussians.append(None)  # 拟合失败
            print('f2')


        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
        
        # 生成拟合的高斯数据
        xdata, ydata = np.meshgrid(np.arange(0, 2*area+1), np.arange(0, 2*area+1))
        gaussian_data = gaussian_2d((xdata, ydata), *popt).reshape(xdata.shape)
        
        # 绘制拟合的高斯等高线
        # ax2.contourf(xdata, ydata, gaussian_data, cmap='viridis')
        # ax3.contourf(image_data, cmap='viridis')
        if num % 100 == 0:
            print(num)
        num += 1

       #  plt.show()

    return np.array(corrected_positions) #, fitted_gaussians


# fig = plt.figure(figsize=(8, 8))
# ax1 = fig.add_subplot(111)

# ax1.scatter(Fe_positions[:, 0], Fe_positions[:, 1], color='blue', s=10, label='Fe')
# ax1.scatter(Bi_positions[:, 0], Bi_positions[:, 1], color='red', s=10, label='Bi')
# ax1.imshow(enhanced_data, cmap='gray')

# plt.show()

# 对 Fe 和 Bi 原子进行高斯拟合修正
Fe_positions = fit_and_correct(Fe_positions, enhanced_data)
Bi_positions = fit_and_correct(Bi_positions, enhanced_data)

np.savetxt('Fe_positions_refined.txt',Fe_positions)
np.savetxt('Bi_positions_refined.txt',Bi_positions)

# 3D 绘图部分保持不变

# 设置 3D 绘图
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(Fe_positions[:, 0], Fe_positions[:, 1], color='blue', s=10, label='Fe')
ax1.scatter(Bi_positions[:, 0], Bi_positions[:, 1], color='red', s=10, label='Bi')
ax1.imshow(enhanced_data, cmap='gray')
plt.savefig('peak_refined.png',dpi=300)
plt.show()
