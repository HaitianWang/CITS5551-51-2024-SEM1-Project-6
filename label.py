import rasterio
import numpy as np
import matplotlib.pyplot as plt

red_path = ""  # 红色地址
nir_path = ""  # 红外地址

# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

ndvi_mapped = np.where(ndvi > 0.6, 0, np.where(ndvi >= 0.3, 1, 2))

# 定义颜色映射
colors = {0: 'red', 1: 'green', 2: 'blue'}

# 创建彩色图像
ndvi_color_mapped = np.zeros((ndvi_mapped.shape[0], ndvi_mapped.shape[1], 3), dtype=np.uint8)
# print(ndvi_color_mapped[2000:3000, 1000:1500])
print("ndvi_mapped形状:", ndvi_mapped.shape)
print("ndvi_color_mapped形状:", ndvi_color_mapped.shape)

# 对每个值着色
for i, color in colors.items():
    # 获取颜色的RGB值
    color_rgb = np.array(plt.cm.colors.to_rgba(color)[:3]) * 255
    # 根据颜色和值进行索引和赋值
    ndvi_color_mapped[ndvi_mapped == i] = color_rgb

# 显示彩色图像
plt.imshow(ndvi_color_mapped)
plt.axis('off')
plt.show()
