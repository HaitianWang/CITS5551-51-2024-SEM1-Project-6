import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

red_path = "Red.tif"  # 红色地址
nir_path = "NIR.tif"  # 红外地址

# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

# 分类映射：这里您需要定义3类的条件
ndvi_mapped = np.where(ndvi > 0.54, 0, 
               np.where(ndvi >= 0.41, 1, 
               np.where(ndvi < 0.41, 2, 3)))  # 假设NDVI小于0.41分类为3



# 将NDVI映射矩阵转换为DataFrame
df = pd.DataFrame(ndvi_mapped)

# 定义保存路径
csv_path = "label_matrix.csv"

# 保存为CSV文件
df.to_csv(csv_path, index=False, header=False)

# 定义颜色映射，添加黄色为新的分类3
colors = {0: 'green', 1: 'red', 2: 'white', 3: 'yellow'}

# 创建彩色图像
ndvi_color_mapped = np.zeros((ndvi_mapped.shape[0], ndvi_mapped.shape[1], 3), dtype=np.uint8)

# 对每个值着色
for i, color in colors.items():
    # 获取颜色的RGB值
    color_rgb = np.array(plt.cm.colors.to_rgba(color)[:3]) * 255
    # 根据颜色和值进行索引和赋值
    ndvi_color_mapped[ndvi_mapped == i] = color_rgb

ndvi_color_path = "label_matrix_image.png"

# 显示彩色图像
plt.imshow(ndvi_color_mapped)
plt.axis('off')

# 保存图像到文件
plt.savefig(ndvi_color_path, bbox_inches='tight', pad_inches=0, dpi=96)

# 关闭图形窗口
plt.close()










