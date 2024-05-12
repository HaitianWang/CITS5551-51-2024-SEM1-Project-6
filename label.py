import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

red_path = "C:\\Users\\Siyuan\\Downloads\\Big Raw Dataset\\result_Red.tif"  # 红色地址
nir_path = "C:\\Users\\Siyuan\\Downloads\\Big Raw Dataset\\result_NIR.tif"  # 红外地址


# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

# 将大于0.54的值映射为0，0.41-0.54的值映射为1，其他值映射为2
ndvi_mapped = np.where(ndvi > 0.54, 0, np.where(ndvi >= 0.41, 1, np.where(ndvi < 0.41, 2, 3)))
print("矩阵大小：", ndvi_mapped.shape)

# 将3替换为np.nan
ndvi_mapped = ndvi_mapped.astype(np.float32)
ndvi_mapped[ndvi_mapped == 3] = np.nan

# 查找非NaN值的索引
non_nan_indices = np.where(~np.isnan(ndvi_mapped))

# 获取非NaN值的最小行索引和最大行索引
min_row = np.min(non_nan_indices[0])
max_row = np.max(non_nan_indices[0])

# 获取非NaN值的最小列索引和最大列索引
min_col = np.min(non_nan_indices[1])
max_col = np.max(non_nan_indices[1])

# 裁剪矩阵，去掉NaN值的行和列
cropped_matrix = ndvi_mapped[min_row:max_row+1, min_col:max_col+1]
print("矩阵大小：", cropped_matrix.shape)

# 将NDVI映射矩阵转换为DataFrame
df = pd.DataFrame(cropped_matrix)

# 定义保存路径
csv_path = "ndvi_label.csv"

# 保存为CSV文件
df.to_csv(csv_path, index=False, header=False)

# 定义颜色映射
colors = {0: 'green', 1: 'red', 2: 'white'}

# 创建彩色图像
ndvi_color_mapped = np.zeros((cropped_matrix.shape[0], cropped_matrix.shape[1], 3), dtype=np.uint8)

# 对每个值着色
for i, color in colors.items():
    # 获取颜色的RGB值
    color_rgb = np.array(plt.cm.colors.to_rgba(color)[:3]) * 255
    # 根据颜色和值进行索引和赋值
    ndvi_color_mapped[cropped_matrix == i] = color_rgb

ndvi_color_path = "label3.png"

# 显示彩色图像
plt.imshow(ndvi_color_mapped)
plt.axis('off')

# 保存图像到文件
plt.savefig(ndvi_color_path, bbox_inches='tight', pad_inches=0, dpi=96)

# 关闭图形窗口
plt.close()
