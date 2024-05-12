import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

red_path = r"F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\2021 09 06 Test Images & Labels\result_Red.tif"  # 红色地址
nir_path = r"F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\2021 09 06 Test Images & Labels\result_NIR.tif"  # 红外地址

# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

ndvi_mapped = np.where(ndvi > 0.54, 0, np.where(ndvi >= 0.41, 1, np.where(ndvi < 0.41, 2, 3)))


# 将NDVI映射矩阵转换为DataFrame
df = pd.DataFrame(ndvi_mapped)

# 定义保存路径
csv_path = r"E:\UWA\GENG 5551\GENG 5551 - NDVI Label v3.csv"

# 保存为CSV文件
df.to_csv(csv_path, index=False, header=False)