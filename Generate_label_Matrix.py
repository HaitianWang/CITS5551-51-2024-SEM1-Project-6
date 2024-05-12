import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

red_path = "result_Red.tif"  # 红色地址
nir_path = "result_NIR.tif"  # 红外地址

# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

ndvi_mapped = np.where(ndvi > 0.54, 0, np.where(ndvi >= 0.41, 1, 2))


# 将NDVI映射矩阵转换为DataFrame
df = pd.DataFrame(ndvi_mapped)

# 定义保存路径
csv_path = "ndvi_label.csv"

# 保存为CSV文件
df.to_csv(csv_path, index=False, header=False)
