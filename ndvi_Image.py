import rasterio
import numpy as np
import matplotlib.pyplot as plt

red_path = "" # input your own path
nir_path = "" # input your own path

# 打开红色通道TIF文件
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# 打开红外TIF文件
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# 计算NDVI
ndvi_normalization = (nir_array - red_array) / (nir_array + red_array + 1e-8)  # 避免分母为零

print("\n Origin Normalizational NDVI data frame \n")
print(ndvi_normalization[2000:3000, 1000:1500])


# 将NDVI映射到0-255的范围
print("\n Mapping NDVI data frame to 0~255 \n")
ndvi_Mapping = ((ndvi_normalization + 1) * 127.5).astype(np.uint8)
print(ndvi_Mapping[2000:3000, 1000:1500])

ndvi_gray_path = "ndvi_gray_saved.tif"

# 保存NDVI_Mapping灰度图像
with rasterio.open(ndvi_gray_path, 'w', driWver='GTiff',
                   width=ndvi_normalization.shape[1], height=ndvi_normalization.shape[0],
                   count=1, dtype=rasterio.uint8,
                   crs=profile['crs'], transform=profile['transform']) as dst:
    dst.write(ndvi_Mapping, 1)

    

#定义颜色映射（colormap），这里使用了一个简单的彩虹颜色映射
cmap = plt.cm.jet

# 创建一个RGB图像，根据NDVI值上色
ndvi_color = cmap(ndvi_Mapping / 255.0)  # 将NDVI值映射到0-1的范围
ndvi_color = (ndvi_color[:, :, :3] * 255).astype(np.uint8)  # 将颜色值转换为整数类型

# 定义保存路径
ndvi_color_path = "ndvi_color_saved.png"

# 显示着色后的NDVI图像
plt.imshow(ndvi_color)
plt.axis('off')

# 保存图像到文件
plt.savefig(ndvi_color_path, bbox_inches='tight', pad_inches=0, dpi=96)

# 关闭图形窗口
plt.close()
