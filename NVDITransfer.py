import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

Image.MAX_IMAGE_PIXELS = None

# 打开红色波段图像和近红外波段图像
image_red = Image.open(r'G:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\NewData\P4M 2021 09 06 Kondinin E2 barley\2021 09 06 Kondinin barley E2\2021 09 06 Kondinin E2\map\result_Red.tif')
image_nir = Image.open(r'G:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\NewData\P4M 2021 09 06 Kondinin E2 barley\2021 09 06 Kondinin barley E2\2021 09 06 Kondinin E2\map\result_NIR.tif')

# 将图像转换为数字矩阵
matrix_red = np.array(image_red)
matrix_nir = np.array(image_nir)

# 计算NDVI
ndvi = (matrix_nir - matrix_red) / (matrix_nir + matrix_red + 0.00001)  # 添加一个小常数以避免除以零

# 输出NDVI矩阵
print(ndvi[4000:4003, 1900:1910])



#ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8) 

#ndvi_image = Image.fromarray(ndvi_normalized)
#ndvi_image.save(r'G:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\ndvi_output.tiff')

#df = pd.DataFrame(ndvi)
#df.to_csv(r'G:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\ndvi_output.csv', index=False)



# norm = Normalize(vmin=-1, vmax=1)
# # 创建一个颜色映射对象
# cmap = plt.cm.RdYlGn  # 这是一个红-黄-绿的颜色映射，适合展示NDVI

# # 应用颜色映射
# ndvi_colored = cmap(norm(ndvi))

# # 显示图像
# plt.imshow(ndvi_colored)
# plt.colorbar()  # 显示颜色条
# plt.title('NDVI Colored Map')
# plt.show()



# plt.imshow(ndvi_colored)
# plt.colorbar()
# plt.title('NDVI Colored Map')
# plt.savefig(r'G:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\colored_ndvi.png')  # 保存为PNG文件
# plt.close()  # 关闭绘图窗口

