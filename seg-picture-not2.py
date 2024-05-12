from PIL import Image

# 修改Pillow的像素上限
Image.MAX_IMAGE_PIXELS = None

# 加载图片
image_path = '/Users/lunaxu/Downloads/BigRawDataset/result.tif'  # 替换为你的.tif图片路径
image = Image.open(image_path)

# 获取图片的原始尺寸
width, height = image.size

# 设置切割的边界，确保不会超出图片原有的尺寸
left = 267  # 从左边切掉267个像素
top = 258   # 从顶部切掉258个像素
right = width - 267  # 从右边切掉267个像素
bottom = height - 258  # 从底部切掉258个像素

# 确保切割尺寸不超出原图范围
if right <= left or bottom <= top:
    print("裁剪尺寸超出原始图片范围，请调整参数。")
else:
    # 进行裁剪
    cropped_image = image.crop((left, top, right, bottom))

    # 保存裁剪后的图片
    save_path = '/Users/lunaxu/Downloads/BigRawDataset/result1.tif'  # 替换为你想保存新图片的路径
    cropped_image.save(save_path, format='TIFF')

    print("图片已裁剪并保存至：", save_path)
