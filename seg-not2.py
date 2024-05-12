import pandas as pd
import numpy as np

# 加载CSV文件
df = pd.read_csv('/Users/lunaxu/Downloads/BigRawDataset/ndvi_label.csv')  # 确保路径和文件名正确


# 定义一个函数来寻找第一个包含不是2的值的行或列索引
def find_first_not_two(data, axis=0):
    # 创建一个布尔掩码，其中非2的值为True
    mask = data != 2
    # 检查每一行或每一列是否有任何非2的值
    any_not_two = mask.any(axis=axis)

    # 如果在某一轴上至少有一个True，找到第一个True的位置
    if any_not_two.any():
        for index, value in enumerate(any_not_two):
            if value:  # 找到第一个True的位置
                return index + 1  # 加1使索引从1开始
    else:
        return "所有的值都是2，没有找到非2的数据"


# 计算每个方向的结果
left_to_right = find_first_not_two(df, axis=1)  # 检查行
top_to_bottom = find_first_not_two(df, axis=0)  # 检查列
right_to_left = find_first_not_two(df.iloc[:, ::-1], axis=1)  # 从右到左检查行
bottom_to_top = find_first_not_two(df.iloc[::-1, :], axis=0)  # 从下到上检查列

# 打印结果
print("从左到右第一个包含非2的行索引:", left_to_right)
print("从上到下第一个包含非2的列索引:", top_to_bottom)
print("从右到左第一个包含非2的行索引:", right_to_left)
print("从下到上第一个包含非2的列索引:", bottom_to_top)
