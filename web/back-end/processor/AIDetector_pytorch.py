# 导入必要的库
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2
from random import randint


# 定义检测器类
class Detector(object):

    # 初始化方法
    def __init__(self):
        self.img_size = 640  # 图像尺寸
        self.threshold = 0.4  # 检测阈值
        self.max_frame = 160  # 最大帧数
        self.init_model()  # 初始化模型

    # 初始化模型方法
    def init_model(self):
        self.weights = 'weights/best.pt'  # 权重文件路径
        self.device = '0' if torch.cuda.is_available() else 'cpu'  # 检查是否有GPU
        self.device = select_device(self.device)  # 选择设备
        model = attempt_load(self.weights, map_location=self.device)  # 加载模型
        model.to(self.device).eval()  # 将模型设置为评估模式
        model.float()  # 设置模型为浮点精度
        self.m = model  # 保存模型
        self.names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]  # 随机颜色

    # 图像预处理方法
    def preprocess(self, img):
        img0 = img.copy()  # 复制原始图像
        img = letterbox(img, new_shape=self.img_size)[0]  # 调整图像大小
        img = img[:, :, ::-1].transpose(2, 0, 1)  # 调整图像通道顺序
        img = np.ascontiguousarray(img)  # 转换为连续数组
        img = torch.from_numpy(img).to(self.device)  # 转换为PyTorch张量并移动到设备
        img = img.float()  # 转换为浮点数
        img /= 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 添加批次维度
        return img0, img  # 返回原始图像和预处理后的图像

    # 绘制边界框方法
    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # 计算线条厚度
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]  # 获取类别对应的颜色
            print("x1,x2,y1,y2:", x1, x2, y1, y2)

            # 确保 x1 <= x2 和 y1 <= y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, (255, 0, 0), thickness=tl, lineType=cv2.LINE_AA)

            tf = max(tl - 1, 1)  # 字体厚度
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]

            # 标签矩形坐标
            label_rect_top_left = (c1[0], c1[1] - t_size[1] - 3)
            label_rect_bottom_right = (c1[0] + t_size[0], c1[1])
            label_rect_coords = [label_rect_top_left, label_rect_bottom_right]

            print("cls_id", cls_id)
            if (cls_id == "BasalCellCarcinoma"):
                cls_id = "BasalCellCarcinoma\n基底细胞癌"
            if (cls_id == "Melanoma"):
                cls_id = "Melanoma\n黑色素瘤"
            print("label_rect_coords", label_rect_coords)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("SimHei.ttf", 20, encoding="utf-8")

            draw.rectangle(label_rect_coords, fill="white", outline="lightblue", width=35)
            draw.text((c1[0], c1[1] - t_size[1] - 3), cls_id, font=font, fill="black")
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image

    # 检测方法
    def detect(self, im):
        im0, img = self.preprocess(im)  # 预处理图像

        pred = self.m(img, augment=False)[0]  # 模型预测
        pred = pred.float()  # 转换为浮点数
        pred = non_max_suppression(pred, self.threshold, 0.3)  # 非极大值抑制

        pred_boxes = []
        image_info = {}
        count = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 调整坐标

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))
                    count += 1
                    key = '{}-{:02}'.format(lbl, count)
                    image_info[key] = ['{}×{}'.format(x2 - x1, y2 - y1), np.round(float(conf), 3)]

        im = self.plot_bboxes(im, pred_boxes)  # 绘制边界框
        return im, image_info  # 返回处理后的图像和检测信息
