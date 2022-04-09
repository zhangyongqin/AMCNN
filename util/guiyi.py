

import os
import re

import cv2
import numpy as np

ROOT_DIR = os.path.abspath("../")
img_path = os.path.join("2d_images")
images_train = os.listdir(img_path)
imglist = sorted(images_train, key=lambda i: int(re.match(r'(\d+)', i).group()))
print(imglist)

i = 0
for img in imglist:

    if img.endswith('.tif'):
        print(i)
        src = os.path.join(os.path.abspath(img_path), img)  # 原先的图片名


        image = cv2.imread(src, 0)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 限制对比度的自适应阈值均衡化
        dst = clahe.apply(image)
        # 使用全局直方图均衡化
        equa = cv2.equalizeHist(image)
        # print(dst)
        cv2.imwrite("test2/" + str(i) + '.png', equa)
        i=i+1








