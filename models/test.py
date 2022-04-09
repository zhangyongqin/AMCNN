import cv2
import tensorflow as tf


img1 = cv2.imread("../deform/train/1.tif",0)

x,y = img1.shape

for index1 in range(x):
	for index2 in range(y):
		print(img1[index1][index2])
