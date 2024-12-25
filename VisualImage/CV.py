import cv2
from matplotlib import pyplot as plt
import numpy as np
from django.db import models


def ImageSize(img):
    size = (512, 512)
    return cv2.resize(img, size)


def GrayImage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    img = np.zeros((512, 512, 3), np.uint8)
    drawing = False  # 如果按下鼠标，则为真
    mode = True  # 如果为真，绘制矩形。按 m 键可以切换到曲线
    ix, iy = -1, -1

    Image = cv2.imread('../Photo/RQ.jpg')
    Image = ImageSize(Image)
    GrayImage = GrayImage(Image)
    #res, Img = cv2.threshold(GrayImage, 127,255, cv2.THRESH_BINARY) #二值化
    low = np.array([2, 0, 0])
    upper = np.array([140, 40, 720])
    HsvImage = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    Img = cv2.inRange(HsvImage, low, upper)  #img应为HSV
    cv2.imshow('Image', Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
