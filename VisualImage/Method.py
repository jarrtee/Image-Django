import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt


#定义class
class Visual:
    def __init__(self, image):
        self.image = image

    def ImageGray(self, img):  #图片灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def ImageSize(self, img, height, width):
        size = (width, height)
        return cv2.resize(img, size)


if __name__ == '__main__':
    Image = cv2.imread('../Photo/RQ.jpg')
    Image = Visual(Image).ImageGray(Image)
    cv2.imshow('Image', Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
