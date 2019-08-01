import torch
import torchvision
import torchvision.transforms as transforms
from model import ConvNet
import cv2

if __name__ == '__main__':
    img = cv2.imread("./input_images/in1.jpg")

    cv2.imshow("Image",img)
    cv2.waitKey(0)
