import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from GuidedBP import GuidedBackprop

# Convert gradient image to an opencv one and normalize it
def grad2OpenCV(gradImg):
    gradImg /= gradImg.max()
    return np.uint8(gradImg * 255)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768))]
)

# Used model
model = models.vgg19_bn(pretrained=True).cuda()

# Guided backprop class
GBP = GuidedBackprop(model)

# Control variables
classInd = -1
threshVal = 10

# Open camera stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera")
    exit(-1)

while(1):
    # Read and display image
    img = cap.read()
    #img = cv2.imread("./input_images/snake.jpg")
    cv2.imshow("Video",img)

    # Create image tensor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)
    imageT = transform(pil_im).unsqueeze(0).cuda()
    imageT.requires_grad_(True)

    # Get image class if necessary
    if classInd == -1:
        _, classInd = torch.max(model(imageT),1)
        classInd = classInd[0].cpu().item()

    # Get gradients
    guided_grads = GBP.generate_gradients(imageT, classInd)

    # Sum absolute gradients along channels and convert to opencv
    gradImg = grad2OpenCV(np.sum(np.abs(guided_grads), axis=0))

    # Threshold
    _,binImage = cv2.threshold(gradImg,threshVal,255,cv2.THRESH_BINARY)

    # Get contours
    _, contours, _ = cv2.findContours(binImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Find biggest
    index = -1
    maxSize = -1
    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)
        if area > maxSize:
            maxSize = area
            index = i

    # Draw biggest contour
    contImage = np.zeros(binImage.shape,binImage.dtype)
    if index > -1:
        cv2.drawContours(contImage,contours,index,255,-1)

    # Show thresholded gradient image and biggest contour
    cv2.imshow('Guided grads', binImage)
    cv2.imshow('Object', contImage)

    # Key controls
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 13:
        classInd = -1
    elif k == 43:
        threshVal += 1
        print(("Threshold increased to: %d" %threshVal))
    elif k == 45:
        threshVal -= 1
        print(("Threshold decreased to: %d" %threshVal))
