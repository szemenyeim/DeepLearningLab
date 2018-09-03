import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from GuidedBP import GuidedBackprop

text_file = open("imageNet.txt", "r")
classNames = text_file.read().split('\n')

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
model = models.alexnet(pretrained=True).cuda()

# Guided backprop class
GBP = GuidedBackprop(model)

# Control variables
classInd = -1
threshVal = 15

# Open camera stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera")
    exit(-1)

while(1):
    # Read and display image
    ret,img = cap.read()
    if not ret:
        print("Failed to read from camera")
        exit(-1)

    # Create image tensor:
    #BGR - RGB
    # Convert to PIL
    # Apply transform, unsqueeze and convert to cuda
    # Require gradient

    # Get image class if necessary

    # get class name
    text = classNames[classInd]
    cv2.putText(img,text,(20,40), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))

    # Get gradients

    # Sum absolute gradients along channels and convert to opencv

    # Threshold

    # Get contours

    # Find biggest

    # Draw biggest contour

    # Show thresholded gradient image and biggest contour
    cv2.imshow("Video",img)
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
