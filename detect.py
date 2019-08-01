import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import cv2
import sys
import progressbar

root = "E:/Traffic/trafficSigns" if sys.platform == 'win32' else "./data"

net = torch.load(root + '/model.pth')

transform = transforms.Compose([
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])

names = {11:"Bike",52:"No Turn",49:"Parking",45:"Stop",36:"Crossing",17:"One way",8:"Road work",9:"Traffic Lights",22:"Roundabout",43:"Crossing"}

if __name__ == '__main__':
    #val(1)
    path = "./Images/"

    for img_name in glob.glob1(path,"*.JPG"):
        img = cv2.imread(path+img_name)
        img = cv2.resize(img,(1280,960))

        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        img_bin = cv2.inRange(img_hsv,np.array([0,100,80]),np.array([255,255,255]))

        SE = np.ones((3, 3), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, SE,iterations=4)

        _,contours,_ = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ROIs = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]

        for ROI in ROIs:
            x1 = ROI[0]-15
            x2 = ROI[0]+ROI[2]+15
            y1 = ROI[1]-15
            y2 = ROI[1]+ROI[3]+15

            imgROI = cv2.cvtColor(cv2.resize(np.copy(img[y1:y2,x1:x2]),(32,32)),cv2.COLOR_BGR2RGB)
            imgROI = torch.Tensor(imgROI).permute(2,0,1)/255.0
            imgROI = torch.unsqueeze(transform(imgROI),0).cuda()

            out = net(imgROI)

            _,pred = torch.max(out,0)

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(img,names[pred.item()],(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,255,0))


        cv2.imshow("Image",img)
        ret = cv2.waitKey(0)
        if ret == 27:
            exit(0)
