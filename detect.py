import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import cv2
import sys
import progressbar

# Root directory
root = "E:/Traffic/trafficSigns" if sys.platform == 'win32' else "./data"

# Load saved model
net = torch.load(root + '/model.pth')

# Normalization
transform = transforms.Compose([
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])

# Relevant class names
classNames = ['Bump', 'Bumpy road', 'Bus stop', 'Children', 'Crossing (blue)', 'Crossing (red)', 'Cyclists',
              'Danger (other)', 'Dangerous left turn', 'Dangerous right turn', 'Give way', 'Go ahead', 'Go ahead or left',
              'Go ahead or right', 'Go around either way', 'Go around left', 'Go around right', 'Intersection', 'Limit 100',
              'Limit 120', 'Limit 20', 'Limit 30', 'Limit 50', 'Limit 60', 'Limit 70', 'Limit 80', 'Limit 80 over',
              'Limit over', 'Main road', 'Main road over', 'Multiple dangerous turns', 'Narrow road (left)',
              'Narrow road (right)', 'No stopping', 'No takeover', 'No takeover (truck)', 'No takeover (truck) end',
              'No takeover end', 'No turn', 'No turn (both directions)', 'No turn (truck)', 'No waiting', 'One way road',
              'Parking', 'Priority', 'Road works', 'Roundabout', 'Slippery road', 'Stop', 'Traffic light', 'Train crossing',
              'Train crossing (no barrier)', 'Turn left', 'Turn right', 'Wild animals']

if __name__ == '__main__':

    # Path for images
    path = "./Images/"

    # Iterate through all images
    for img_name in glob.glob1(path,"*.JPG"):

        # Read and resize image to 1280x960
        img = cv2.imread(path+img_name)
        img = cv2.resize(img,(1280,960))

        # Make hsv image
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # Threshold using saturation and value
        img_bin = cv2.inRange(img_hsv,np.array([0,100,80]),np.array([255,255,255]))

        # Run a few iterations of closing
        SE = np.ones((3, 3), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, SE,iterations=4)

        # Retrieve external contours
        contours,_ = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding rects for contours larger than 1000 square pixels
        ROIs = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]

        # Iterate through bounding boxes
        for ROI in ROIs:
            # Get corner points
            x1 = ROI[0]-15
            x2 = ROI[0]+ROI[2]+15
            y1 = ROI[1]-15
            y2 = ROI[1]+ROI[3]+15

            # Cut out the box, resize it to 32x32 and convert it to RGB
            imgROI = cv2.cvtColor(cv2.resize(np.copy(img[y1:y2,x1:x2]),(32,32)),cv2.COLOR_BGR2RGB)

            # Convert to torch tensor, permute so that it is ch X h X w, and divide it by 255
            imgROI = torch.Tensor(imgROI).permute(2,0,1)/255.0

            # Normalize, aunsquueze and convert to cuda
            imgROI = torch.unsqueeze(transform(imgROI),0).cuda()

            # Forward
            out = net(imgROI)

            # Get predicted class
            _,pred = torch.max(out,0)

            # Draw rectangle and write text
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(img,classNames[pred.item()],(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,255,0))

        # Show image
        cv2.imshow("Image",img)
        ret = cv2.waitKey(0)
        if ret == 27:
            exit(0)
