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

        # Make hsv image

        # Threshold using saturation and value

        # Run a few iterations of closing

        # Retrieve external contours

        # Get bounding rects for contours larger than 1000 square pixels
        ROIs = []

        # Iterate through bounding boxes
        for ROI in ROIs:
            # Get corner points

            # Cut out the box, resize it to 32x32 and convert it to RGB

            # Convert to torch tensor, permute so that it is ch X h X w, and divide it by 255

            # Normalize, unsquueze and convert to cuda

            # Forward

            # Get predicted class

            # Draw rectangle and write text

        # Show image
        cv2.imshow("Image",img)
        ret = cv2.waitKey(0)
        if ret == 27:
            exit(0)
