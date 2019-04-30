"""
Alex Brockman - Project #4 Classification

Takes large image and splits it into smaller images with specified sizes

Code samples borrowed from: https://gogul09.github.io/software/image-classification-python
"""

import cv2

#Read image to be split
img = cv2.imread('river_satellite.jpg')
for r in range(0,img.shape[0],256):
    for c in range(0,img.shape[1],256):
        cv2.imwrite(f"dataset/new_train/img{r}_{c}.jpg",img[r:r+256, c:c+256,:])