from turtle import shape
import cv2
import numpy as np
#import matplotlib 
#from matplotlib import pyplot as plt


img1 = cv2.imread('ImagesQuery/ittakestwo.jpg',0)
img2 = cv2.imread('ImageTrain/ittakestwo.jpg',0)


#sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()



kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print(des1,shape)

#imgKp1 = cv2.drawKeypoints(img1,kp1,None)
#imgKp2 = cv2.drawKeypoints(img2,kp2,None)


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)


good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
#print(len(good))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

#cv2.imshow('Kp1',imgKp1)
#cv2.imshow('Kp2',imgKp2)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)


cv2.waitKey(0)


"""
Brute-Force Matching with SIFT Descriptors and Ratio Test

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

plt.imshow(img3),plt.show()

"""