#from enum import Flag
#from pydoc import classname
#from traceback import print_tb
from sre_constants import SUCCESS
import cv2
import numpy as np
import os


path = 'ImagesQuery'
arb = cv2.ORB_create(nfeatures=1000)
images = []
classNames = []
myList = os.listdir(path)
print(myList)
print('Total Classes Dectected', len(myList))
for c1 in myList:
    imgCur = cv2.imread(f'{path}/{c1}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(c1)[0])
print(classNames)

def findDes(images):
    desList=[]
    for img in images:
        kp,des = arb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img, desList,thres=15):
    kp2,des2 = arb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
            
    except:
        pass
    # print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal   
    

desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    id = findID(img2,desList)
    if id != -1:
        cv2.putText(imgOriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    cv2.imshow('img2',imgOriginal)
    cv2.waitKey(1)



# img1 = cv2.imread('ImagesQuery/.jpg',0)
# img2 = cv2.imread('ImageTrain/.jpg',0)

# arb = cv2.ORB_create(nfeatures=1000)

# kp1, des1 = arb.detectAndComupte(img1,None)
# kp2, des2 = arb.detectAndComupte(img2,None)

# # imgKp1 = cv2.drawKeypoints(img1,kp1,None)
# # imgKp2 = cv2.drawKeypoints(img2,kp2,None)



# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# print(len(good))
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)



# # cv2.imshow('kp',imgKp1)
# # cv2.imshow('kp',imgKp2)


# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.waitKey(0)



