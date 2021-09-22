# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 07:38:06 2021

@author: sandesh
"""

import cv2
import numpy as np


input_image=cv2.imread('img.jpg')
input_image=cv2.resize(input_image,(400,550),interpolation=cv2.INTER_AREA)
gray_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

#initiate ORB object
orb=cv2.ORB_create(nfeatures=1000)

#find keypoints using ORB
keypoints, descriptors=orb.detectAndCompute(gray_image, None)

#draw only the location of the keypoints without size
final_keypoints=cv2.drawKeypoints(gray_image,keypoints,input_image,(0,255,0))

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
