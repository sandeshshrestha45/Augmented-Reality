# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 07:48:17 2021

@author: sandesh
"""

import cv2
import numpy as np

#initialize ORB feature detector
MIN_MATCHES=20
detector=cv2.ORB_create(nfeatures=5000)

#prepare FLANN based matcher
index_params=dict(algorithm=1,trees=3)
search_params=dict(checks=100)
flann=cv2.FlannBasedMatcher(index_params, search_params)


#Load image and keypoints
def load_input():
    input_image=cv2.imread('img.jpg')
    input_image=cv2.resize(input_image, (400,500),interpolation=cv2.INTER_AREA)
    gray_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #find keypoints with ORB
    keypoints, descriptors=detector.detectAndCompute(gray_image, None)
    
    return  gray_image, keypoints, descriptors


#compute matches between train and query descriptors
def compute_matches(descriptors_input,descriptors_output):
    if(len(descriptors_output)!=0 and len(descriptors_input)!=0):
        matches=flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
        good=[]
        for m,n in matches:
            if m.distance<0.68*n.distance:
                good.append([m])
        return good
    else:
        return None



if __name__=='__main__':
    #get information from the input image
    input_image, input_keypoints, input_descriptors=load_input()
    #get camera ready
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    while(ret):
        ret,frame=cap.read()
        #condition for error escape
        if(len(input_keypoints)<MIN_MATCHES):
            continue
        #resize input frame for fast computation
        frame=cv2.resize(frame,(700,600))
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #computing and matching the keypoints of input image and query image
        output_keypoints,output_descriptors=detector.detectAndCompute(gray_frame, None)
        matches=compute_matches(input_descriptors,output_descriptors)
        if(matches!=None):
            output_final=cv2.drawMatchesKnn(input_image,input_keypoints, frame, output_keypoints, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Final Output', output_final)
        else:
            cv2.imshow('Final Output',frame)
        key=cv2.waitKey(5)
        if(key==27):
            break
