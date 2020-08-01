#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

#Creating an Object
#OpenCv- Contrib module should be downloaded for running the three Models


fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();
fgbg2 = cv2.createBackgroundSubtractorMOG2();
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();

# capture frames from a Webcam
cap = cv2.VideoCapture(0);
while(1):
    #read frames
    ret, img = cap.read();
    
    # apply mask for background subtraction
    fgmask1 = fgbg1.apply(img);
    fgmask2 = fgbg2.apply(img);
    fgmask3 = fgbg3.apply(img);
    
    cv2.imshow('Original', img);
    cv2.imshow('MOG' , fgmask1);
    cv2.imshow('MOG2' , fgmask2);
    cv2.imshow('GMG' , fgmask3);
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cap.release();
    cv2.destroyAllWindows();


# In[3]:


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3));

# creating object
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG();

#capture frames from a camera
cap = cv2.VideoCapture(0);
while(1):
    #read frames
    ret, img = cap.read();
    
    #apply mask for backgroud subtraction
    fgmask = fgbg.apply(img);
    
    #with noise frame
    cv2.imshow('GMG noise', fgmask);
    
    #apply transformation to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);
    
    #after removing noise
    cv2.imshow('GMG', fgmask);
    
    key = cv2.waitkey(1)
    if key == ord('q'):
        break
    cap.release();
    cv2.destroyAllWindows();


# In[2]:


## MANASH PRATIM KAKATI
## PG CERTIFICATION IN AI & ML
## E&ICT ACADAMY, IIT GUWAHATI


# In[ ]:




