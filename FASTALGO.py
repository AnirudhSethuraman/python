#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Thu May  9 18:48:50 2019

#@author: anirudh"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#img load
image1 = cv2.imread("i.jpg")

#COLOR IMAGE
image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

#BLACK AND WHITE
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
fx, plots = plt.subplots(1,2,figsize=(20,10))

#original image show
plots[0].set_title("RGB IMAGE")
plots[0].imshow(image)

#to black and white
plots[1].set_title("B&W IMAGE")
plots[1].imshow(gray, cmap="gray")

#FAST Algo inz
fast= cv2.FastFeatureDetector_create()

#keypoints with nonmax
key_w_nonmax = fast.detect(gray, None)
fast.setNonmaxSuppression(False)

#Keypoints without nonmax
key_wo_nonmax = fast.detect(gray, None)
imagearray1 = np.copy(image)
imagearray2 = np.copy(image)

#Draw Keypoints
cv2.drawKeypoints(image, key_w_nonmax, imagearray1, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image, key_wo_nonmax, imagearray2, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fx, plots=plt.subplots(1,2, figsize=(20,10))
plots[0].set_title("with nonmax")
plots[0].imshow(imagearray1)


plots[1].set_title("without nonmax")
plots[1].imshow(imagearray2)