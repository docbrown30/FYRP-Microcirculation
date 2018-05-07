# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:26:20 2018

@author: DocBrown

This file experiments with filters from the paper:
   "Image Processing and Machine Learning for Diagnostic Analysis of Microcirculation"
"""

import cv2
import pywt
import numpy as np
#import PIL

#from matplotlib import pyplot as plt

print(cv2.__version__)
vidcap = cv2.VideoCapture('E:\Projects\Videos\movie1.mp4')
success,image = vidcap.read()

#Greyscal conversion of image
img1 = cv2.cvtColor( image,cv2.COLOR_BGR2GRAY )

#Remove defect from image by removing dark pixels
b = img1 < 100  #remove defect using numpy array operation - loop too slow
img1[b] = 255

#histogram equalize the image
img2 = cv2.equalizeHist(img1)

img = img2
#cv2.imwrite("E:/Projects/Results/hist.jpg",equ)


#Implement Gaussian blurring of the image
gblur1= cv2.GaussianBlur(img,(5,5),0)
gblur2= cv2.GaussianBlur(img,(11,11),0)
gblur3= cv2.GaussianBlur(img,(15,15),0)
gblur4= cv2.GaussianBlur(img,(19,19),0)
gblur5= cv2.GaussianBlur(img,(23,23),0)
gblur6= cv2.GaussianBlur(img,(27,27),0)
gblur7= cv2.GaussianBlur(img,(43,53),0)
gblur8= cv2.GaussianBlur(img,(59,59),0)
gblur9= cv2.GaussianBlur(img,(73,73),0)

cv2.imwrite("E:/Projects/Results/gblur1.jpg",gblur1)
cv2.imwrite("E:/Projects/Results/gblur2.jpg",gblur2)
cv2.imwrite("E:/Projects/Results/gblur3.jpg",gblur3)
cv2.imwrite("E:/Projects/Results/gblur4.jpg",gblur4)
cv2.imwrite("E:/Projects/Results/gblur5.jpg",gblur5)
cv2.imwrite("E:/Projects/Results/gblur6.jpg",gblur6)
cv2.imwrite("E:/Projects/Results/gblur7.jpg",gblur7)
cv2.imwrite("E:/Projects/Results/gblur8.jpg",gblur8)
cv2.imwrite("E:/Projects/Results/gblur9.jpg",gblur9)



#Use an unsharp mask to sharpen the image 
unsharp_image = cv2.addWeighted(img, 1.5, gblur1, -0.5, 0, img)
cv2.imwrite("E:/Projects/Results/usharp.jpg",unsharp_image)

unsharp_image1 = cv2.addWeighted(img, 1.5, gblur1, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp1.jpg",unsharp_image1)

unsharp_image2 = cv2.addWeighted(img, 1.5, gblur2, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp2.jpg",unsharp_image2)

unsharp_image3 = cv2.addWeighted(img, 1.5, gblur3, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp3.jpg",unsharp_image3)

unsharp_image4 = cv2.addWeighted(img, 1.5, gblur4, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp4.jpg",unsharp_image4)

unsharp_image5 = cv2.addWeighted(img, 1.5, gblur5, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp5.jpg",unsharp_image5)

unsharp_image6 = cv2.addWeighted(img, 1.5, gblur6, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp6.jpg",unsharp_image6)

unsharp_image7 = cv2.addWeighted(img, 1.5, gblur7, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp7.jpg",unsharp_image7)

unsharp_image8 = cv2.addWeighted(img, 1.5, gblur8, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp8.jpg",unsharp_image8)

unsharp_image9 = cv2.addWeighted(img, 1.5, gblur9, -0.5, 0)
cv2.imwrite("E:/Projects/Results/usharp9.jpg",unsharp_image9)


#Remove noise from image using Mean Denoising - still working out wavelets
denoise1 = cv2.fastNlMeansDenoising(unsharp_image9, None, 10,7,21)
cv2.imwrite("E:/Projects/Results/denoise1.jpg",denoise1)
denoise2 = cv2.fastNlMeansDenoising(unsharp_image9, None, 15,7,21)
cv2.imwrite("E:/Projects/Results/denoise2.jpg",denoise2)
denoise3 = cv2.fastNlMeansDenoising(unsharp_image9, None, 20,7,21)
cv2.imwrite("E:/Projects/Results/denoise3.jpg",denoise3)
denoise4 = cv2.fastNlMeansDenoising(unsharp_image9, None, 25,7,21)
cv2.imwrite("E:/Projects/Results/denoise4.jpg",denoise4)
denoise5 = cv2.fastNlMeansDenoising(unsharp_image9, None, 30,7,21)
cv2.imwrite("E:/Projects/Results/denoise5.jpg",denoise5)
denoise6 = cv2.fastNlMeansDenoising(unsharp_image9, None, 35,7,21)
cv2.imwrite("E:/Projects/Results/denoise6.jpg",denoise6)
denoise7 = cv2.fastNlMeansDenoising(unsharp_image9, None, 40,7,21)
cv2.imwrite("E:/Projects/Results/denoise7.jpg",denoise7)
denoise8 = cv2.fastNlMeansDenoising(unsharp_image9, None, 45,7,21)
cv2.imwrite("E:/Projects/Results/denoise8.jpg",denoise8)
denoise9 = cv2.fastNlMeansDenoising(unsharp_image9, None, 50,7,21)
cv2.imwrite("E:/Projects/Results/denoise9.jpg",denoise9)



vidcap.release()
cv2.destroyAllWindows()