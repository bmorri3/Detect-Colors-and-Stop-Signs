import cv2
import numpy as np
import sys

image = 'BlueBall.jpg'
cimg = cv2.imread(image)
img = cv2.imread(image, 1)
# Blurred it from 5 to 7 to find only one circle
# She said doing the ColorRecognition.py to find just blue, and then do HoughCircles() and it would only find
# the circle
blur = cv2.medianBlur(img,7)
#blur = cv2.medianBlur(img,5)
#cimg = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
edges = cv2.Canny(blur,70, 150)
cv2.imshow('Blur',img)
cv2.waitKey(0)
cv2.imshow('edges',edges)
cv2.waitKey(0)
circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=35,minRadius=100,maxRadius=200)
#circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,250,param1=60,param2=32,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(255,255,255),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
