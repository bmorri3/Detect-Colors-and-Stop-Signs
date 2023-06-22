
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image = 'BlueBall.jpg'
#image = sys.argv[1]
frame = cv2.imread(image)
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#cv2.imshow("frame", frame)
cv2.waitKey(0)
#plt.imshow(hsv)
plt.show()

# define range of blue color in HSV
lower_blue = np.array([90,50,70])
upper_blue = np.array([128,255,255])
lower_red = np.array([0,90,40])
upper_red = np.array([0,255,255])
lower_green = np.array([30,150,150])
upper_green = np.array([80,255,255])

# Threshold the HSV image to get only blue colors

blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#red_mask = cv2.inRange(hsv, lower_red, upper_red)
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask=blue_mask)
cv2.imshow('frame',frame)
cv2.imshow('blue_mask',blue_mask)
cv2.imshow('res',res)

#res = cv2.bitwise_and(frame,frame, mask=red_mask)
#cv2.imshow('red_mask',red_mask)
#cv2.imshow('res',res)

#res = cv2.bitwise_and(frame,frame, mask=green_mask)
#cv2.imshow('green_mask',green_mask)
#cv2.imshow('res',res)

cv2.waitKey(0)

cv2.destroyAllWindows()
