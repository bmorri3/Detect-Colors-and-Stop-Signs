import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

#image = "BlueBall.jpg"
#image = "ColorBlobs.jpg"
image = "ColorBlobs5.jpg"
def readimage(image):
    #    fileinput = input("filename to open : ")
    img = cv2.imread(image)
    #    img = cv2.imread(fileinput, 2)
    print("Dimension:{}, Size:{}, dtype:{} ".format(img.shape, img.size, img.dtype))

    #cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    #cv2.imshow('Display', img)
    print("Enter s to save and esc to exit")
    k = cv2.waitKey(0)
    if k == 27:
        print("Exiting..")
    elif k == ord('s'):
        filename = input("enter the filename to save: ")
        cv2.imwrite(filename, img)
    else:
        print("Enter esc to escape and s to save.. closing image")
    cv2.destroyAllWindows()
    #plt.imshow(img)
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()

def detectColor(image):
    # image = sys.argv[1]
    frame = cv2.imread(image)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("frame", frame)
    cv2.waitKey(0)
    #plt.imshow(hsv)
    plt.show()
    cv2.waitKey(0)

    red = 0
    green = 60
    blue = 120
    yellow = 30
    ...
    sensitivity = 25

    color = red

    #lower_color = np.array([color - sensitivity, 100, 100])
    #upper_color = np.array([color + sensitivity, 255, 255])

    # Light green (1 of 5)
    # Close. Need to refine. Maybe in HoughCircle
    lower_color = np.array([35, 60, 150])
    upper_color = np.array([50, 120, 250])

    # Blue (2 of 5)
    # Pretty good
    lower_color = np.array([85, 100, 150])
    upper_color = np.array([115, 250, 250])

    # Red (3 of 5)
    # Perfect
    lower_color = np.array([160, 60, 150])
    upper_color = np.array([170, 120, 250])

    # Light yellow (4 of 5)
    # Close. Need to refine. Maybe in HoughCircle
    lower_color = np.array([25, 60, 150])
    upper_color = np.array([35, 120, 250])

    # Orange (5 of 5)
    # Circle is off. Need to refine. Maybe in HoughCircle
    lower_color = np.array([10, 50, 200])
    upper_color = np.array([19, 150, 250])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # define range of blue color in HSV
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    lower_red = np.array([0, 90, 40])
    upper_red = np.array([0, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only blue colors

    #mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    #cv2.imshow('res', res)
    #cv2.waitKey(0)
    circle = circleHough(res)

    # Overlay circle and center
    cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 255, 255), 2)
    cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Circled dot', frame)
    cv2.waitKey(0)

    # red_mask = cv2.inRange(hsv, lower_red, upper_red)
    #green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # res = cv2.bitwise_and(frame,frame, mask=red_mask)
    # cv2.imshow('red_mask',red_mask)
    # cv2.imshow('res',res)

    # res = cv2.bitwise_and(frame,frame, mask=green_mask)
    # cv2.imshow('green_mask',green_mask)
    # cv2.imshow('res',res)

    cv2.destroyAllWindows()

def circleHough(image):
    cimg = image
    #img = cv2.imread(image, 1)
    # Blurred it below from 5 to 7 to find only one circle
    # She said doing the ColorRecognition.py to find just blue, and then do HoughCircles() and it would only find
    # the circle
    blur = cv2.medianBlur(image, 9)
    #blur = cv2.medianBlur(image,5)
    #cimg = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(blur, 25, 150)
    #cv2.imshow('Blur', image)
    #cv2.waitKey(0)
    cv2.imshow('edges', edges)

    # Changing param2 lower finds more circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=60, param2=20, minRadius=30, maxRadius=200)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (255, 255, 255), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    #cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circles[0][0]

readimage(image)
detectColor(image)
