import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load an color image in grayscale
def readimage(image):
    #    fileinput = input("filename to open : ")
    img = cv2.imread(image)
    #    img = cv2.imread(fileinput, 2)
    print("Dimension:{}, Size:{}, dtype:{} ".format(img.shape, img.size, img.dtype))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
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
    plt.imshow(img)
    plt.show()


def usematplot():
    img = cv2.imread('Monalisa.jpg')
    plt.imshow(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def drawline():
    """ drawig a line on an image"""

    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a diagonal blue line with thickness of 5 px (bgr)
    img = cv2.line(img, (0, 0), (511, 511), (0, 0, 255), 5)
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], False, (0, 255, 255))
    print(pts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (10, 500), font, 2, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.imwrite("Drawing.jpg", img)
    cv2.destroyAllWindows()


def blendimage():
    img1 = cv2.imread('ml..jpg')
    img2 = cv2.imread('opencv.jpg')

    dst = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def merge2images(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    (y1, x1, c1) = img1.shape
    (y2, x2, c2) = img2.shape
    print(img1.shape)
    print(img2.shape)
    if (y1 > y2):
        d1 = (y1 - y2) / 2
        d2 = float(y1) - d1
        img1 = img1[int(np.floor(d1)): int(np.floor(d2))]
    else:
        d1 = (y2 - y1) / 2
        d2 = float(y2) - d1
        img2 = img2[int(np.floor(d1)): int(np.floor(d2))]
    print(d1, d2)
    if (x1 > x2):
        e1 = (x1 - x2) / 2
        e2 = float(x1) - e1
        img1 = img1[:, int(np.floor(e1)):int(np.floor(e2))]
    else:
        e1 = (x2 - x1) / 2
        e2 = float(x2) - e1
        img2 = img2[:, int(np.floor(e1)):int(np.floor(e2))]
    print(e1, e2)

    print(img1.shape)
    print(img2.shape)
    dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotateimage(image1, angle):
    img = cv2.imread(image1, 0)
    rows, cols = img.shape
    cv2.imshow("original", img)
    #    img = img[:cols, :]  # making it a square image
    #    cv2.imshow("reshaped", img)

    M = cv2.getRotationMatrix2D((2.5 * cols / 6, rows / 3), angle, 1)  # rotation point, angle, scale
    dst = cv2.warpAffine(img, M, (rows, cols))
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sampleupdown(file, flag):
    img = cv2.imread(file)
    cv2.imshow("original", img)
    if flag == 1:
        img2 = cv2.pyrDown(img)
    elif flag == 2:
        img2 = cv2.pyrUp(img)
    cv2.imshow("dst", img2)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.imwrite("new" + file, img2)
    cv2.destroyAllWindows()


def histogram(file):
    img = cv2.imread(file, 0)
    cv2.imshow("original", img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #    plt.hist(img.ravel(),256,[0,256])
    #    plt.plot(hist, color = 'b')
    plt.show()
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    print(cdf, cdf_normalized)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    cv2.imshow("dst", img2)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.imwrite(input("filenane:"), img2)
    cv2.destroyAllWindows()


def morph(image, itr, mode):
    img = cv2.imread(image, 0)
    kernel = np.ones((3, 3), np.uint8)

    if (mode == 1):
        output = cv2.erode(img, kernel, iterations=itr)
        print("Eroding image with {}".format(kernel))
    elif (mode == 2):
        output = cv2.dilate(img, kernel, iterations=itr)
        print("Dilating image with {}".format(kernel))
    elif (mode == 3):
        print("Opening image with {}".format(kernel))
        output = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    else:
        print("Closing image with {}".format(kernel))
        output = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("original", img)
    cv2.imshow("After", output)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.imwrite(input("filenane:"), output)
    cv2.destroyAllWindows()


def main():
    import sys
        merge2images(sys.argv[1], sys.argv[2])
    #    rotateimage(sys.argv[1], int(sys.argv[2]))   #filename and angle
    #    sampleupdown(sys.argv[1], int(sys.argv[2]))
    # histogram(sys.argv[1])
    #morph(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    main()


