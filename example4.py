import numpy as np
import cv2

cap = cv2.VideoCapture('tt5.mp4')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)#varThreshold = 192)

lower_orange = np.array([0,55,150])
upper_orange = np.array([20,110,255])


while(1):
    ret, frame = cap.read()



    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('bgs_fgmask.png',fgmask)
    cv2.imshow('frame1',fgmask)
    cv2.imwrite('bgs_frame1.png', frame)
    cv2.imshow('frame2', frame)

    kernel = np.ones((5, 5), np.uint8)
    blueMaskDilated = cv2.dilate(fgmask, kernel)
    cv2.imshow("Dilate", blueMaskDilated)
    cv2.imshow('bgs_Dilate',blueMaskDilated)

    res = cv2.bitwise_and(frame, frame, mask=blueMaskDilated)
    cv2.imshow("res", res)
    cv2.imwrite('bgs_res1.png', res)

    hsvmask = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    blueMask = cv2.inRange(hsvmask, lower_orange, upper_orange)
    cv2.imshow('bgs_orangeMask1', frame)
    cv2.imshow("bgs_orangeMask1", blueMask)
    cv2.imwrite('bgs_orangeMask1.png', blueMask)
    cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.waitKey()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()