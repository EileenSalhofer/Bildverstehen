import numpy as np
import cv2
import copy

debug_images = True
# Blues has HSV bound values
lower_blue = np.array([40,84,100])
upper_blue = np.array([124,228,255])

# White has RGB bound values
lower_white = np.array([180,180,180], dtype = "uint8")
upper_white = np.array([255,255,255], dtype = "uint8")

def corners(frame):

    #conrners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)


    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', frame)
    cv2.imwrite('dst.png', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def get_biggest_contour(mask):
    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    marker = cv2.minAreaRect(c)
    return c


def get_roi(contoures, frame, tolerance):

    if len(contoures) != 0:
        if debug_images:
            temp = frame.copy()
            cv2.drawContours(temp, contoures, -1, 255, 3)
            cv2.imshow("image", temp)
            cv2.waitKey(0)

        x, y, w, h = cv2.boundingRect(contoures)

        # increase size for more tolerance
        height, width = frame.shape[:2]
        tolerance_x = int((w / height) * tolerance)
        tolerance_y = int((h / height) * tolerance)
        x = x - tolerance_x
        w = w + tolerance_x * 2
        y = y - tolerance_y
        h = h + tolerance_y * 2

        if debug_images:
            temp = frame.copy()
            # draw the contour (in green)
            cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("image", temp)
            cv2.waitKey(0)

        return frame[y:(y + h), x:(x + w)], {"x":x, "y":y, "w":w, "h":h }

def main():
    camera = cv2.VideoCapture("tt5.mp4")

    # keep looping
    while(camera.isOpened()):
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = cv2.imread("ttp4.bmp")
        #resize frame
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        cv2.imshow("Frame", frame)
        cv2.imwrite('Frame.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #corners(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        marker = cv2.minAreaRect(c)

        box = cv2.boxPoints(marker)
        temp = [];
        for b in box:
            temp.append([b])
        box = np.int0(temp)

        #hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Set threshold to filter only green color & Filter it

        whiteMask = cv2.inRange(frame, lower_white, upper_white)

        hsvmask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blueMask = cv2.inRange(hsvmask, lower_blue, upper_blue)
        cv2.imshow("blueMask1", blueMask)
        cv2.imwrite('blueMask1.png', blueMask)
        cv2.waitKey(0)



        # Dilate
        blueMaskDilated = blueMask
        kernel = np.ones((15, 15), np.uint8)
        blueMaskDilated = cv2.dilate(blueMask, kernel)
        cv2.imshow("Dilate", blueMaskDilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        roi, coordinates = get_roi(get_biggest_contour(blueMaskDilated), frame.copy(), 25)
        roi, coordinates = get_roi(get_biggest_contour(blueMaskDilated), frame.copy(), 25)
        hsvmask = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blueMask = cv2.inRange(hsvmask, lower_blue, upper_blue)
        cv2.imshow("roi1", roi)
        cv2.imwrite('roi1.png', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("blueMask2", blueMask)
        cv2.imwrite('blueMask2.png', blueMask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        kernel = np.ones((50, 50), np.uint8)
        blueMaskDilated = cv2.dilate(blueMask, kernel)
        cv2.imshow("blueMaskDilated1", blueMaskDilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        res = cv2.bitwise_and(roi,roi, mask= blueMaskDilated)
        cv2.imshow("res1", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        whiteMask = cv2.inRange(res, lower_white, upper_white)
        cv2.imshow("whiteMask1", whiteMask)
        cv2.waitKey(0)
        cv2.imwrite('whiteMask1.png', whiteMask)
        cv2.destroyAllWindows()
        roi, coordinates = get_roi(get_biggest_contour(whiteMask), roi, 10)

        corners(cv2.cvtColor(whiteMask, cv2.COLOR_GRAY2BGR))
        cv2.imshow("image", whiteMask)
        cv2.waitKey(0)

        """
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(whiteMask,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(whiteMask,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("image", whiteMask)
        cv2.waitKey(0)

        """
        #kernel = np.ones((33,33),np.uint8)
        #erosion = cv2.erode(whiteMaskDilated,kernel,iterations = 2)
        #dilation = cv2.dilate(erosion,kernel,iterations = 1)
        #cv2.imshow("opening", erosion)
        #cv2.waitKey(0)

        #whiteMaskDilated = cv2.GaussianBlur(whiteMaskDilated, (5, 5), 0)
        #whiteMaskDilated = cv2.Canny(whiteMaskDilated, 35, 125)
        #cv2.imshow("image", whiteMaskDilated)
        #cv2.waitKey(0)


        """
        for idx, cont in enumerate(cnts):
            temp = copy.deepcopy(cv2.cvtColor(whiteMask, cv2.COLOR_GRAY2RGB))
            cv2.drawContours(temp, cnts, idx, (0, 255, 0), 2)
            cv2.imshow("image", temp)
            cv2.imshow("image", frame)
            cv2.waitKey(0)
        """
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)


        # show the frame to our screen
        #qcv2.imshow("Frame", edged)
        cv2.imshow("image", frame)
        cv2.waitKey(0)

        if (cv2.waitKey(300) & 0xFF == ord('q')):
            break


    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()