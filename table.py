import numpy as np
import cv2
import table
from helper import print_frame
from helper import get_mask
from helper import get_biggest_contour
from helper import get_roi
from helper import corners

class Table:
    def __init__(self):
        self.border_left = 0
        self.border_right = 0
        self.center = 0
        self.center_left_side = (0, 0)
        self.center_right_side = (0, 0)

        # Blues has HSV bound values
        self.lower_blue = np.array([40, 84, 100], dtype="uint8")
        self.upper_blue = np.array([124, 228, 255], dtype="uint8")

        # green has HSV bound values
        self.lower_green = np.array([40, 60, 120], dtype="uint8")
        self.upper_green = np.array([115, 100, 200], dtype="uint8")

        # White has HSV bound values
        self.lower_white = np.array([00, 0, 200], dtype="uint8")
        self.upper_white = np.array([180, 50, 255], dtype="uint8")

    def get_table(self, frame, blue, pos_straight):

        green_mask = get_mask(frame, self.lower_green, self.upper_green)
        print_frame(green_mask, "green_mask")

        blue_mask = get_mask(frame, self.lower_blue, self.upper_blue)
        print_frame(blue_mask, "blue_mask")

        if blue:
            kernel = np.ones((20, 20), np.uint8)
            blue_mask_dilated = cv2.dilate(blue_mask, kernel)
            # cv2.imshow("b", blue_mask_dilated)
            # cv2.waitKey(0)
            print_frame(blue_mask_dilated, "blue_mask_dilated")
            contours = get_biggest_contour(blue_mask_dilated)
            if len(contours) != 0:
                x, y, w, h = cv2.boundingRect(contours)
            roi, coordinates = get_roi(x, y, w, h, frame.copy(), 25)
        else:
            kernel = np.ones((10, 10), np.uint8)
            green_mask_dilated = cv2.dilate(green_mask, kernel)
            print_frame(green_mask_dilated, "green_mask_dilated")
            # cv2.imshow("g", green_mask_dilated)
            # cv2.waitKey(0)
            contours = get_biggest_contour(green_mask_dilated)
            if len(contours) != 0:
                x, y, w, h = cv2.boundingRect(contours)
            roi, coordinates = get_roi(x, y, w, h, frame.copy(), 25)

        print_frame(roi, "roi")

        white_mask = get_mask(roi, self.lower_white, self.upper_white)

        print_frame(white_mask, "white_mask")
        cor = corners(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))

        # where gives us two arrays where the [0] array contains the line index
        # and the [1] array contains the column index of the points that fulfill the equation
        coord = np.where(cor > 0.01 * cor.max())
        if pos_straight:
            left = (coord[1][np.argmin(coord[1])], coord[0][np.argmin(coord[1])])
            right = (coord[1][np.argmax(coord[1])], coord[0][np.argmax(coord[1])])
            tleft = (0, 0)
            tright = (0, 0)
            diff = int((right[0] - left[0]) / 5)
            # check x values
            for idx in range(len(coord[1])):
                if coord[1][idx] > left[0] and coord[1][idx] < left[0] + diff and (
                        coord[0][idx] < tleft[1] or tleft[1] == 0):
                    tleft = (coord[1][idx], coord[0][idx])
                if coord[1][idx] < right[0] and coord[1][idx] > right[0] - diff and (
                        coord[0][idx] < tright[1] or tright[1] == 0):
                    tright = (coord[1][idx], coord[0][idx])
            top = tright
            bottom = left
            left = tleft


        else:
            left = (coord[1][np.argmin(coord[1])], coord[0][np.argmin(coord[1])])
            right = (coord[1][np.argmax(coord[1])], coord[0][np.argmax(coord[1])])
            bottom = (coord[1][np.argmax(coord[0])], coord[0][np.argmax(coord[0])])

            if abs(bottom[0] - left[0]) < abs(bottom[0] - right[0]):
                length = int(white_mask.shape[1] - white_mask.shape[1] / 3)
                coord[1][coord[1] < length] = white_mask.shape[1]
                coord[0][coord[1] == white_mask.shape[1]] = white_mask.shape[0]
                top = (coord[1][np.argmin(coord[0][:])], coord[0][np.argmin(coord[0])])
            else:
                length = int(white_mask.shape[1] / 3)
                coord[1][coord[1] > length] = white_mask.shape[1]
                coord[0][coord[1] == white_mask.shape[1]] = white_mask.shape[0]
                top = (coord[1][np.argmin(coord[0][:length])], coord[0][np.argmin(coord[0])])

        # cv2.circle(white_mask, left, 2, (255, 255, 255), 3)
        # cv2.circle(white_mask, right, 2, (255, 255, 255), 3)
        # cv2.circle(white_mask, bottom, 6, (255, 255, 255), 6)

        # print_frame(white_mask, "corner_mask")

        # cv2.line(roi, left, bottom, (0, 255, 0), 6)
        # cv2.line(roi, bottom, right, (0, 255, 0), 6)
        # cv2.line(roi, left, top, (0, 255, 0), 6)
        # cv2.line(roi, top, right, (0, 255, 0), 6)
        # print_frame(roi, "table_lines")

        left = (left[0] + coordinates["x"], left[1] + coordinates["y"])
        top = (top[0] + coordinates["x"], top[1] + coordinates["y"])
        right = (right[0] + coordinates["x"], right[1] + coordinates["y"])
        bottom = (bottom[0] + coordinates["x"], bottom[1] + coordinates["y"])
        return left, top, right, bottom


    def define_borders(self, tleft, ttop, tright, tbottom):
        self.border_left = tleft[0]
        self.border_right = tright[0]
        self.border_center = tleft[0] + int((tright[0]-tleft[0])/2)
        self.center_left_side = (tleft[0] + int((self.border_center-tleft[0])/2), tleft[1] + int((tbottom[1]-tleft[1])/2))
        self.center_right_side = (self.border_center + int((tright[0] - self.border_center) / 2), ttop[1] + int((tright[1] - ttop[1])/2))
        pass