import numpy as np
import cv2
import copy

# Blues has HSV bound values
lower_blue = np.array([40, 84, 100], dtype="uint8")
upper_blue = np.array([124, 228, 255], dtype="uint8")

# green has HSV bound values
lower_green = np.array([40, 20, 120], dtype="uint8")
upper_green = np.array([105, 100, 200], dtype="uint8")

# White has HSV bound values
lower_white = np.array([00, 0, 200], dtype="uint8")
upper_white = np.array([180, 50, 255], dtype="uint8")


# White for ball is more tolerant
b_lower_white = np.array([00, 10, 170], dtype="uint8")
b_upper_white = np.array([180, 100, 255], dtype="uint8")

border_left = 0
border_right = 0
border_center = 0

class Ball:

    def __init__(self):
        self.currentPosition_x = 0
        self.currentPosition_y = 0
        self.currentPosition_w = 0
        self.currentPosition_h = 0

        self.lastPosition_center_x = 0
        self.lastPosition_center_y = 0
        self.currentPosition_center_x = 0
        self.currentPosition_center_y = 0

        #for now eight directional definition
        #North:1; NE:2; E:3; SE:4; S:5; SW:6; W:7; NW:8
        self.currentDirection = 0
        self.left = False
        self.up = False

    def calculate_direction(self):
        if self.lastPosition_center_x == 0 and self.lastPosition_center_y == 0:
            return
        if self.lastPosition_center_x < self.currentPosition_center_x:
            self.left = False
        else:
            self.left = True
        #don't forget x and y start in the top left corner of the image
        if self.lastPosition_center_y > self.currentPosition_center_y:
            self.up = True
        else:
            self.up = False
        pass

    def updated_position(self, x, y, w, h):
        self.lastPosition_center_x = self.currentPosition_center_x
        self.lastPosition_center_y = self.currentPosition_center_y
        self.currentPosition_center_x = x + int(w / 2)
        self.currentPosition_center_y = y + int(h / 2)
        self.currentPosition_x = x
        self.currentPosition_y = y
        self.currentPosition_w = w
        self.currentPosition_h = h
        self.calculate_direction()
        pass


    def draw_arrow(self, frame):
        point_top_left = (self.currentPosition_x, self.currentPosition_y)
        point_top_right = (self.currentPosition_x + self.currentPosition_w, self.currentPosition_y)
        point_bottom_right = (self.currentPosition_x + self.currentPosition_w, self.currentPosition_y + self.currentPosition_h)
        point_bottom_left = (self.currentPosition_x, self.currentPosition_y + self.currentPosition_h)

        #ball goes left up
        if self.left and self.up:
            cv2.arrowedLine(frame, point_bottom_right, point_top_left,  (0, 255, 0), 3)
            return
        # ball goes left down
        elif self.left and not self.up:
            cv2.arrowedLine(frame, point_top_right, point_bottom_left, (0, 255, 0), 3)
            return
        # ball goes right up
        elif not self.left and self.up:
            cv2.arrowedLine(frame, point_bottom_left, point_top_right, (0, 255, 0), 3)
            return
        # ball goes right down
        elif not self.left and not self.up:
            cv2.arrowedLine(frame, point_top_left, point_bottom_right, (0, 255, 0), 3)
            return






def print_frame(frame, name, show_image=False):
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    cv2.imwrite(name + '.png', frame)
    if show_image:
        cv2.imshow(name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_mask(frame, lower_color, upper_color):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return mask


def get_roi(x, y, w, h, frame, tolerance):
        # increase size for more tolerance
        height, width = frame.shape[:2]
        if tolerance > 1:
            tolerance_x = int((w / width) * tolerance)
            tolerance_y = int((h / height) * tolerance)
            x = x - tolerance_x
            w = w + tolerance_x * 2
            y = y - tolerance_y
            h = h + tolerance_y * 2

        return frame[y:(y + h), x:(x + w)], {"x": x, "y": y, "w": w, "h": h}


def get_biggest_contour(mask):
    (_, contours, _) = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return contours
    c = max(contours, key=cv2.contourArea)
    return c


def corners(frame):
    # conrners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Threshold for an optimal value, it may vary depending on the image.
    return dst


def get_table(frame, blue):

    green_mask = get_mask(frame, lower_green, upper_green)
    print_frame(green_mask, "green_mask")

    blue_mask = get_mask(frame, lower_blue, upper_blue)
    print_frame(blue_mask, "blue_mask")

    if blue:
        kernel = np.ones((20, 20), np.uint8)
        blue_mask_dilated = cv2.dilate(blue_mask, kernel)
        print_frame(blue_mask_dilated, "blue_mask_dilated")
        contours = get_biggest_contour(blue_mask_dilated)
        if len(contours) != 0:
            x, y, w, h = cv2.boundingRect(contours)
        roi, coordinates = get_roi(x, y, w, h, frame.copy(), 25)
    else:
        kernel = np.ones((10, 10), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel)
        print_frame(green_mask_dilated, "green_mask_dilated")
        contours = get_biggest_contour(green_mask_dilated)
        if len(contours) != 0:
            x, y, w, h = cv2.boundingRect(contours)
        roi, coordinates = get_roi(x, y, w, h, frame.copy(), 25)

    print_frame(roi, "roi")

    white_mask = get_mask(roi, lower_white, upper_white)

    print_frame(white_mask, "white_mask")

    cor = corners(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))

    # where gives us two arrays where the [0] array contains the line index
    # and the [1] array contains the column index of the points that fulfill the equation
    coord = np.where(cor > 0.01 * cor.max())

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

    cv2.circle(white_mask, left, 2, (255, 255, 255), 3)
    cv2.circle(white_mask, right, 2, (255, 255, 255), 3)
    cv2.circle(white_mask, bottom, 6, (255, 255, 255), 6)

    #print_frame(white_mask, "corner_mask")

    cv2.line(roi, left, bottom, (0, 255, 0), 6)
    cv2.line(roi, bottom, right, (0, 255, 0), 6)
    cv2.line(roi, left, top, (0, 255, 0), 6)
    cv2.line(roi, top, right, (0, 255, 0), 6)
    print_frame(roi, "table_lines")

    left = (left[0] + coordinates["x"], left[1] + coordinates["y"])
    top = (top[0] + coordinates["x"], top[1] + coordinates["y"])
    right = (right[0] + coordinates["x"], right[1] + coordinates["y"])
    bottom = (bottom[0] + coordinates["x"], bottom[1] + coordinates["y"])
    return left, top, right, bottom
    pass


def narrow_down_roi_for_ball(ball, tleft, tright):
    global border_left
    global border_right
    if ball.lastPosition_center_x == 0 and ball.lastPosition_center_y == 0:
        return {}
        # ball goes left up
    if ball.left and ball.up:
        x = ball.currentPosition_x - 2 * ball.currentPosition_w + int(ball.currentPosition_w/2)
        y = ball.currentPosition_y - 2 * ball.currentPosition_h
        w = ball.currentPosition_w + 3 * ball.currentPosition_w
        h = ball.currentPosition_h + 3 * ball.currentPosition_h

    # ball goes left down
    elif ball.left and not ball.up:
        x = ball.currentPosition_x - 2 * ball.currentPosition_w + int(ball.currentPosition_w/2)
        y = ball.currentPosition_y - int(ball.currentPosition_h/2)
        w = ball.currentPosition_w + 3 * ball.currentPosition_w
        h = ball.currentPosition_h + 2 * ball.currentPosition_h

    # ball goes right up
    elif not ball.left and ball.up:
        x = ball.currentPosition_x - int(ball.currentPosition_w/2)
        y = ball.currentPosition_y - 2 * ball.currentPosition_h
        w = ball.currentPosition_w + 2 * ball.currentPosition_w
        h = ball.currentPosition_h + 3 * ball.currentPosition_h

    # ball goes right down
    elif not ball.left and not ball.up:
        x = ball.currentPosition_x - int(ball.currentPosition_w/2)
        y = ball.currentPosition_y - int(ball.currentPosition_h/2)
        w = ball.currentPosition_w + 2 * ball.currentPosition_w
        h = ball.currentPosition_h + 2 * ball.currentPosition_h

    if ball.currentPosition_center_x < border_center and border_left != tleft and x < border_left:
        temp = border_left - x
        x = x + temp
    if ball.currentPosition_center_x > border_center and border_right != tright and x+w > border_right:
        temp = (x+w) - border_right
        x = x - temp


    return {"x": x, "y": y, "w": w, "h": h}



def get_ball_position(ball, current_frame, roi_coordinates, fgbg, tleft, tright):
    global border_left
    global border_right
    global border_center

    #add current frame for background subtraction
    background_gmask = fgbg.apply(current_frame.copy())


    test = False
    # narrow background to the roi and dilate the result
    coordinates = narrow_down_roi_for_ball(ball, tleft, tright)
    if len(coordinates) != 0:
        #we know the position and direction of the ball so we can update te roi
        roi_coordinates = coordinates
        test = True


    roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"], background_gmask.copy(), 1)
    roi_current_frame = cv2.erode(roi_current_frame, np.ones((3, 3), np.uint8))
    d_background_gmask = cv2.dilate(roi_current_frame, np.ones((7, 7), np.uint8))

    #substract back and foreground
    roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"], current_frame.copy(), 1)
    fg = cv2.bitwise_and(roi_current_frame, roi_current_frame, mask=d_background_gmask)

    #now filter out our white ball
    white_mask = get_mask(fg, b_lower_white, b_upper_white)
    white_mask = cv2.dilate(white_mask, np.ones((15, 15), np.uint8))

    #cv2.imshow("temp", fg)
    #cv2.imshow("w", white_mask)
    #cv2.waitKey(0)


    (_, contours, _) = cv2.findContours(white_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    max_contour = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(max_contour)
    if test:
        for c in contours:
            tx, ty, tw, th = cv2.boundingRect(c)
            right = roi_coordinates["x"] + tx + tw
            left = roi_coordinates["x"] + tx
            if tx == x1 and ty == y1 and tx + tw == x1+w1 and ty + th == y1 + h1:
                continue

            #ball goes right but contour is left of it
            if not ball.left and right < ball.currentPosition_center_x:
                continue

            #ball goes left but contour is right of it
            if ball.left and left > ball.currentPosition_center_x:
                continue

            #ball goes left and there is something moving in front of it
            if ball.left and ball.currentPosition_center_x < border_center and \
                            right < ball.currentPosition_center_x and right > border_left:
                border_left = roi_coordinates["x"] + tx+tw
                border_right = tright


            # ball goes right and there is something moving in front of it
            if not ball.left and ball.currentPosition_center_x > border_center and\
                            left > ball.currentPosition_center_x  and left < border_right:
                border_right = roi_coordinates["x"] + tx
                border_left = tleft

    #draw a border around tha ball
    top_left = (roi_coordinates["x"] + x1, roi_coordinates["y"] + y1)
    top_right = (roi_coordinates["x"] + x1 + w1, roi_coordinates["y"] + y1)
    bottom_right = (roi_coordinates["x"] + x1 + w1, roi_coordinates["y"] + y1 + h1)
    bottom_left = (roi_coordinates["x"] + x1, roi_coordinates["y"] + y1 + h1)
    result = draw_border(top_left, top_right, bottom_right, bottom_left, current_frame.copy(), (255, 255, 0))


    #height, width = result.shape[:2]
    #result = cv2.resize(result, (int(width / 2), int(height / 2)))
    #cv2.imshow("temp", fg)
    #cv2.imshow("w", white_mask)
    #cv2.waitKey(0)


    ball.updated_position(roi_coordinates["x"] + x1, roi_coordinates["y"] + y1, w1, h1)
    return result


def draw_border(top_left, top_right, bottom_right, bottom_left, frame, color):
    cv2.line(frame, top_left, bottom_left, color, 6)
    cv2.line(frame, bottom_left, bottom_right, color, 6)
    cv2.line(frame, top_left, top_right, color, 6)
    cv2.line(frame, top_right, bottom_right, color, 6)
    return frame


def main(file, video):
    global border_left
    global border_right
    global border_center
    if video:
        camera = cv2.VideoCapture(file)

        (grabbed, first_frame) = camera.read()
        print_frame( first_frame, "input_first_frame")

        tleft, ttop, tright, tbottom = get_table(first_frame, False)
        x = tleft[0]
        y = 0
        w = tright[0]-tleft[0]
        h = tbottom[1]
        roi_first_frame, coordinates = get_roi(x, y, w, h, first_frame.copy(), 1)
        #print_frame(roi_first_frame, "roi_first_frame", True)
        border_left = tleft[0]
        border_right = tright[0]
        border_center = tleft[0] + int((tright[0]-tleft[0])/2)

        last_frame = first_frame.copy()

        fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=70, detectShadows=False)
        fgbg.apply(last_frame)

        ball = Ball()

        while camera.isOpened():
            (grabbed, current_frame) = camera.read()
            if not grabbed:
                break

            result = get_ball_position(ball, current_frame.copy(),
                                       {"x": tbottom[0], "y": y, "w": ttop[0] - tbottom[0], "h": h}, fgbg, tleft[0], tright[0])
            if len(result) == 0:
                continue

            result = draw_border(tleft, ttop, tright, tbottom, result.copy(), (0, 255, 0))
            if len(result) == 0:
                continue

            ball.draw_arrow(result)

            height, width = result.shape[:2]
            result = cv2.resize(result, (int(width / 2), int(height / 2)))
            cv2.imshow("run", result)
            cv2.waitKey(1)

            last_frame = current_frame.copy()

    else:
        frame = cv2.imread(file)
        print_frame(frame, "input_image")

    pass


if __name__ == "__main__":
    main("gtt4.mp4", True)
    pass

