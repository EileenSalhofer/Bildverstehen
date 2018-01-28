import numpy as np
import cv2


# Blues has HSV bound values
lower_blue = np.array([40, 84, 100], dtype="uint8")
upper_blue = np.array([124, 228, 255], dtype="uint8")

# green has HSV bound values
lower_green = np.array([40, 60, 120], dtype="uint8")
upper_green = np.array([115, 100, 200], dtype="uint8")

# White has HSV bound values
lower_white = np.array([00, 0, 200], dtype="uint8")
upper_white = np.array([180, 50, 255], dtype="uint8")


# White for ball is more tolerant
b_lower_white = np.array([00, 10, 170], dtype="uint8")
b_upper_white = np.array([180, 100, 255], dtype="uint8")

border_left = 0
border_right = 0
border_center = 0
center_left_side = (0, 0)
center_right_side = (0, 0)


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

        self.lastBox = []
        self.box = []

        self.left = False
        self.up = False

        self.lastArrowHead = (0, 0)
        self.lastArrowTail = (0, 0)
        self.arrowhead = (0, 0)
        self.arrowtail = (0, 0)

        self.oldguessedpath = np.array([])
        self.lastLeft = False
        self.lastUp = False

        self.leftRightChange = False
        self.upDownChange = False

        self.frameCounter = 0

    def calculate_direction(self):
        global border_left
        global border_right
        if self.lastPosition_center_x == 0 and self.lastPosition_center_y == 0:
            return

        if self.left and (border_left + int(self.currentPosition_w)/3) > self.currentPosition_x:
            self.lastLeft = self.left
            self.left = False
            return

        if not self.left and (border_right - int(self.currentPosition_w)/3) < self.currentPosition_x:
            self.lastLeft = self.left
            self.left = True
            return

        if self.lastPosition_center_x < self.currentPosition_center_x:
            self.lastLeft = self.left
            self.left = False
        else:
            self.lastLeft = self.left
            self.left = True
        # don't forget x and y start in the top left corner of the image
        if self.lastPosition_center_y > self.currentPosition_center_y:
            self.lastUp = self.up
            self.up = True
        else:
            self.lastUp = self.up
            self.up = False
        pass

    def updated_position(self, box):
        self.box = box
        max_x = max([box[0][0], box[1][0], box[2][0], box[3][0]])
        min_x = min([box[0][0], box[1][0], box[2][0], box[3][0]])
        max_y = max([box[0][1], box[1][1], box[2][1], box[3][1]])
        min_y = min([box[0][1], box[1][1], box[2][1], box[3][1]])

        self.lastPosition_center_x = self.currentPosition_center_x
        self.lastPosition_center_y = self.currentPosition_center_y
        self.currentPosition_center_x = int((max_x+min_x)/2)
        self.currentPosition_center_y = int((max_y+min_y)/2)
        self.currentPosition_x = min_x
        self.currentPosition_y = min_y
        self.currentPosition_w = (max_x-min_x)
        self.currentPosition_h = (max_y-min_y)
        self.calculate_direction()
        pass

    def draw_arrow(self, frame):
        box = self.box

        center_line_a =(  (box[1][0] + int((box[0][0] - box[1][0])/2)) if box[0][0] > box[1][0] else (box[0][0] + int((box[1][0] - box[0][0])/2)), (box[1][1] + int((box[0][1] - box[1][1])/2)) if box[0][1] > box[1][1] else (box[0][1] + int((box[1][1] - box[0][1])/2)))
        center_line_b =( (box[2][0] + int((box[1][0] - box[2][0])/2)) if box[1][0] > box[2][0] else (box[1][0] + int((box[2][0] - box[1][0])/2)), (box[2][1] + int((box[1][1] - box[2][1])/2)) if box[1][1] > box[2][1] else (box[1][1] + int((box[2][1] - box[1][1])/2)))
        center_line_c =( (box[3][0] + int((box[2][0] - box[3][0])/2)) if box[2][0] > box[3][0] else (box[2][0] + int((box[3][0] - box[2][0])/2)), (box[3][1] + int((box[2][1] - box[3][1])/2)) if box[2][1] > box[3][1] else (box[2][1] + int((box[3][1] - box[2][1])/2)))
        center_line_d =( (box[0][0] + int((box[3][0] - box[0][0])/2)) if box[3][0] > box[0][0] else (box[3][0] + int((box[0][0] - box[3][0])/2)), (box[0][1] + int((box[3][1] - box[0][1])/2)) if box[3][1] > box[0][1] else (box[3][1] + int((box[0][1] - box[3][1])/2)))

        left = center_line_a if center_line_a[0] < center_line_b[0] else center_line_b
        left = left if left[0] < center_line_c[0] else center_line_c
        left = left if left[0] < center_line_d[0] else center_line_d

        right = center_line_a if center_line_a[0] > center_line_b[0] else center_line_b
        right = right if right[0] > center_line_c[0] else center_line_c
        right = right if right[0] > center_line_d[0] else center_line_d

        if self.left and left and right:
            self.lastArrowHead = self.arrowhead
            self.lastArrowTail = self.arrowtail
            self.arrowhead = left
            self.arrowtail = right
            cv2.arrowedLine(frame, right, left, (0, 255, 0), 3)
            return
        elif left and right:
            self.lastArrowHead = self.arrowhead
            self.lastArrowTail = self.arrowtail
            self.arrowhead = right
            self.arrowtail = left
            cv2.arrowedLine(frame, left, right, (0, 255, 0), 3)
            return

    def position_change(self):
        if self.box[0][0] == self.lastBox[0][0] and self.box[0][1] == self.lastBox[0][1] and \
            self.box[1][0] == self.lastBox[1][0] and self.box[1][1] == self.lastBox[1][1] and \
            self.box[2][0] == self.lastBox[2][0] and self.box[2][1] == self.lastBox[2][1] and \
            self.box[3][0] == self.lastBox[3][0] and self.box[3][1] == self.lastBox[3][1]:
            self.box = []
            self.lastBox = []
            return False

        return True

    def draw_parable(self, frame):
        global center_left_side
        global center_right_side
        global border_left
        global border_right

        if not (self.up == True and self.lastUp == False) and self.frameCounter == 0 and self.lastLeft == self.left:
            self.upDownChange = False
            self.leftRightChange = False
            if self.lastLeft == self.left:
                cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                return self.oldguessedpath

            if self.left and (self.arrowhead[0]-2*abs(self.arrowtail[0] - self.arrowhead[0])) <= center_left_side[0] or not self.left and (self.arrowhead[0]+2*abs(self.arrowhead[0] - self.arrowtail[0])) >= center_right_side[0]:
                return np.array([])

        elif self.frameCounter == 0 or (self.up == True and self.lastUp == False) or not (self.lastLeft == self.left):
            self.frameCounter += 1
            self.upDownChange = False
            self.leftRightChange = False
            if not (self.lastLeft == self.left):
                self.leftRightChange = True
            else:
                self.upDownChange = True
            return np.array([])
        elif self.upDownChange == True and self.up == True and self.lastUp == True:
            self.upDownChange = False
            self.leftRightChange = False
            self.frameCounter = 0

            # ball goes left up
            if self.left and self.up:
                p1 = (self.arrowhead[0] - int(5*abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(2*abs(self.arrowtail[1] - self.arrowhead[1])))
                # p2 = (center_left_side[0] + int(abs(self.arrowhead[0]-center_left_side[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(2*abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1] - self.arrowhead[1])))
                p3 = self.arrowhead



                pts = np.array(parable_get_points_between(p1, p2, p3))

                pts[pts[0:,0] < border_left, 0] = border_left + (border_left-pts[pts[0:,0] < border_left, 0])

                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)

                # cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts
                # ball goes right up

            elif not self.left and self.up:
                p1 = self.arrowhead
                # p2 = (center_right_side[0] - int(abs(center_right_side[0]-self.arrowhead[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] + int(2*abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1] - self.arrowhead[1])))
                p3 = (self.arrowhead[0] + int(5*abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(2*abs(self.arrowtail[1] - self.arrowhead[1])))

                pts = np.array(parable_get_points_between(p1, p2, p3))

                pts[pts[0:,0] > border_right, 0] = border_right - (pts[pts[0:,0] > border_right, 0]-border_right)

                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)

                # cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts
            return np.array([])
        elif self.leftRightChange == True and self.lastLeft == self.left:
            self.frameCounter = 0
            self.upDownChange = False
            self.leftRightChange = False

            # ball goes left up
            if self.left and self.up:
                p1 = center_left_side
                #p2 = (center_left_side[0] + int(abs(self.arrowhead[0]-center_left_side[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.arrowhead

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts

            # ball goes left down
            elif self.left and not self.up:
                p1 = center_left_side
                #p2 = (center_left_side[0] + int(abs(self.arrowhead[0]-center_left_side[0])/2), self.arrowhead[1] + int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] + int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.arrowhead

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts


            # ball goes right up
            elif not self.left and self.up:
                p1 = self.arrowhead
                #p2 = (center_right_side[0] - int(abs(center_right_side[0]-self.arrowhead[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0]  + int(abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = center_right_side

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts

            # ball goes right down
            elif not self.left and not self.up:
                p1 = self.arrowhead
                p2 = (self.arrowhead[0] + int(abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] + int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                #p2 = (self.arrowhead[0] + int(3.2*abs(self.arrowhead[0]-self.arrowtail[0])), self.arrowhead[1] + int(1.5*abs(self.arrowtail[1]-self.arrowhead[1])))
                p3 = center_right_side

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255 , 255), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldguessedpath = pts
        else:
            self.frameCounter = 1
            #self.upDownChange = False
            #self.leftRightChange = False
        pass


def parable_get_points_between(ptr1, ptr2, ptr3):
    A1 = -(ptr1[0]**2)+(ptr2[0]**2)
    B1 = -(ptr1[0]) + (ptr2[0])
    D1 = -(ptr1[1]) + (ptr2[1])

    A2 = -(ptr2[0] ** 2) + (ptr3[0] ** 2)
    B2 = -(ptr2[0]) + (ptr3[0])
    D2 = -(ptr2[1]) + (ptr3[1])

    Bm = -(B2/B1)

    A3 = Bm * A1 + A2
    D3 = Bm * D1 + D2

    a = D3/A3
    b = (D1 - (A1 * a)) / B1
    c = ptr1[1] - (a * (ptr1[0]**2)) - (b * ptr1[0])

    points = []
    points.append(ptr1)

    ptr2_in_array = False
    # Calculate points values
    for x in range(abs(ptr1[0]-ptr3[0])):
        x = x+ptr1[0]
        if x > ptr2[0] and not ptr2_in_array:
            points.append(ptr2)
            ptr2_in_array = True
            continue
        y = abs(int((a * (x ** 2)) + (b * x) + c))
        points.append([x, y])
    return points


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


def get_table(frame, blue, pos_straight):

    green_mask = get_mask(frame, lower_green, upper_green)
    print_frame(green_mask, "green_mask")

    blue_mask = get_mask(frame, lower_blue, upper_blue)
    print_frame(blue_mask, "blue_mask")

    if blue:
        kernel = np.ones((20, 20), np.uint8)
        blue_mask_dilated = cv2.dilate(blue_mask, kernel)
        #cv2.imshow("b", blue_mask_dilated)
        #cv2.waitKey(0)
        print_frame(blue_mask_dilated, "blue_mask_dilated")
        contours = get_biggest_contour(blue_mask_dilated)
        if len(contours) != 0:
            x, y, w, h = cv2.boundingRect(contours)
        roi, coordinates = get_roi(x, y, w, h, frame.copy(), 25)
    else:
        kernel = np.ones((10, 10), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel)
        print_frame(green_mask_dilated, "green_mask_dilated")
        #cv2.imshow("g", green_mask_dilated)
        #cv2.waitKey(0)
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
    if pos_straight:
        left = (coord[1][np.argmin(coord[1])], coord[0][np.argmin(coord[1])])
        right = (coord[1][np.argmax(coord[1])], coord[0][np.argmax(coord[1])])
        tleft = (0,0)
        tright = (0, 0)
        diff = int((right[0] - left[0])/5)
        #check x values
        for idx in range(len(coord[1])):
            if coord[1][idx] > left[0] and coord[1][idx] < left[0]+diff and (coord[0][idx] < tleft[1] or tleft[1] == 0):
                tleft = (coord[1][idx], coord[0][idx])
            if coord[1][idx] < right[0] and coord[1][idx] > right[0]-diff and (coord[0][idx] < tright[1] or tright[1] == 0):
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

    #cv2.circle(white_mask, left, 2, (255, 255, 255), 3)
    #cv2.circle(white_mask, right, 2, (255, 255, 255), 3)
    #cv2.circle(white_mask, bottom, 6, (255, 255, 255), 6)

    #print_frame(white_mask, "corner_mask")

    #cv2.line(roi, left, bottom, (0, 255, 0), 6)
    #cv2.line(roi, bottom, right, (0, 255, 0), 6)
    #cv2.line(roi, left, top, (0, 255, 0), 6)
    #cv2.line(roi, top, right, (0, 255, 0), 6)
    #print_frame(roi, "table_lines")

    left = (left[0] + coordinates["x"], left[1] + coordinates["y"])
    top = (top[0] + coordinates["x"], top[1] + coordinates["y"])
    right = (right[0] + coordinates["x"], right[1] + coordinates["y"])
    bottom = (bottom[0] + coordinates["x"], bottom[1] + coordinates["y"])
    return left, top, right, bottom
    pass


def narrow_down_roi_for_ball(ball, tleft, tright):
    global border_left
    global border_right
    if ball.lastPosition_center_x == 0 and ball.lastPosition_center_y == 0 or \
                            ball.lastPosition_center_x == ball.currentPosition_center_x \
                    and ball.lastPosition_center_y == ball.currentPosition_center_y:
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
    temp = [1]
    if len(coordinates) != 0 and coordinates["x"] + coordinates["w"] < tright and coordinates["x"] > tleft:
        #we know the position and direction of the ball so we can update te roi
        roi_coordinates = coordinates
        temp = [roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"]]
        test = True

    if len(ball.box) != 0 and len(ball.lastBox) != 0 and not ball.position_change() or min(temp) < 0:
        ball.updated_position(ball.box)
        return []

    roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"], background_gmask.copy(), 1)
    roi_current_frame = cv2.erode(roi_current_frame, np.ones((3, 3), np.uint8))
    d_background_gmask = cv2.dilate(roi_current_frame, np.ones((7, 7), np.uint8))

    #substract back and foreground
    roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"], current_frame.copy(), 1)
    fg = cv2.bitwise_and(roi_current_frame, roi_current_frame, mask=d_background_gmask)

    #now filter out our white ball
    white_mask = get_mask(fg, b_lower_white, b_upper_white)
    white_mask = cv2.dilate(white_mask, np.ones((15, 15), np.uint8))
    #white_mask = cv2.erode(white_mask, np.ones((5, 5), np.uint8))

    #cv2.imshow("temp", fg)
    #cv2.imshow("w", white_mask)
    #cv2.waitKey(0)


    (_, contours, _) = cv2.findContours(white_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        ball.updated_position(ball.box)
        return []
    max_contour = max(contours, key=cv2.contourArea)


    x1, y1, w1, h1 = cv2.boundingRect(max_contour)
    box = cv2.minAreaRect(max_contour)
    box = ((box[0][0] + roi_coordinates["x"], box[0][1] + roi_coordinates["y"]), box[1], box[2])
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    ball.updated_position(box)


    if test:
        for c in contours:
            if cv2.contourArea(c) < 100:
                continue

            tx, ty, tw, th = cv2.boundingRect(c)
            right = roi_coordinates["x"] + tx + tw
            left = roi_coordinates["x"] + tx

            if tx == x1 and ty == y1 and tx + tw == x1+w1 and ty + th == y1 + h1:
                continue

            #ball goes left and there is something moving in front of it
            if ball.left and ball.currentPosition_center_x < border_center and \
                            right < ball.currentPosition_center_x                 :
                border_left = roi_coordinates["x"] + tx+tw
                border_right = tright

            # ball goes right and there is something moving in front of it
            if not ball.left and ball.currentPosition_center_x > border_center and\
                            left > ball.currentPosition_center_x  and left < border_right:
                border_right = roi_coordinates["x"] + tx
                border_left = tleft


    result = current_frame.copy()
    cv2.drawContours(result, [box.astype("int")], -1, (250, 150, 0), 5)

    return result


def draw_border(top_left, top_right, bottom_right, bottom_left, frame, color):
    cv2.line(frame, top_left, bottom_left, color, 6)
    cv2.line(frame, bottom_left, bottom_right, color, 6)
    cv2.line(frame, top_left, top_right, color, 6)
    cv2.line(frame, top_right, bottom_right, color, 6)
    return frame


def main(file, video, blue_table, pos_straight):
    global border_left
    global border_right
    global border_center
    global center_left_side
    global center_right_side
    if video:
        camera = cv2.VideoCapture(file)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))

        (grabbed, first_frame) = camera.read()
        print_frame( first_frame, "input_first_frame")

        tleft, ttop, tright, tbottom = get_table(first_frame, blue_table, pos_straight)
        x = tleft[0]
        y = 0
        w = tright[0]-tleft[0]
        h = tbottom[1]
        roi_first_frame, coordinates = get_roi(x, y, w, h, first_frame.copy(), 1)
        #print_frame(roi_first_frame, "roi_first_frame", True)
        border_left = tleft[0]
        border_right = tright[0]
        border_center = tleft[0] + int((tright[0]-tleft[0])/2)
        center_left_side = (tleft[0] + int((border_center-tleft[0])/2), tleft[1] + int((tbottom[1]-tleft[1])/2))
        center_right_side = (border_center + int((tright[0] - border_center) / 2), ttop[1] + int((tright[1] - ttop[1])/2))

        last_frame = first_frame.copy()

        fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=70, detectShadows=False)
        fgbg.apply(last_frame)

        result = draw_border(tleft, ttop, tright, tbottom, last_frame.copy(), (0, 255, 0))
        #cv2.imshow("run", result)
        #cv2.waitKey(0)

        ball = Ball()

        while camera.isOpened():
            (grabbed, current_frame) = camera.read()
            if not grabbed:
                break

            result = get_ball_position(ball, current_frame.copy(),
                                       {"x": tbottom[0], "y": y, "w": ttop[0] - tbottom[0], "h": h}, fgbg, tleft[0], tright[0])
            if len(result) == 0:
                result = current_frame.copy()

            result = draw_border(tleft, ttop, tright, tbottom, result.copy(), (0, 255, 0))
            if len(result) == 0:
                continue

            ball.draw_arrow(result)
            ball.draw_parable(result)

            out.write(result)
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
    #  main("ggg.mp4", True, False, True)
    main("gtt4.mp4", True, False, False)
    pass

