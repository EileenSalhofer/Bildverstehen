import numpy as np
import cv2
import table
from helper import parable_get_points_between
from helper import get_roi
from helper import get_mask

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

        self.oldup = np.array([])
        self.oldleftright = np.array([])
        self.lastLeft = False
        self.lastUp = False

        self.leftRightChange = False
        self.upDownChange = False

        self.frameCounter = 0

        self.table = None

        # White for ball is more tolerant
        self.b_lower_white = np.array([00, 10, 170], dtype="uint8")
        self.b_upper_white = np.array([180, 100, 255], dtype="uint8")

    def calculate_direction(self):
        """This Function sets the Flags for if the ball is going left or up according the last directions.

        Args:
            self: This class.

        Returns:
            None.

        """
        border_left = self.table.border_left
        border_right = self.table.border_right
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
        """This Function updates current x and y position of the ball. Calls class function calculate_direction.

        Args:
            self: This class.
            box: A box around the current position of the ball.

        Returns:
            None.

        """
        self.box = box
        if len(self.box) == 0:
            return
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
        """This Function draws an arrow according to the current direction of the ball along the ball.

        Args:
            self: This class.
            frame: Image or Frame to draw the arrow in.

        Returns:
            None.

        """
        box = self.box
        if len(box) == 0:
            return
        center_line_a =(  (box[1][0] + int((box[0][0] - box[1][0])/2)) if box[0][0] > box[1][0] else
                          (box[0][0] + int((box[1][0] - box[0][0])/2)),
                          (box[1][1] + int((box[0][1] - box[1][1])/2)) if box[0][1] > box[1][1] else
                          (box[0][1] + int((box[1][1] - box[0][1])/2)))
        center_line_b =( (box[2][0] + int((box[1][0] - box[2][0])/2)) if box[1][0] > box[2][0] else
                         (box[1][0] + int((box[2][0] - box[1][0])/2)),
                         (box[2][1] + int((box[1][1] - box[2][1])/2)) if box[1][1] > box[2][1] else
                         (box[1][1] + int((box[2][1] - box[1][1])/2)))
        center_line_c =( (box[3][0] + int((box[2][0] - box[3][0])/2)) if box[2][0] > box[3][0] else
                         (box[2][0] + int((box[3][0] - box[2][0])/2)),
                         (box[3][1] + int((box[2][1] - box[3][1])/2)) if box[2][1] > box[3][1] else
                         (box[2][1] + int((box[3][1] - box[2][1])/2)))
        center_line_d =( (box[0][0] + int((box[3][0] - box[0][0])/2)) if box[3][0] > box[0][0] else
                         (box[3][0] + int((box[0][0] - box[3][0])/2)),
                         (box[0][1] + int((box[3][1] - box[0][1])/2)) if box[3][1] > box[0][1] else
                         (box[3][1] + int((box[0][1] - box[3][1])/2)))

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
        """This Function is to check if the ball moved since current and the last frame.
            If not this could indicate an error.

        Args:
            self: This class.

        Returns:
            Returns True if the postion of the ball changed. False if the ball stayed in the same position.

        """
        if self.box[0][0] == self.lastBox[0][0] and self.box[0][1] == self.lastBox[0][1] and \
            self.box[1][0] == self.lastBox[1][0] and self.box[1][1] == self.lastBox[1][1] and \
            self.box[2][0] == self.lastBox[2][0] and self.box[2][1] == self.lastBox[2][1] and \
            self.box[3][0] == self.lastBox[3][0] and self.box[3][1] == self.lastBox[3][1]:
            self.box = []
            self.lastBox = []
            return False

        return True

    def draw_parable(self, frame):
        """This Function draws a parable into the current image.
            Condition is that the ball moved in the last two frames in the same direction and
            that there was a direction change in the last two frames. Calls parable_get_points_between from help.py

        Args:
            self: This class.
            frame: Image or Frame to draw the parable in.

        Returns:
            None.

        """
        border_left = self.table.border_left
        border_right = self.table.border_right
        if not (self.up == True and self.lastUp == False) and self.frameCounter == 0 and self.lastLeft == self.left:
            self.upDownChange = False
            self.leftRightChange = False
            if self.lastLeft == self.left:
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0), 6)
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)
                return

            if self.left and (self.arrowhead[0]-2*abs(self.arrowtail[0] - self.arrowhead[0])) <= self.table.center_left_side[0] or\
                            not self.left and (self.arrowhead[0]+2*abs(self.arrowhead[0] - self.arrowtail[0])) >= self.table.center_right_side[0]:
                return

        elif self.frameCounter == 0 or (self.up == True and self.lastUp == False) or not (self.lastLeft == self.left):
            self.frameCounter += 1
            self.upDownChange = False
            self.leftRightChange = False
            if not (self.lastLeft == self.left):
                self.leftRightChange = True
            else:
                self.upDownChange = True
            cv2.polylines(frame, [self.oldup], False, (255, 0, 0), 6)
            cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)
            return
        elif self.upDownChange == True and self.up == True and self.lastUp == True:
            self.upDownChange = False
            self.leftRightChange = False
            self.frameCounter = 0

            # ball goes left up
            if self.left and self.up:
                p1 = (self.arrowhead[0] - int(5*abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(2*abs(self.arrowtail[1] - self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(2*abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1] - self.arrowhead[1])))
                p3 = self.arrowhead



                pts = np.array(parable_get_points_between(p1, p2, p3))

                pts[pts[0:,0] < border_left, 0] = border_left + (border_left-pts[pts[0:,0] < border_left, 0])

                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)

                self.oldup = pts
            # ball goes right up
            elif not self.left and self.up:
                p1 = self.arrowhead
                p2 = (self.arrowhead[0] + int(2*abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1] - self.arrowhead[1])))
                p3 = (self.arrowhead[0] + int(5*abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(2*abs(self.arrowtail[1] - self.arrowhead[1])))

                pts = np.array(parable_get_points_between(p1, p2, p3))

                pts[pts[0:,0] > border_right, 0] = border_right - (pts[pts[0:,0] > border_right, 0]-border_right)

                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)

                self.oldup = pts
            return
        elif self.leftRightChange == True and self.lastLeft == self.left:
            self.frameCounter = 0
            self.upDownChange = False
            self.leftRightChange = False

            # ball goes left up
            if self.left and self.up:
                p1 = self.table.center_left_side
                p2 = (self.arrowhead[0] - int(abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.arrowhead

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0), 6)

                self.oldleftright = pts


            # ball goes right up
            elif not self.left and self.up:
                p1 = self.arrowhead
                p2 = (self.arrowhead[0]  + int(abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.table.center_right_side

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0 ), 6)

                self.oldleftright = pts

        else:
            self.frameCounter = 1
        pass

    def get_ball_position(self, ball, current_frame, roi_coordinates, fgbg, tleft, tright):
        """This function tries the find the ball in the current frame.

        Args:
            self: This class.
            ball: Ball Class containing all important information of the ball.
            current_frame: The current frame to find the ball in.
            roi_coordinates: current roi.
            fgbg: OpenCV forground background subtraction to filter out moving objects.
            tleft: Most left point for the roi.
            tright: Most right point for the roi.

        Returns:
            Image or Frame with a box drawn around the ball. If ball is not found an empty array is returned.

        """
        border_left = ball.table.border_left
        border_right = ball.table.border_right
        border_center = ball.table.border_center

        # add current frame for background subtraction
        background_gmask = fgbg.apply(current_frame.copy())

        found_contours = False
        # narrow background to the roi and dilate the result
        coordinates = self.narrow_down_roi_for_ball(ball, tleft, tright)

        if len(coordinates) != 0 and coordinates["x"] + coordinates["w"] < tright and coordinates["x"] > tleft:
            # we know the position and direction of the ball so we can update te roi
            roi_coordinates = coordinates
            check_negative_val = [roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"],
                                  roi_coordinates["h"]]
            found_contours = True

            if len(ball.box) != 0 and len(ball.lastBox) != 0 and not ball.position_change() or min(
                    check_negative_val) < 0:
                ball.updated_position(ball.box)
                return []

        # cv2.imshow("b", background_gmask)
        # cv2.waitKey(0)
        roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"],
                                                 roi_coordinates["h"], background_gmask.copy(), 1)
        roi_current_frame = cv2.erode(roi_current_frame, np.ones((3, 3), np.uint8))
        d_background_gmask = cv2.dilate(roi_current_frame, np.ones((7, 7), np.uint8))

        # substract back and foreground
        roi_current_frame, coordinates = get_roi(roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"],
                                                 roi_coordinates["h"], current_frame.copy(), 1)
        fg = cv2.bitwise_and(roi_current_frame, roi_current_frame, mask=d_background_gmask)

        # now filter out our white ball
        white_mask = get_mask(fg, self.b_lower_white, self.b_upper_white)
        white_mask = cv2.dilate(white_mask, np.ones((15, 15), np.uint8))
        # white_mask = cv2.erode(white_mask, np.ones((5, 5), np.uint8))

        # cv2.imshow("w", white_mask)
        # cv2.waitKey(0)

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

        if found_contours:
            for c in contours:
                if cv2.contourArea(c) < 100:
                    continue

                tx, ty, tw, th = cv2.boundingRect(c)
                right = roi_coordinates["x"] + tx + tw
                left = roi_coordinates["x"] + tx

                if tx == x1 and ty == y1 and tx + tw == x1 + w1 and ty + th == y1 + h1:
                    continue

                # ball goes left and there is something moving in front of it
                if ball.left and ball.currentPosition_center_x < border_center and \
                                right < ball.currentPosition_center_x and right > border_left:
                    ball.table.border_left = roi_coordinates["x"] + tx + tw
                    ball.table.border_right = tright

                # ball goes right and there is something moving in front of it
                if not ball.left and ball.currentPosition_center_x > border_center and \
                                left > ball.currentPosition_center_x and left < border_right:
                    ball.table.border_right = roi_coordinates["x"] + tx
                    ball.table.border_left = tleft

        result = current_frame.copy()
        cv2.drawContours(result, [box.astype("int")], -1, (250, 150, 0), 5)

        return result

    def narrow_down_roi_for_ball(self, ball, tleft, tright):
        """This Function tries to narrow down the roi for tracking the ball
            depending on the known position and direction of the ball.

        Args:
            self: This class.
            ball: Ball Class containing all important information of the ball.
            tleft: Most left point of the roi.
            tright: Most right point of the roi.

        Returns:
            Returns new roi coordinates.

        """
        border_left = ball.table.border_left
        border_right = ball.table.border_right
        border_center = ball.table.border_center
        if ball.lastPosition_center_x == 0 and ball.lastPosition_center_y == 0 or \
                                ball.lastPosition_center_x == ball.currentPosition_center_x \
                        and ball.lastPosition_center_y == ball.currentPosition_center_y:
            return {}
        # ball goes left up
        if ball.left and ball.up:
            x = ball.currentPosition_x - 2 * ball.currentPosition_w + int(ball.currentPosition_w / 2)
            y = ball.currentPosition_y - 2 * ball.currentPosition_h
            w = ball.currentPosition_w + 3 * ball.currentPosition_w
            h = ball.currentPosition_h + 3 * ball.currentPosition_h

        # ball goes left down
        elif ball.left and not ball.up:
            x = ball.currentPosition_x - 2 * ball.currentPosition_w + int(ball.currentPosition_w / 2)
            y = ball.currentPosition_y - int(ball.currentPosition_h / 2)
            w = ball.currentPosition_w + 3 * ball.currentPosition_w
            h = ball.currentPosition_h + 2 * ball.currentPosition_h

        # ball goes right up
        elif not ball.left and ball.up:
            x = ball.currentPosition_x - int(ball.currentPosition_w / 2)
            y = ball.currentPosition_y - 2 * ball.currentPosition_h
            w = ball.currentPosition_w + 2 * ball.currentPosition_w
            h = ball.currentPosition_h + 3 * ball.currentPosition_h

        # ball goes right down
        elif not ball.left and not ball.up:
            x = ball.currentPosition_x - int(ball.currentPosition_w / 2)
            y = ball.currentPosition_y - int(ball.currentPosition_h / 2)
            w = ball.currentPosition_w + 2 * ball.currentPosition_w
            h = ball.currentPosition_h + 2 * ball.currentPosition_h

        if ball.currentPosition_center_x < border_center and border_left != tleft and x < border_left:
            temp = border_left - x
            x = x + temp
        if ball.currentPosition_center_x > border_center and border_right != tright and x + w > border_right:
            temp = (x + w) - border_right
            x = x - temp

        return {"x": x, "y": y, "w": w, "h": h}