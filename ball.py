import numpy as np
import cv2
import table
from helper import parable_get_points_between

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

    def calculate_direction(self):
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
        border_left = self.table.border_left
        border_right = self.table.border_right
        border_center = self.table.border_center
        if not (self.up == True and self.lastUp == False) and self.frameCounter == 0 and self.lastLeft == self.left:
            self.upDownChange = False
            self.leftRightChange = False
            if self.lastLeft == self.left:
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0), 6)
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)
                return

            if self.left and (self.arrowhead[0]-2*abs(self.arrowtail[0] - self.arrowhead[0])) <= self.table.center_left_side[0] or not self.left and (self.arrowhead[0]+2*abs(self.arrowhead[0] - self.arrowtail[0])) >= self.table.center_right_side[0]:
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
                # p2 = (center_left_side[0] + int(abs(self.arrowhead[0]-center_left_side[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(2*abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1] - self.arrowhead[1])))
                p3 = self.arrowhead



                pts = np.array(parable_get_points_between(p1, p2, p3))

                pts[pts[0:,0] < border_left, 0] = border_left + (border_left-pts[pts[0:,0] < border_left, 0])

                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)

                # cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldup = pts
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
                cv2.polylines(frame, [self.oldleftright], False, (0, 0, 255), 6)

                # cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldup = pts
            return
        elif self.leftRightChange == True and self.lastLeft == self.left:
            self.frameCounter = 0
            self.upDownChange = False
            self.leftRightChange = False

            # ball goes left up
            if self.left and self.up:
                p1 = self.table.center_left_side
                #p2 = (center_left_side[0] + int(abs(self.arrowhead[0]-center_left_side[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0] - int(abs(self.arrowtail[0] - self.arrowhead[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.arrowhead

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldleftright = pts


            # ball goes right up
            elif not self.left and self.up:
                p1 = self.arrowhead
                #p2 = (center_right_side[0] - int(abs(center_right_side[0]-self.arrowhead[0])/2), self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])))
                p2 = (self.arrowhead[0]  + int(abs(self.arrowhead[0] - self.arrowtail[0])),
                      self.arrowhead[1] - int(abs(self.arrowtail[1]-self.arrowhead[1])/2))
                p3 = self.table.center_right_side

                pts = np.array(parable_get_points_between(p1, p2, p3))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 6)
                cv2.polylines(frame, [self.oldup], False, (255, 0, 0 ), 6)

                #cv2.polylines(frame, [self.oldguessedpath], False, (0, 0, 255), 6)
                self.oldleftright = pts

        else:
            self.frameCounter = 1
            #self.upDownChange = False
            #self.leftRightChange = False
        pass

