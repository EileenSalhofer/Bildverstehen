import numpy as np
import cv2
import ball as b
import table as t
from helper import print_frame
from helper import get_mask
from helper import get_biggest_contour
from helper import get_roi
from helper import corners
from helper import draw_border



# White for ball is more tolerant
b_lower_white = np.array([00, 10, 170], dtype="uint8")
b_upper_white = np.array([180, 100, 255], dtype="uint8")




def narrow_down_roi_for_ball(ball, tleft, tright):
    border_left = ball.table.border_left
    border_right = ball.table.border_right
    border_center = ball.table.border_center
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
    border_left = ball.table.border_left
    border_right = ball.table.border_right
    border_center = ball.table.border_center

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
                            right < ball.currentPosition_center_x and right > border_left:
                ball.table.  border_left = roi_coordinates["x"] + tx+tw
                ball.table.border_right = tright

            # ball goes right and there is something moving in front of it
            if not ball.left and ball.currentPosition_center_x > border_center and\
                            left > ball.currentPosition_center_x  and left < border_right:
                ball.table.border_right = roi_coordinates["x"] + tx
                ball.table.border_left = tleft


    result = current_frame.copy()
    cv2.drawContours(result, [box.astype("int")], -1, (250, 150, 0), 5)

    return result




def main(file, video, blue_table, pos_straight):
    ball = b.Ball()
    table = t.Table()

    if video:
        camera = cv2.VideoCapture(file)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))

        (grabbed, first_frame) = camera.read()
        print_frame( first_frame, "input_first_frame")

        tleft, ttop, tright, tbottom = table.get_table(first_frame, blue_table, pos_straight)
        table.define_borders(tleft, ttop, tright, tbottom)
        ball.table = table

        x = tleft[0]
        y = 0
        w = tright[0]-tleft[0]
        h = tbottom[1]
        roi_first_frame, coordinates = get_roi(x, y, w, h, first_frame.copy(), 1)
        #print_frame(roi_first_frame, "roi_first_frame", True)

        last_frame = first_frame.copy()

        fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=70, detectShadows=False)
        fgbg.apply(last_frame)

        result = draw_border(tleft, ttop, tright, tbottom, last_frame.copy(), (0, 255, 0))
        #cv2.imshow("run", result)
        #cv2.waitKey(0)

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
    #main("ggg.mp4", True, False, True)
    main("gtt4.mp4", True, False, False)
    pass

