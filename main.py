import numpy as np
import cv2
import argparse
import ball as b
import table as t
import os.path
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
    """This Function tries to narrow down the roi for tracking the ball
        depending on the known position and direction of the ball.

    Args:
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
    """This function tries the find the ball in the current frame.

    Args:
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

    #add current frame for background subtraction
    background_gmask = fgbg.apply(current_frame.copy())

    found_contours = False
    # narrow background to the roi and dilate the result
    coordinates = narrow_down_roi_for_ball(ball, tleft, tright)

    if len(coordinates) != 0 and coordinates["x"] + coordinates["w"] < tright and coordinates["x"] > tleft:
        #we know the position and direction of the ball so we can update te roi
        roi_coordinates = coordinates
        check_negative_val = [roi_coordinates["x"], roi_coordinates["y"], roi_coordinates["w"], roi_coordinates["h"]]
        found_contours = True

        if len(ball.box) != 0 and len(ball.lastBox) != 0 and not ball.position_change() or min(check_negative_val) < 0:
            ball.updated_position(ball.box)
            return []

    #cv2.imshow("b", background_gmask)
    #cv2.waitKey(0)
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

    if found_contours:
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


def main(file, blue_table, table_straight, print_output, create_video, do_prediction):
    """The main function of the application.

    Args:
        file: File to read the video from.
        blue_table: Flag if table is blue.
        table_straight: Flag if table is straight.
        print_output: Flag if result should be shown.
        create_video: Flag if should be created into a video.
        do_prediction:  Flag if prediction parable should be drawn.

    Returns:
        None.

    """
    ball = b.Ball()
    ball.table = t.Table()

    if not os.path.isfile(file):
        print("ERROR: Could not find file %s" % file)
        return
    camera = cv2.VideoCapture(file)
    (grabbed, first_frame) = camera.read()
    height, width = first_frame.shape[:2]

    if create_video:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter((file+'_out.avi'),fourcc, 20.0, (width, height))

    print_frame( first_frame, "input_first_frame")

    tleft, ttop, tright, tbottom = ball.table.get_table(first_frame, blue_table, table_straight)

    last_frame = first_frame.copy()

    fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=70, detectShadows=False)
    fgbg.apply(last_frame)

    result = draw_border(tleft, ttop, tright, tbottom, last_frame.copy(), (0, 255, 0))
    print_frame(result, "currentFrame", print_output)
    if create_video:
        out.write(result)

    while camera.isOpened():
        (grabbed, current_frame) = camera.read()
        if not grabbed:
            break

        result = get_ball_position(ball, current_frame.copy(),
                                   {"x": tbottom[0], "y": 0, "w": ttop[0] - tbottom[0], "h": tbottom[1]}, fgbg, tleft[0], tright[0])
        if len(result) == 0:
            result = current_frame.copy()

        result = draw_border(tleft, ttop, tright, tbottom, result.copy(), (0, 255, 0))
        if len(result) == 0:
            ball.draw_arrow(result)
            ball.draw_parable(result)
            continue

        ball.draw_arrow(result)
        if do_prediction:
            ball.draw_parable(result)

        if create_video:
            out.write(result)

        print_frame(result, "currentFrame", print_output)

        last_frame = result.copy()
    camera.release()
    if create_video:
        out.release()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Motion Prediction of a Ping Pong Ball')

    parser.add_argument('-f', action="store", dest="videoFile", default="", type=str,
                        help="Path to the input video")#, required=True)
    parser.add_argument('-gt', action="store_true", dest="greenTable",
                        help="If Flag is set table is assumed to be green. Default is blue (default False)")
    parser.add_argument('-ts', action="store_true", dest="tableStraight",
                        help="If Flag is set table is assumed to stand horizontal in the Video (default False)")
    parser.add_argument('-ns', action="store_true", dest="doNotShow",
                        help="If Flag is set results is NOT show while processing (default False)")
    parser.add_argument('-o', action="store_true", dest="createOutput",
                        help="If Flag is set output video is created (default False)")
    parser.add_argument('-np', action="store_true", dest="noPrediction",
                        help="If Flag is set prediction path parable are NOT drawn in the output (default False)")

    arg = parser.parse_args()

    main(arg.videoFile, not arg.greenTable, arg.tableStraight, not arg.doNotShow, arg.createOutput, not arg.noPrediction)
    # main("ggg.mp4", True, False, True, False, True)
    #main("gtt4.mp4", True, False, True, False, True)
    #main("tt1.mp4", False, False, True, False, True)
    #main("ggt1.mp4", True, False, True, False, True)
    #main("ggt.mp4", True, False, True, False, True )

    pass

