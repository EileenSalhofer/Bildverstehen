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

