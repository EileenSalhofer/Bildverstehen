import cv2
import numpy as np


def get_mask(frame, lower_color, upper_color):
    """This Function uses color thresholding to create a filter mask.

    Args:
        lower_color: lower bound for color thresholding.
        upper_color: upper bound for color thresholding.

    Returns:
        Returns mask for colors in range of the lower and upper bound.

    """
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return mask


def print_frame(frame, name, show_image=False):
    """This Function saves and displays given image.

    Args:
        frame: The frame or image to be displayed.
        name: The displayed and saved name.
        show_image: Control Flag if False image wont be displayed or printed.

    Returns:
        None

    """
    if show_image:
        cv2.imwrite(name + '.png', frame)

        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
        cv2.imshow(name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pass


def parable_get_points_between(ptr1, ptr2, ptr3):
    """This Function calculates an parable resulting from the three
        given points and returns the parable as an array of points.

    Args:
        ptr1: First Point on the parable with the lowest x coordinate.
        ptr2: Second Point on the parable between ptr1 and ptr2.
        ptr3: Third Point on the parable with the highest x coordinate.

    Returns:
        Returns an array of points defining the parable between the points.

    """
    A1 = -(ptr1[0] ** 2) + (ptr2[0] ** 2)
    B1 = -(ptr1[0]) + (ptr2[0])
    D1 = -(ptr1[1]) + (ptr2[1])

    A2 = -(ptr2[0] ** 2) + (ptr3[0] ** 2)
    B2 = -(ptr2[0]) + (ptr3[0])
    D2 = -(ptr2[1]) + (ptr3[1])

    Bm = -(B2 / B1)

    A3 = Bm * A1 + A2
    D3 = Bm * D1 + D2

    a = D3 / A3
    b = (D1 - (A1 * a)) / B1
    c = ptr1[1] - (a * (ptr1[0] ** 2)) - (b * ptr1[0])

    points = []
    points.append(ptr1)

    ptr2_in_array = False
    # Calculate points values
    for x in range(abs(ptr1[0] - ptr3[0])):
        x = x + ptr1[0]
        if x > ptr2[0] and not ptr2_in_array:
            points.append(ptr2)
            ptr2_in_array = True
            continue
        y = abs(int((a * (x ** 2)) + (b * x) + c))
        points.append([x, y])
    return points


def get_roi(x, y, w, h, frame, tolerance):
    """This Function cuts out the roi from a frame or image. x,y are on the top left.

    Args:
        x: x coordinate of the roi.
        y: y coordinate of the roi.
        w: The width of the roi.
        h: The height of the roi.
        frame: Frame or image form that the roi is cut out.
        tolerance: A relative increase of the roi. Will be ignored if one or lower.

    Returns:
        Returns the roi and the coordinates to trace back from where it was cut out.

    """
    # increase size for more tolerance
    height, width = frame.shape[:2]
    if tolerance > 1:
        tolerance_x = int((w / width) * tolerance)
        tolerance_y = int((h / height) * tolerance)
        x = x - tolerance_x
        if x < 1:
            x = 1
        w = w + tolerance_x * 2
        y = y - tolerance_y
        if y < 1:
            y = 1
        h = h + tolerance_y * 2

    return frame[y:(y + h), x:(x + w)], {"x": x, "y": y, "w": w, "h": h}


def get_biggest_contour(mask):
    """This Function finds the biggest contour in the given image.

    Args:
        mask: The image to search for the contour.

    Returns:
        The biggest contour. If none is found an empty array.

    """
    (_, contours, _) = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return contours
    c = max(contours, key=cv2.contourArea)
    return c


def corners(frame):
    """This Function returns the corners in a image.

    Args:
        frame: The image to search for corners.

    Returns:
        The return a list of corners found in the image

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    return dst


def draw_border(pt1, pt2, pt3, pt4, frame, color):
    """This Function clockwise draws a line between the single points.

    Args:
        pt1: The first point.
        pt2: The second point.
        pt3: The third point.
        pt4: The fourth point.
        frame: Image to draw the border
        color: Color of the border

    Returns:
        Returns image containing the newly drawn border

    """
    cv2.line(frame, pt1, pt2, color, 6)
    cv2.line(frame, pt2, pt3, color, 6)
    cv2.line(frame, pt3, pt4, color, 6)
    cv2.line(frame, pt4, pt1, color, 6)
    return frame
