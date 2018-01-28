import cv2
import numpy as np


def get_mask(frame, lower_color, upper_color):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return mask


def print_frame(frame, name, show_image=False):
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    cv2.imwrite(name + '.png', frame)
    if show_image:
        cv2.imshow(name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parable_get_points_between(ptr1, ptr2, ptr3):
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

    return dst

def draw_border(top_left, top_right, bottom_right, bottom_left, frame, color):
    cv2.line(frame, top_left, bottom_left, color, 6)
    cv2.line(frame, bottom_left, bottom_right, color, 6)
    cv2.line(frame, top_left, top_right, color, 6)
    cv2.line(frame, top_right, bottom_right, color, 6)
    return frame
