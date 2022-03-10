import re
import os
import cv2
import pytesseract

from common import image_utils
from common.image_utils import ImageOp

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


def prepare_turn_image(image, low_thresh):
    crop = image[:int(image.shape[0] / 4), :]
    grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(grayImage, low_thresh, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for cnt in cnts:
        if cv2.contourArea(cnt) < 15:
            cv2.fillPoly(thresh, [cnt], color=(0, 0, 0))
    return thresh


def get_turn(image, low_thresh=130, channel_name=None):
    selected_image = prepare_turn_image(image, low_thresh)
    cv2.imwrite('./bin.png', 255 - selected_image)
    # map_text = pytesseract.image_to_string(selected_image, config='--oem 3 --psm 6').split("\n")

    contours, _ = cv2.findContours(selected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rows, row_mins, row_maxes = split_in_rows(contours)

    delta = 15
    turn = None
    for i in rows.keys():
        row_min_i = max(row_mins[i] - delta, 0)
        row_max_i = min(row_maxes[i] + delta, selected_image.shape[0])
        row_image = selected_image[row_min_i:row_max_i]
        if DEBUG and channel_name is not None:
            image_utils.save_image(row_image, channel_name, "binary_%d" % i, ImageOp.TURN_PIECES)
        row_text = pytesseract.image_to_string(
            255 - row_image, config='--psm 6 -c load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0')
        cleared_row_text = row_text.replace("\n", "").replace("\x0c", "")
        cleared_row_text = re.sub(r"[^a-zA-Z0-9 ]", "", cleared_row_text)
        if 'Scores' not in cleared_row_text and 'Stars' not in cleared_row_text:
            cleared_row_numbers = re.sub(r"[^0-9 ]", "", cleared_row_text).replace("  ", " ").strip()
            cleared_row_text_split = cleared_row_numbers.split(" ")
            if len(cleared_row_text_split) > 2 and cleared_row_text_split[2].isnumeric():
                turn = cleared_row_text_split[2]

    if turn is None:
        if low_thresh > 160:
            if DEBUG:
                print("turn not recognised")
        else:
            return get_turn(image, low_thresh=low_thresh + 10)
    else:
        if DEBUG:
            print("turn %s" % turn)
    return turn


def split_in_rows(contours):
    rows = {}
    row_mins = {}
    row_maxes = {}

    for c in contours:
        (y, x, h, w) = cv2.boundingRect(c)
        if w > 10 and h > 10:
            found_row = False
            for i in rows.keys():
                if not found_row:
                    row_min_i = row_mins[i]
                    row_max_i = row_maxes[i]
                    if x >= row_min_i and x + w <= row_max_i:
                        found_row = True
                        rows[i] = rows[i] + [c]
                    elif x >= row_min_i and x <= row_max_i and x + w >= row_max_i:
                        row_max_i = x + w
                        found_row = True
                        rows[i] = rows[i] + [c]
                    elif x <= row_min_i and x + w >= row_min_i and x + w <= row_max_i:
                        row_min_i = x
                        found_row = True
                        rows[i] = rows[i] + [c]
                    elif x <= row_min_i and x + w >= row_max_i:
                        row_min_i = x
                        row_max_i = x + w
                        found_row = True
                        rows[i] = rows[i] + [c]

            if not found_row:
                new_i = len(rows.keys())
                rows[new_i] = [c]
                row_mins[new_i] = x
                row_maxes[new_i] = x + w
    return rows, row_mins, row_maxes
