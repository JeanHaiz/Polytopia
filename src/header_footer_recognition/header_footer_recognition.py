import re
import os
import cv2
import numpy as np
import pytesseract

from typing import List
from typing import Union
from typing import Callable
from typing import Tuple

from common import image_utils
from common.image_operation import ImageOp

from database.database_client import get_database_client
from header_footer_recognition.header_footer_recognition_error import HeaderFooterRecognitionException

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

database_client = get_database_client()


def turn_recognition_request(
        patch_uuid: str,
        turn_requirement_id: str,
        channel_id: int,
        channel_name: str,
        message_id: int,
        resource_number: int,
        filename: str,
        callback: Union[Callable[[str, str], None], Callable[[str, int, str], None]]
) -> None:
    image = image_utils.load_image(channel_name, filename, ImageOp.MAP_INPUT)
    
    if image is None:
        raise HeaderFooterRecognitionException("TURN RECOGNITION - IMAGE NOT FOUND: %s, %s" % (channel_name, filename))
    
    turn = get_turn(
        image,
        filename,
        channel_name
    )
    
    if turn is not None:
        database_client.set_filename_header(message_id, resource_number, turn)
        last_turn = database_client.get_last_turn(channel_id)
        if last_turn is None or int(last_turn) < turn:
            database_client.set_new_last_turn(channel_id, turn)
    else:
        raise HeaderFooterRecognitionException("TURN RECOGNITION - TURN NOT RECOGNISED")
    
    callback(
        patch_uuid,
        turn_requirement_id
    )
    
    print("turn recognition done, callback sent", flush=True)


def __prepare_turn_image(image: np.ndarray, low_thresh: int) -> np.ndarray:
    crop = image[:int(image.shape[0] / 4), :]
    gray_image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray_image, low_thresh, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for cnt in cnts:
        if cv2.contourArea(cnt) < 15:
            cv2.fillPoly(thresh, [cnt], color=(0, 0, 0))
    return thresh


def get_turn(
        image: np.ndarray,
        filename: str,
        channel_name: str,
        low_thresh=130
) -> int:
    selected_image = __prepare_turn_image(image, low_thresh)
    if DEBUG:
        image_utils.save_image(255 - selected_image, channel_name, filename + "_bin", ImageOp.SCORE_PROCESSED_IMAGE)
    # map_text = pytesseract.image_to_string(selected_image, config='--oem 3 --psm 6').split("\n")

    contours, _ = cv2.findContours(selected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rows, row_mins, row_maxes = __split_in_rows(contours)

    delta = 15
    turn = -1
    for i in rows.keys():
        row_min_i = max(row_mins[i] - delta, 0)
        row_max_i = min(row_maxes[i] + delta, selected_image.shape[0])
        row_image = selected_image[row_min_i:row_max_i]
        if DEBUG and channel_name is not None:
            image_utils.save_image(row_image, channel_name, filename + "_binary_%d" % i, ImageOp.TURN_PIECES)
        row_text = pytesseract.image_to_string(
            255 - row_image, config='--psm 6 -c load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0')
        cleared_row_text = row_text.replace("\n", "").replace("\x0c", "")
        cleared_row_text = re.sub(r"[^a-zA-Z0-9 ]", "", cleared_row_text)
        if 'Scores' not in cleared_row_text and 'Stars' not in cleared_row_text:
            cleared_row_numbers = re.sub(r"[^0-9 ]", "", cleared_row_text).replace("  ", " ").strip()
            cleared_row_text_split = cleared_row_numbers.split(" ")
            if len(cleared_row_text_split) > 2 and cleared_row_text_split[2].isnumeric():
                if DEBUG:
                    print("Turn was " + str(turn) + " and is now " + str(int(cleared_row_text_split[2])))
                turn = int(cleared_row_text_split[2])

    if turn is None or turn == -1:
        if low_thresh > 160:
            if DEBUG:
                print("turn not recognised")
        else:
            return get_turn(image, filename, channel_name, low_thresh=low_thresh + 10)
    else:
        if DEBUG:
            print("Recognised turn %d" % turn)
    return turn


def __split_in_rows(contours: List) -> Tuple[dict, dict, dict]:
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
                    elif row_min_i <= x <= row_max_i <= x + w:
                        row_max_i = x + w
                        found_row = True
                        rows[i] = rows[i] + [c]
                    elif x + w >= row_min_i >= x <= row_max_i:
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
