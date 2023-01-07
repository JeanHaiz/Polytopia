import re
import os
import cv2
import pytesseract

import numpy as np

from common.logger_utils import logger
from common import image_utils
from common.image_operation import ImageOp

from score_recognition.score_recognition_error import RecognitionException
from score_recognition import recognition_callback_utils

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


def score_recognition_request(
        patch_uuid: str,
        score_requirement_id: str,
        channel_id: int,
        channel_name: str,
        message_id: int,
        resource_number: int,
        filename: str
):
    image = image_utils.load_image(channel_name, filename, ImageOp.MAP_INPUT)
    
    if image is None:
        raise RecognitionException("SCORE RECOGNITION - IMAGE NOT FOUND: %s, %s" % (channel_name, filename))

    score = __recognise_scores(image)
    
    if score is None:
        raise RecognitionException("SCORE RECOGNITION - SCORE NOT RECOGNISED: %s, %s" % (channel_name, filename))

    image_utils.save_image(image, channel_name, filename, ImageOp.MAP_PROCESSED_IMAGE)

    recognition_callback_utils.send_recognition_completion(
        patch_uuid,
        score_requirement_id
    )
    
    print("score recognition done, callback sent", flush=True)


def __recognise_scores(image):
    clear = __clear_noise_optimised(image)
    logger.debug("read image scores")
    image_reading = __read(clear)
    logger.debug("image reading: %s" % image_reading)
    image_text = image_reading.split('\n')
    if DEBUG:
        print("image text", image_text)
    scores = [__read_line(t) for t in image_text if "score" in t]
    logger.debug(scores)
    if DEBUG:
        print("scores", scores)
    return scores


def __read(image, config=''):
    return pytesseract.image_to_string(image, config=config)


def __crop(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    min_line_length = 500
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 2, threshold=200, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=10)
    heights = sorted([lines[i][0][1] for i in range(len(lines))])
    return image[heights[0]:heights[-1], :]


def __read_line(line):
    print("line", line)
    line = re.sub(r"(\\[a-zA-Z0-9]*)", "", line)
    line = re.sub(r"[^a-zA-Z0-9,:]", "", line)
    print("re-line", line)
    s1 = line.split(",")

    s1 = [s1_i for s1_i in s1 if s1_i != '' and len(s1_i) > 1]

    if len(s1) >= 2:
        if "Unknownruler" in s1[0]:
            player = "Unknown ruler"
        elif "Ruledbyyou" in s1[0]:
            player = "Ruled by you"
        else:
            player = s1[0][len("Ruledby"):]

        if player is not None:
            player = re.sub(r"[^a-zA-Z0-9 ]", "", player)

        s2 = "".join(s1[1:]).split(":")
        if len(s2) >= 2:
            score = int(s2[1].split("points")[0].replace(",", ""))
        else:
            print("s2 error", line)
            return
    else:
        print("s1 error", line)
        return
    return player, score


def __clear_noise_optimised(image):
    value = (np.array(0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0])).astype(np.uint8)
    return cv2.threshold(value, 170, 255, cv2.THRESH_BINARY_INV)[1]
