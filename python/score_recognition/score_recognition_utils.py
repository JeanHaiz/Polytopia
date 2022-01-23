import re
import cv2
import pytesseract

import numpy as np
from common.logger_utils import logger

from .score_image_processing_utils import clear_noise


def is_score_reconition_request(message, attachment, filename):
    # TODO: complete with image analysis heuristics
    # if attachment.content_type.startswith("image/"):
    # else:
    # logger.info("content not supported: %s" % message.id)
    return "📈" in [r.emoji for r in message.reactions]
    # return true  # discord.PartialEmoji(name="📈") in [r.emoji for r in message.reactions]


def read_scores(image):
    # image = cv2.imread(path)
    cropped = crop(image)
    clear = clear_noise(cropped)
    scores_you = get_scores(clear, only_you=True)
    scores = get_scores(image)
    return scores + scores_you


def read(image, config=''):
    return pytesseract.image_to_string(image, config=config)


def crop(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    minLineLength = 500
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/2, threshold=200, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=10)
    heights = sorted([lines[i][0][1] for i in range(len(lines))])
    return image[heights[0]:heights[-1], :]


def get_scores(image, only_you=False):
    logger.debug("read image scores")
    image_reading = read(image)
    logger.debug("image reading: %s" % image_reading)
    image_text = image_reading.replace("¢", "c").split('\n')
    print("image text", only_you, image_text)
    scores = [read_line(t) for t in image_text if "score" in t and (not only_you or "Ruled by you" in t)]
    logger.debug(scores)
    print("scores", only_you, scores)
    return scores


def read_line(line):
    print("line", line)
    line = line.replace(".", "").replace(" ", "")

    s1 = line.split(",")
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
    return (player, score)
