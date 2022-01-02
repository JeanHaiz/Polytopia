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
    scores = get_scores(clear)
    return scores


def read(image):
    return pytesseract.image_to_string(image)


def crop(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    minLineLength = 500
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/2, threshold=200, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=10)
    heights = sorted([lines[i][0][1] for i in range(len(lines))])
    return image[heights[0]:heights[-1], :]


def get_scores(image):
    logger.debug("read image scores")
    logger.debug(read(image))
    logger.debug([f(t) for t in read(image).split('\n') if "score" in t])
    return [f(t) for t in read(image).split('\n') if "score" in t]


def f(t):
    s1 = t.split(", ")
    if "Unknown ruler" in s1[0].strip() or "Ruled by you" in s1[0].strip():
        player = s1[0]
    else:
        player = s1[0].strip().split(" ")[-1]
    s2 = s1[1].split(": ")
    score = int(s2[1].split(" ")[0].replace(",", ""))
    return (player, score)
