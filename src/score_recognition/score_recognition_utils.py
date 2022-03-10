import re
import cv2
import pytesseract

import numpy as np
from common.logger_utils import logger


def is_score_reconition_request(reactions, attachment, filename):
    # TODO: complete with image analysis heuristics
    return "📈" in [r.emoji for r in reactions]


def read(image, config=''):
    return pytesseract.image_to_string(image, config=config)


def crop(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    minLineLength = 500
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 2, threshold=200, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=10)
    heights = sorted([lines[i][0][1] for i in range(len(lines))])
    return image[heights[0]:heights[-1], :]


def read_scores(image):
    clear = clear_noise_optimised(image)
    logger.debug("read image scores")
    image_reading = read(clear)
    logger.debug("image reading: %s" % image_reading)
    image_text = image_reading.split('\n')
    print("image text", image_text)
    scores = [read_line(t) for t in image_text if "score" in t]
    logger.debug(scores)
    print("scores", scores)
    return scores


def read_line(line):
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
    return (player, score)


def clear_noise_optimised(image):
    value = (np.array(0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0])).astype(np.uint8)
    return cv2.threshold(value, 170, 255, cv2.THRESH_BINARY_INV)[1]


def ii(image, xx, yy, img_x, img_y):
    if yy >= img_y or xx >= img_x:
        return 0
    pixel = image[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]


def clear_noise_slow(image):
    # pixel intensity = 0.30R + 0.59G + 0.11B
    img_y = len(image)
    img_x = len(image[0])
    new_image = image.copy()
    new_image.fill(255)
    width = image.shape[1]
    height = image.shape[0]

    fg = 255
    bg = 0
    fg_int = 170

    # Loop through every pixel in the box and color the
    # pixel accordingly
    for x in range(width):
        for y in range(height):
            if ii(image, x, y, img_x, img_y) > fg_int:
                new_image[y][x] = bg
            else:
                new_image[y][x] = fg
    # new_image = [[ii(pixel) for pixel in row] for row in image]
    return new_image
