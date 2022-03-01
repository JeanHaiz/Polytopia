import os
import cv2

from common import image_utils
from common.image_utils import ImageOp

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


def get_one_color_edges(image, channel=None, filename=None, low=50, high=150):
    edges = cv2.Canny(image, low, high, apertureSize=3)
    if DEBUG and filename and channel:
        image_utils.save_image(edges, channel.name, filename, ImageOp.ONE_COLOR_EDGES)
    return edges


def get_three_color_edges(img, channel_name=None, filename=None, border=None, low_thresh=200, high_thresh=250):
    if border is not None:
        img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT)

    # Split out each channel
    blue, green, red = cv2.split(img)

    # Run canny edge detection on each channel
    # TODO improve efficiency!
    blue_edges = cv2.Canny(blue, low_thresh, high_thresh)
    green_edges = cv2.Canny(green, low_thresh, high_thresh)
    red_edges = cv2.Canny(red, low_thresh, high_thresh)

    # Join edges back into image
    edges = blue_edges | green_edges | red_edges

    if DEBUG and filename is not None:
        image_utils.save_image(edges, channel_name, filename, ImageOp.THREE_COLOR_EDGES)
    return edges
