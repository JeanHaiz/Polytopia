#!/usr/bin/python

# Processes an image to extract the text portions. Primarily
# used for pre-processing for performing OCR.

# Based on the paper "Font and Background Color Independent Text Binarization" by
# T Kasar, J Kumar and A G Ramakrishnan
# http://www.m.cs.osakafu-u.ac.jp/cbdar2007/proceedings/papers/O1-1.pdf

# Copyright (c) 2012, Jason Funk <jasonlfunk@gmail.com>
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import numpy as np

from common import image_processing_utils

DEBUG = 0


# Determine pixel intensity
# Apparently human eyes register colors differently.
# TVs use this formula to determine
# pixel intensity = 0.30R + 0.59G + 0.11B
def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        # print "pixel out of bounds ("+str(y)+","+str(x)+")"
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]


# A quick test to check whether the contour is
# a connected shape
def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1


# Helper function to return a given contour
def c(index):
    global contours
    return contours[index]


# Count the number of real children
def count_children(index, h_, contour, min_size):
    # No children
    if h_[index][2] < 0:
        return 0
    else:
        # If the first child is a contour we care about
        # then count it, otherwise don't
        if keep(c(h_[index][2]), min_size):
            count = 1
        else:
            count = 0

            # Also count all of the child's siblings and their children
        count += count_siblings(h_[index][2], h_, contour, min_size, True)
        return count


# Quick check to test if the contour is a child
def is_child(index, h_, min_size):
    return get_parent(index, h_, min_size) > 0


# Get the first parent of the contour that we care about
def get_parent(index, h_, min_size):
    parent = h_[index][3]
    while not keep(c(parent), min_size) and parent > 0:
        parent = h_[parent][3]

    return parent


# Count the number of relevant siblings of a contour
def count_siblings(index, h_, contour, min_size, inc_children=False):
    # Include the children if necessary
    if inc_children:
        count = count_children(index, h_, contour, min_size)
    else:
        count = 0

    # Look ahead
    p_ = h_[index][0]
    while p_ > 0:
        if keep(c(p_), min_size):
            count += 1
        if inc_children:
            count += count_children(p_, h_, contour, min_size)
        p_ = h_[p_][0]

    # Look behind
    n = h_[index][1]
    while n > 0:
        if keep(c(n), min_size):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour, min_size)
        n = h_[n][1]
    return count


# Whether we care about this contour
def keep(contour, min_size):
    global img_y, img_x
    return keep_box(contour, img_x, img_y, min_size) and connected(contour)


# Whether we should keep the containing box of this
# contour based on it's shape
def keep_box(contour, img_x, img_y, min_size):
    xx, yy, w_, h_ = cv2.boundingRect(contour)

    # width and height need to be floats
    w_ *= 1.0
    h_ *= 1.0

    # Test it's shape - if it's too oblong or tall it's
    # probably not a real character
    if w_ / h_ < 0.1 or w_ / h_ > 10:
        if DEBUG:
            print("\t Rejected because of shape: (" + str(xx) + "," + str(yy) + "," +
                  str(w_) + "," + str(h_) + ")" + str(w_ / h_))
        return False

    # check size of the box
    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < min_size):
        if DEBUG:
            print("\t Rejected because of size")
        return False

    return True


def include_box(index, h_, contour, min_size):
    if DEBUG:
        print(str(index) + ":")
        if is_child(index, h_, min_size):
            print("\tIs a child")
            print("\tparent " + str(get_parent(index, h_, min_size)) + " has " + str(
                count_children(get_parent(index, h_, min_size), h_, contour, min_size)) + " children")
            print("\thas " + str(count_children(index, h_, contour, min_size)) + " children")

    if is_child(index, h_, min_size) and count_children(get_parent(index, h_, min_size), h_, contour, min_size) <= 2:
        if DEBUG:
            print("\t skipping: is an interior to a letter")
        return False

    if count_children(index, h_, contour, min_size) > 2:
        if DEBUG:
            print("\t skipping, is a container of letters")
        return False

    if DEBUG:
        print("\t keeping")
    return True


def get_corner_background_intensity(x_, y_, width, height):
    return [
        # bottom left corner 3 pixels
        ii(x_ - 1, y_ - 1),
        ii(x_ - 1, y_),
        ii(x_, y_ - 1),

        # bottom right corner 3 pixels
        ii(x_ + width + 1, y_ - 1),
        ii(x_ + width, y_ - 1),
        ii(x_ + width + 1, y_),

        # top left corner 3 pixels
        ii(x_ - 1, y_ + height + 1),
        ii(x_ - 1, y_ + height),
        ii(x_, y_ + height + 1),

        # top right corner 3 pixels
        ii(x_ + width + 1, y_ + height + 1),
        ii(x_ + width, y_ + height + 1),
        ii(x_ + width + 1, y_ + height)
    ]


def clear_noise(orig_img, border_width=50, low_threshold=200, high_threshold=250, blur_kernel=2, min_size=15):
    global img, img_y, img_x, contours

    # Add a border to the image for processing sake
    img = cv2.copyMakeBorder(orig_img, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT)

    # Calculate the width and height of the image
    img_y = len(img)
    img_x = len(img[0])

    if DEBUG:
        print("Image is " + str(len(img)) + "x" + str(len(img[0])))

    edges = image_processing_utils.get_three_color_edges(img, low_thresh=low_threshold, high_thresh=high_threshold)

    # Find the contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    hierarchy = hierarchy[0]

    processed = edges.copy() if DEBUG else None
    rejected = edges.copy() if DEBUG else None

    keepers = select_contours(hierarchy, min_size, processed, rejected)

    new_image = invert_bg_fg_if_needed(edges, keepers)

    # blur a bit to improve ocr accuracy
    if blur_kernel is not None:
        new_image = cv2.blur(new_image, (blur_kernel, blur_kernel))
    return new_image


def select_contours(hierarchy, min_size, processed=None, rejected=None):
    # These are the boxes that we are determining
    keepers = []

    # For each contour, find the bounding rectangle and decide
    # if it's one we care about
    for index_, contour_ in enumerate(contours):
        if DEBUG:
            print("Processing #%d" % index_)

        x, y, w, h = cv2.boundingRect(contour_)

        # Check the contour and it's bounding box
        if keep(contour_, min_size) and include_box(index_, hierarchy, contour_, min_size):
            # It's a winner!
            keepers.append([contour_, [x, y, w, h]])
            if DEBUG and processed is not None:
                cv2.rectangle(processed, (x, y), (x + w, y + h), (100, 100, 100), 1)
                cv2.putText(processed, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        else:
            if DEBUG and rejected is not None:
                cv2.rectangle(rejected, (x, y), (x + w, y + h), (100, 100, 100), 1)
                cv2.putText(rejected, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    return keepers


def invert_bg_fg_if_needed(edges, keepers):
    # Make a white copy of our image
    new_image = edges.copy()
    new_image.fill(255)

    # For each box, find the foreground and background intensities
    for index_, (contour_, box) in enumerate(keepers):

        # Find the average intensity of the edge pixels to
        # determine the foreground intensity
        fg_int = sum(ii(p[0][0], p[0][1]) for p in contour_) / len(contour_)

        if DEBUG:
            print("FG Intensity for #%d = %d" % (index_, fg_int))

        # Find the intensity of three pixels going around the
        # outside of each corner of the bounding box to determine
        # the background intensity
        x_, y_, width, height = box
        bg_int = get_corner_background_intensity(x_, y_, width, height)

        # Find the median of the background
        # pixels determined above
        bg_int = np.median(bg_int)

        if DEBUG:
            print("BG Intensity for #%d = %s" % (index_, repr(bg_int)))

        # Determine if the box should be inverted
        if fg_int >= bg_int:
            fg = 255
            bg = 0
        else:
            fg = 0
            bg = 255

        # Loop through every pixel in the box and color the
        # pixel accordingly
        for x in range(x_, x_ + width):
            for y in range(y_, y_ + height):
                if y >= img_y or x >= img_x:
                    if DEBUG:
                        print("pixel out of bounds (%d,%d)" % (y, x))
                    continue
                if ii(x, y) > fg_int:
                    new_image[y][x] = bg
                else:
                    new_image[y][x] = fg
    return new_image
