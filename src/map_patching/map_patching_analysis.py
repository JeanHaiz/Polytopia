import os
import cv2
import math

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from typing import List
from typing import Tuple
from typing import Optional
from datetime import datetime
from sklearn.cluster import DBSCAN

from common import image_utils
from common import image_processing_utils
from common.image_operation import ImageOp
from map_patching.image_param import ImageParam
from map_patching.corner_orientation import CornerOrientation
from database_interaction.database_client import DatabaseClient

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

# TODO: add analyse background loop over sizes


def analyse_map(
        map_image: np.ndarray,
        database_client: Optional[DatabaseClient],
        channel_name: str,
        filename: str) -> Tuple[str, ImageParam]:

    map_image_no_alpha = map_image[:, :, 0:3]
    print("DEBUG IMAGE")
    print(type(map_image_no_alpha))
    print(map_image_no_alpha.dtype)
    print(map_image_no_alpha.shape)

    alpha, scale = get_cloud_alpha_ter(map_image_no_alpha)
    image_utils.save_image(alpha, channel_name, filename + "_mask", ImageOp.MAP_PROCESSED_IMAGE)
    print("image scale", scale)

    map_with_alpha = attach_alpha(map_image_no_alpha, alpha)
    scaled_image = scale_image(map_with_alpha, scale)
    corners = get_corners(scaled_image, channel_name, filename)
    image_params = ImageParam(filename, scale, corners)
    if DEBUG:
        print(image_params)
    if database_client is not None:
        store_image_params(database_client, image_params)
    return store_transformed_image(scaled_image, channel_name, filename), image_params


def analyse_background(database_client: Optional[DatabaseClient], map_size: str) -> Tuple[str, ImageParam]:
    background_template = image_utils.get_background_template(map_size)
    _, scale = get_cloud_alpha_ter(background_template, is_background=True)
    print("background scale", scale)
    scaled_image = scale_image(background_template, scale)

    path, filename = image_utils.set_processed_background(scaled_image, map_size)

    corners = get_corners(scaled_image, "templates", "processed_background_template_%s" % map_size)

    image_params = ImageParam(filename, scale, corners)

    if DEBUG:
        print(image_params)
    # TODO: fix this, it won't work
    if database_client is not None:
        store_background_image_params(database_client, map_size, image_params)
    return path, image_params


def attach_alpha(map_image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    b_channel, g_channel, r_channel = cv2.split(map_image)
    return cv2.merge((b_channel, g_channel, r_channel, alpha))


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    # new_dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    new_dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


def store_image_params(database_client: DatabaseClient, image_params: ImageParam) -> None:
    return database_client.set_image_params(image_params)


def store_background_image_params(database_client: DatabaseClient, map_size: str, image_params: ImageParam) -> None:
    return database_client.set_background_image_params(map_size, image_params)


def store_transformed_image(image: np.ndarray, channel_name: str, filename: str) -> str:
    return image_utils.save_image(image, channel_name, filename, ImageOp.MAP_PROCESSED_IMAGE)


def get_cloud_alpha(
        map_image: np.ndarray,
        ksize: int = 7,
        sigma: int = 15,
        is_background: bool = False) -> Tuple[np.ndarray, float]:

    template = image_utils.get_cloud_template()
    hh, ww = template.shape[:2]

    pawn = template[:, :, 0:3]
    alpha_template = template[:, :, 3]
    alpha = cv2.merge([alpha_template, alpha_template, alpha_template])
    if DEBUG:
        print("alpha hist", np.histogram(alpha_template, bins=[-4, -3, -2, -1, 0, 1]))

    result_set = []

    # TODO Optimisation possibilities:
    # - not go linear, look for the point
    # - start in the middle and go out, based on gradient
    # - stop when it goes down again
    # - change spacing
    # - move range based on estiamted scaling
    # - add half-templates
    for int_scale in range(40, 400, 20):
        scale = int_scale / 100.0
        new_dim = (int(map_image.shape[1] * scale), int(map_image.shape[0] * scale))
        resized = cv2.resize(map_image, new_dim, interpolation=cv2.INTER_AREA)
        if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break

        img_alpha = np.ones((resized.shape[0:2]))

        corr_img = cv2.matchTemplate(resized, pawn, cv2.TM_CCOEFF_NORMED, mask=alpha)
        correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)
        threshold = 100

        maximum = np.max(blur)

        max_val = np.max(blur)
        rad = int(math.sqrt(hh * hh + ww * ww) / 4)

        count = 0

        while max_val > threshold:

            # find max value of correlation image
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

            if max_val > threshold:
                # draw match on copy of input
                img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] = \
                    img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] - alpha_template / 255.0

                # write black circle at max_loc in corr_img
                cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)
                count += 1

            else:
                break
        # cv2.imwrite("./alpha_before.png", (img_alpha * 255 / np.max(img_alpha)).astype("uint8"))
        hist = np.histogram(img_alpha, bins=[-4, -3, -2, -1, 0, 1])

        if DEBUG:
            print("hist", scale, int(sum(hist[0][:-1])), int(np.dot(hist[0][:-1], hist[1][1:-1] - 1) / scale), hist)
        img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8) * 255
        # mask = cv2.merge([img_alpha_c, img_alpha_c, img_alpha_c]).astype(np.uint8)
        alpha_area = np.sum(img_alpha_c == 0) / (img_alpha_c.shape[0] * img_alpha_c.shape[1])
        result_set.append([scale, maximum, maximum / scale, count, alpha_area, img_alpha_c])

    if False:  # not is_background:
        selected_scaling = sorted(result_set, key=lambda x: abs(1.21 - x[0]))[0]
    else:
        selected_scaling = sorted(result_set, key=lambda x: -x[2])[0]

    print("selected_scaling", selected_scaling[:5])
    resized_original = cv2.resize(
        selected_scaling[5], (map_image.shape[1], map_image.shape[0]), interpolation=cv2.INTER_AREA)
    return resized_original, selected_scaling[0]


def get_corners(
    image: np.ndarray,
    channel_name: str,
    filename: str
) -> List[Tuple[int, int, CornerOrientation]]:

    # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    LIGHT_COLOUR = (0, 0, 0)
    DARK_COLOUR = (10, 10, 10)

    mask = cv2.inRange(image[:, :, 0:3], LIGHT_COLOUR, DARK_COLOUR)

    contours, _ = cv2.findContours(255 - mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contour = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x: -x[1])[0]
    contour_mask = cv2.drawContours(np.zeros_like(mask), [selected_contour[0]], 0, (255, 255, 255))
    # edges = image_processing_utils.get_one_color_edges(mask)

    if DEBUG:
        image_utils.save_image(mask, channel_name, filename + "_background_edges", ImageOp.MAP_PROCESSED_IMAGE)
        image_utils.save_image(contour_mask, channel_name, filename + "_edges", ImageOp.MAP_PROCESSED_IMAGE)

    blur_edges = cv2.GaussianBlur(contour_mask, (11, 11), 3)

    _, threshold_edges = cv2.threshold(blur_edges, 3, np.max(blur_edges), cv2.THRESH_BINARY)

    # print("non-zero", np.sum(threshold_edges != 0))
    # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 200, 50)
    linesP = cv2.HoughLinesP(
        threshold_edges,
        10,
        np.pi / 180,
        40,
        None,
        int(max(image.shape[0], image.shape[0]) / 10),
        int(max(image.shape[0], image.shape[0]) / 20))

    print("all lines %d" % len(linesP))

    if DEBUG:
        lines_image = image.copy()
        for i in range(0, len(linesP)):
            line_i = linesP[i][0]
            cv2.line(lines_image, (line_i[0], line_i[1]), (line_i[2], line_i[3]), (0, 0, 255, 255), 3, cv2.LINE_AA)
        image_utils.save_image(lines_image, channel_name, filename + "_all_lines", ImageOp.MAP_PROCESSED_IMAGE)

    linesP = [line for line in linesP if slope_in_range(line[0], 0.5, 0.7)]
    print("filtered lines %d" % len(linesP))

    if DEBUG:
        lines_image = image.copy()
        for i in range(0, len(linesP)):
            line_i = linesP[i][0]
            cv2.line(lines_image, (line_i[0], line_i[1]), (line_i[2], line_i[3]), (0, 0, 255, 255), 3, cv2.LINE_AA)
        image_utils.save_image(lines_image, channel_name, filename + "_filtered_lines", ImageOp.MAP_PROCESSED_IMAGE)

    slopes = norm([get_line_slope(line[0]) for line in linesP])
    heights = norm([get_line_h0(line[0]) for line in linesP])

    labels = cluster_lines(slopes, heights)

    selected_lines = select_lines_by_length(linesP, labels)

    if DEBUG:
        lines_image = image.copy()
        for i in range(0, len(selected_lines)):
            line_i = selected_lines[i][0]
            cv2.line(lines_image, (line_i[0], line_i[1]), (line_i[2], line_i[3]), (0, 0, 255, 255), 3, cv2.LINE_AA)
        image_utils.save_image(lines_image, channel_name, filename + "_selected_lines", ImageOp.MAP_PROCESSED_IMAGE)

    return find_all_intersections(selected_lines)


def get_squared_length(line: List[int]) -> int:
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    return dx * dx + dy * dy


def select_lines_by_length(lines: List[np.ndarray], labels: np.ndarray) -> List[List[List[np.int32]]]:
    selected_lines = []
    for i in range(max(labels) + 1):
        label_lines = [(line, get_squared_length(line[0])) for j, line in enumerate(lines) if labels[j] == i]
        label_lines.sort(key=lambda x: -x[1])
        longest_line = label_lines[0][0]
        selected_lines.append(longest_line.tolist())
    return selected_lines


def get_line_h0(line: List[np.int32]) -> float:
    slope = get_line_slope(line)
    return float(line[1] - slope * line[0])


def norm(array: List[float]) -> np.ndarray:
    np_array: np.ndarray = np.array(array)
    min_array: float = min(array)
    max_array: float = max(array)
    value: np.ndarray = (np_array - min_array) / (max_array - min_array)
    return value


def get_line_slope(line: List[np.int32]) -> float:
    dx = float(line[2] - line[0])
    dy = float(line[3] - line[1])
    return dy / dx if dx != 0 else 100


def slope_in_range(line: List[np.int32], min_slope: float = 0.5, max_slope: float = 0.7) -> bool:
    # xstart, ystart, xend, yend
    return min_slope <= abs(get_line_slope(line)) <= max_slope


def get_intersection(first_line: List[List[np.int32]], second_line: List[List[np.int32]]) -> Optional[Tuple[int, int]]:
    h1 = get_line_h0(first_line[0])
    h2 = get_line_h0(second_line[0])
    p1 = get_line_slope(first_line[0])
    p2 = get_line_slope(second_line[0])
    if p1 - p2 == 0:
        return None
    x = int((h2 - h1) / (p1 - p2))
    y = int(p1 * x + h1)
    if not (-2000 <= x <= 10000) or not (-2000 <= y <= 10000):
        return None
    return x, y


def find_all_intersections(
    selected_lines: List[List[List[np.int32]]]
) -> List[Tuple[int, int, CornerOrientation]]:

    corners = []
    for line1 in selected_lines:
        for line2 in selected_lines:
            if line1 > line2:
                if get_line_slope(line1[0]) != get_line_slope(line2[0]):
                    intersection = get_intersection(line1, line2)
                    if intersection is not None:
                        corner_orientation = get_corner_orientation(line1[0], line2[0], intersection)
                        if corner_orientation is not None:
                            corners.append((*intersection, corner_orientation.value))
    return corners


def cluster_lines(slopes: np.ndarray, heights: np.ndarray) -> np.ndarray:
    line_functions = np.array(list(zip(slopes, heights)))
    clustering = DBSCAN(eps=0.3, min_samples=1).fit(line_functions)
    return clustering.labels_


def get_corner_orientation(
    line1: List[np.int32],
    line2: List[np.int32],
    intersection: Tuple[int, int]
) -> Optional[CornerOrientation]:
    # print(line1)
    # print(line2)
    center1 = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
    center2 = ((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2)

    if center1[0] <= intersection[0] <= center2[0] or center2[0] <= intersection[0] <= center1[0]:  # top or bottom
        if intersection[1] >= center1[1] and intersection[1] >= center2[1]:
            return CornerOrientation.BOTTOM
        elif intersection[1] <= center1[1] and intersection[1] <= center2[1]:
            return CornerOrientation.TOP
        else:
            return None
    elif center1[1] <= intersection[1] <= center2[1] or center2[1] <= intersection[1] <= center1[1]:  # left or right
        if intersection[0] >= center1[0] and intersection[0] >= center2[0]:
            return CornerOrientation.RIGHT
        elif intersection[0] <= center1[0] and intersection[0] <= center2[0]:
            return CornerOrientation.LEFT
        else:
            return None
    else:
        return None


def get_cloud_alpha_bis(
        map_image: np.ndarray,
        ksize: int = 7,
        sigma: int = 15,
        is_background: bool = False) -> Tuple[np.ndarray, float]:

    template = image_utils.get_cloud_template()

    # print("alpha hist", np.histogram(alpha_template, bins=[-4, -3, -2, -1, 0, 1]))

    result_set = []

    # TODO Optimisation possibilities:
    # - not go linear, look for the point
    # - start in the middle and go out, based on gradient
    # - stop when it goes down again
    # - change spacing
    # - move range based on estiamted scaling
    # - add half-templates
    for int_scale in range(300, 400, 5):
        scale = int_scale / 100.0
        # new_dim = (int(map_image.shape[1] * scale), int(map_image.shape[0] * scale))
        new_dim = (int(template.shape[1] / scale), int(template.shape[0] / scale))
        resized_template = cv2.resize(template[:, :, 0:3], new_dim, interpolation=cv2.INTER_AREA)
        if map_image.shape[0] < resized_template.shape[0] or map_image.shape[1] < resized_template.shape[1]:
            break

        hh, ww = resized_template.shape[:2]
        img_alpha = np.ones((map_image.shape[0:2]))

        alpha_template = cv2.resize(template[:, :, 3], new_dim, interpolation=cv2.INTER_AREA)
        alpha = cv2.merge([alpha_template, alpha_template, alpha_template])

        corr_img = cv2.matchTemplate(map_image, resized_template, cv2.TM_CCOEFF_NORMED, mask=alpha)
        correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)
        threshold = 83

        maximum = np.max(blur)

        max_val = np.max(blur)
        rad = int(math.sqrt(hh * hh + ww * ww) / 4)

        count = 0

        while max_val > threshold:

            # find max value of correlation image
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

            if max_val > threshold:
                # draw match on copy of input
                img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] = \
                    img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] - alpha_template / 255.0

                # write black circle at max_loc in corr_img
                cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)
                count += 1

            else:
                break
        # cv2.imwrite("./alpha_before.png", (img_alpha * 255 / np.max(img_alpha)).astype("uint8"))
        # hist = np.histogram(img_alpha, bins=[-4, -3, -2, -1, 0, 1])

        img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8) * 255
        alpha_area = np.sum(img_alpha_c == 0) / (img_alpha_c.shape[0] * img_alpha_c.shape[1])
        result_set.append([scale, maximum, maximum / scale, count, alpha_area, img_alpha_c])
        if DEBUG:
            cv2.imwrite("./cloud_matching/%s_%.2f.png" % (str(is_background), scale), img_alpha_c)

    if False:  # not is_background:
        selected_scaling = sorted(result_set, key=lambda x: abs(1.21 - x[0]))[0]
    else:
        selected_scaling = sorted(result_set, key=lambda x: -x[3])[0]

    print("selected_scaling", selected_scaling[:5])
    resized_original = cv2.resize(
        selected_scaling[5], (map_image.shape[1], map_image.shape[0]), interpolation=cv2.INTER_AREA)
    return resized_original, selected_scaling[0]


def get_cloud_alpha_ter(
        map_image: np.ndarray,
        ksize: int = 7,
        sigma: int = 15,
        is_background: bool = False) -> Tuple[np.ndarray, float]:

    template = image_utils.get_cloud_template()

    raw_scale = get_scale(map_image)
    scale = 1.0 / (raw_scale / template.shape[0])

    resized_template, resized_template_alpha, resized_template_alpha_layer = resize_template(template, scale)

    blur = get_template_matching(map_image, resized_template, resized_template_alpha, ksize, sigma)

    if DEBUG:
        maximum = np.max(blur)

    img_alpha = np.ones((map_image.shape[0:2]))

    img_alpha = match_full_clouds(blur, img_alpha, resized_template, resized_template_alpha_layer)

    cloud_less_map_image = cv2.bitwise_and(map_image, map_image, mask=img_alpha.astype(np.uint8))

    if DEBUG:
        cv2.imwrite("./cloudless_map/cloud_less_map%s.png" % str(datetime.now()), cloud_less_map_image)

    img_alpha = match_border_clouds(cloud_less_map_image, img_alpha, resized_template, resized_template_alpha_layer)

    if DEBUG:
        cv2.imwrite("./cloudless_map/border_less_map%s.png" % str(datetime.now()), img_alpha.clip(0, 1) * 255)

    img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8) * 255

    if DEBUG:
        alpha_area = np.sum(img_alpha_c == 0) / (img_alpha_c.shape[0] * img_alpha_c.shape[1])
        print("result set", scale, maximum, maximum / scale, alpha_area)
        cv2.imwrite("./cloud_matching/%s_%.2f.png" % (str(is_background), scale), img_alpha_c)

    return img_alpha_c, scale


def match_full_clouds(
        blur: np.ndarray,
        img_alpha: np.ndarray,
        template: np.ndarray,
        template_alpha: np.ndarray) -> np.ndarray:

    threshold = 70
    max_val = np.max(blur)
    hh, ww = template_alpha.shape[:2]
    rad = int(math.sqrt(hh * hh + ww * ww) / 4)

    while max_val > threshold:

        # find max value of correlation image
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

        if max_val > threshold:
            # draw match on copy of input
            img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] = \
                img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] - template_alpha / 255.0

            # write black circle at max_loc in corr_img
            cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)

        else:
            break

    return img_alpha


def pad_matching_template(
        blur: np.ndarray,
        img_alpha: np.ndarray,
        partial_template: np.ndarray,
        partial_template_alpha: np.ndarray) -> np.ndarray:
    border_blur = get_template_matching(blur, partial_template, partial_template_alpha, 7, 15)

    missing_height = img_alpha.shape[0] - border_blur.shape[0]
    top = int(missing_height / 2)
    bottom = missing_height - top
    missing_width = img_alpha.shape[1] - border_blur.shape[1]
    left = int(missing_width / 2)
    right = missing_width - left

    padded_blur = cv2.copyMakeBorder(border_blur, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    if DEBUG:
        cv2.imwrite("./cloudless_map/padded_blur_%s.png" % str(datetime.now()), padded_blur)
    return padded_blur


def match_border_clouds(
        map_image: np.ndarray,
        img_alpha: np.ndarray,
        template: np.ndarray,
        template_alpha: np.ndarray) -> np.ndarray:

    full_hh, full_ww = template_alpha.shape[:2]
    rad = int(math.sqrt(full_hh * full_hh + full_ww * full_ww) / 4)

    border_threshold = 30

    border_width = max(full_hh, full_ww)

    img_shape = img_alpha.shape

    if img_shape[0] > img_shape[1]:

        left_template_alpha = template_alpha[:, :int(full_ww / 2)]
        right_template_alpha = template_alpha[:, int(full_ww / 2):]

        left_template = template[:, :int(full_ww / 2)]
        right_template = template[:, int(full_ww / 2):]

        blur_left = pad_matching_template(map_image, img_alpha, left_template, left_template_alpha)
        blur_right = pad_matching_template(map_image, img_alpha, right_template, right_template_alpha)

        borders = [
            ((0, 0), blur_right[:, :border_width], right_template_alpha),
            ((0, img_shape[1] - border_width), blur_left[:, -border_width:], left_template_alpha)]
    else:
        print("top-bottom borders")

        top_template = template[:int(full_hh / 2), :]
        bottom_template = template[int(full_hh / 2):, :]

        top_template_alpha = template_alpha[:int(full_hh / 2), :]
        bottom_template_alpha = template_alpha[int(full_hh / 2):, :]

        blur_top = pad_matching_template(map_image, img_alpha, top_template, top_template_alpha)
        blur_bottom = pad_matching_template(map_image, img_alpha, bottom_template, bottom_template_alpha)

        borders = [
            ((0, 0), blur_bottom[:border_width, :], bottom_template_alpha),
            ((img_shape[0] - border_width, 0), blur_top[-border_width:, :], top_template_alpha)]

    for padding, border_image, partial_template_alpha in borders:

        hh, ww = partial_template_alpha.shape

        max_val = np.max(border_image)

        while max_val > border_threshold:

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(border_image)

            padded_max_loc = max_loc[0] + padding[0] - int(hh / 2), max_loc[1] + padding[1] - int(ww / 2)  # y, x

            if max_val > border_threshold:

                background = img_alpha[
                    padded_max_loc[1]: padded_max_loc[1] + hh,
                    padded_max_loc[0]: padded_max_loc[0] + ww]

                img_alpha[
                    padded_max_loc[1]: padded_max_loc[1] + hh,
                    padded_max_loc[0]: padded_max_loc[0] + ww] = \
                    background - partial_template_alpha[:background.shape[0], :background.shape[1]]

                cv2.circle(border_image, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)
            else:
                break

    return img_alpha


def get_template_matching(
        map_image: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        ksize: int,
        sigma: int) -> np.ndarray:
    corr_img = cv2.matchTemplate(map_image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)
    return blur


def resize_template(template: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_dim = (int(template.shape[1] / scale), int(template.shape[0] / scale))
    resized_template = cv2.resize(template[:, :, 0:3], new_dim, interpolation=cv2.INTER_AREA)

    resized_template_alpha_layer = cv2.resize(template[:, :, 3], new_dim, interpolation=cv2.INTER_AREA)
    resized_template_alpha = cv2.merge(
        [resized_template_alpha_layer, resized_template_alpha_layer, resized_template_alpha_layer])

    return resized_template, resized_template_alpha, resized_template_alpha_layer


def get_interpoint_space(centers: List[np.ndarray]) -> int:
    center_pd = pd.DataFrame(np.reshape(centers, (len(centers), 2)), columns=['x', 'y'])
    grouped_centers = center_pd.groupby(by="x").agg(list)
    values = [c[i] - c[i + 1] for i, c in grouped_centers["y"].iteritems() for i in range(len(c) - 1)]
    num, bins = np.histogram(values, bins=int((np.max(values) - np.min(values)) / 5) + 1)
    max_index = np.argmax(num)
    min_bin = bins[max_index]
    max_bin = bins[max_index + 1]

    return int(np.median([v for v in values if min_bin <= v <= max_bin]))


def match_template(template: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    LIGHT_COLOUR = (0, 0, 0)
    DARK_COLOUR = (10, 10, 10)

    background = 255 - cv2.inRange(template, LIGHT_COLOUR, DARK_COLOUR)
    contours, _ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contour = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x: -x[1])[0]
    mask = cv2.drawContours(np.zeros_like(background), [selected_contour[0]], -1, (255, 255, 255), thickness=cv2.FILLED)

    shape = template.shape

    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    frame = cv2.copyMakeBorder(
        masked_edges,
        int(shape[0] / 2),
        int(shape[0] / 2) - (1 if shape[0] % 2 == 0 else 0),
        int(shape[1] / 2),
        int(shape[1] / 2) - (1 if shape[1] % 2 == 0 else 0),
        cv2.BORDER_CONSTANT, 0)

    corr_img = cv2.matchTemplate(frame, masked_edges, cv2.TM_CCOEFF, mask=mask)

    correlation_raw = (255 * (corr_img - np.min(corr_img)) / (np.max(corr_img) - np.min(corr_img))).astype(np.uint8)

    masked_correlation_raw = cv2.bitwise_and(correlation_raw, correlation_raw, mask=mask)

    (_, thresh) = cv2.threshold(masked_correlation_raw, 30, 255, cv2.THRESH_BINARY)

    return masked_correlation_raw, mask, thresh


def get_scale(template: np.ndarray) -> float:
    edges = image_processing_utils.get_one_color_edges(template)
    # template_edge = get_one_color_edges(template)
    if DEBUG:
        cv2.imwrite("./self_edges.png", edges)

    correlation_raw, mask, thresh = match_template(template, edges)

    if DEBUG:
        cv2.imwrite("./self_matching.png", correlation_raw)
        cv2.imwrite("./self_mask.png", mask)
        cv2.imwrite("./self_masked_matching.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = [np.reshape(np.mean(c, axis=0).astype(np.int32), (1, 1, 2)) for c in contours]

    if DEBUG:
        contour_mask = cv2.drawContours(np.zeros((1080, 2312, 3)), centers, -1, (255, 255, 255))
        cv2.imwrite("./self_contour.png", contour_mask)

    return get_interpoint_space(centers)
