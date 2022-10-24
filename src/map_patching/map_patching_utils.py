import os
import cv2
import math
import discord
import numpy as np
import pandas as pd

from typing import List
from typing import Union
from typing import Tuple
from typing import Optional

from common import image_utils
from common.image_operation import ImageOp
from common import image_processing_utils
from database_interaction.database_client import DatabaseClient

from map_patching.map_patching_errors import MapPatchingErrors

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


""" def getLines(image, minLineLength=500, channel: discord.TextChannel = None, filename: str = None):
    processed = image.copy()
    edges = image_processing_utils.get_one_color_edges(image, channel=channel)
    lines = cv2.HoughLinesP(image=edges, rho=0.1, theta=np.pi / 180, threshold=150, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=100)

    for i in range(lines.shape[0]):
        cv2.line(processed, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                 cv2.LINE_AA)
        if DEBUG and filename is not None and channel is not None:
            image_utils.save_image(image, channel.name, filename, ImageOp.HOUGH_LINES)
    return processed """


def select_contours(image: np.ndarray) -> Optional[np.ndarray]:
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_half_perimeter = 0
    output = None

    for contour_ in contours:

        x, y, w, h = cv2.boundingRect(contour_)

        if w > int(image.shape[0] / 7) and h > int(image.shape[1] / 7):
            if w + h > max_half_perimeter:
                max_half_perimeter = w + h
                output = contour_

    return output


def draw_contour(
        image: np.ndarray,  # image: edges
        contour_: np.ndarray,
        channel_name: str = None,
        epsilon: float = None,
        epsilon_factor: float = 0.005,
        filename: str = None) -> Optional[np.ndarray]:
    mask = np.zeros(image.shape[:2], np.uint8)

    convex_hull = cv2.convexHull(contour_, returnPoints=True)

    if epsilon is None:
        epsilon = epsilon_factor * cv2.arcLength(convex_hull, True)

    approx = cv2.approxPolyDP(convex_hull, epsilon, True)

    while len(approx) > 12:
        if DEBUG:
            print("approx too long: ", len(approx))
        epsilon /= 0.9
        approx = cv2.approxPolyDP(contour_, epsilon, True)

    if len(approx) < 4:
        return None

    if DEBUG:
        print("simplified contour has", len(approx), "points with epsilon=", epsilon)

    if DEBUG and filename is not None and channel_name is not None:
        processed = image.copy()
        cv2.drawContours(processed, [approx], 0, (255, 255, 255), 3)
        cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

        image_utils.save_image(processed, channel_name, filename, ImageOp.OVERLAY)
        image_utils.save_image(mask, channel_name, filename, ImageOp.MASK)

    if DEBUG:
        print("mask area: %d M" % (mask.sum() / 10e6))

    return approx


def compute_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Returns the vertices of the approximate diamond: up, down, left, right
    """
    c = [i[0] for i in vertices]

    up_y = np.min(c, axis=0)[1]
    up = sorted([pair for pair in c if pair[1] == up_y], key=lambda p: p[0])[0]

    down_y = np.max(c, axis=0)[1]
    down = sorted([pair for pair in c if pair[1] == down_y], key=lambda p: p[0])[0]

    left_x = np.min(c, axis=0)[0]
    left = sorted([pair for pair in c if pair[0] == left_x], key=lambda p: p[1])[0]

    right_x = np.max(c, axis=0)[0]
    right = sorted([pair for pair in c if pair[0] == right_x], key=lambda p: p[1])[0]

    if DEBUG:
        print("vertices:\n", np.array([up, down, left, right]))
    return np.array([up, down, left, right])


def get_transformation_dimensions(
        vertices: np.ndarray,
        is_vertical: bool,
        reference_position: Tuple[np.int64, np.int64],
        reference_size: Tuple[np.int64, np.int64]) -> Tuple[Tuple[int, int], float, np.ndarray]:

    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]

    if is_vertical:  # d_w / d_h < ratio:  # smaller width or larger height proportionally
        # missing width; scale on height
        scale_factor = d_h / reference_size[1]  # then divide by scale factor to find right size
        scaled_padding = (
            int(reference_position[0] - (vertices[0][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[0][1] - 50) / scale_factor)
        )
    else:
        # missing height; scale on width
        scale_factor = d_w / reference_size[0]
        scaled_padding = (
            int(reference_position[0] - (vertices[2][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[2][1] - 50) / scale_factor)
        )
    
    padded_and_scaled_vertices = [[int(a) for a in b] for b in vertices / scale_factor + scaled_padding]

    if DEBUG:
        print("scale factor:", scale_factor)
        print("padding:", scaled_padding)
        print("cropped and scaled vertices:\n", padded_and_scaled_vertices)
    return scaled_padding, scale_factor, padded_and_scaled_vertices


def crop_padding_(
        image: np.ndarray,
        padding: Tuple[int, int],
        filename: str,
        channel_name: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    if padding[0] < 0:
        image = image[:, -padding[0]:]
        padding = (0, padding[1])
    if padding[1] < 0:
        image = image[-padding[1]:, :]
        padding = (padding[0], 0)
    if DEBUG and filename is not None and channel_name is not None:
        image_utils.save_image(image, channel_name, filename, ImageOp.PADDING)
    return image, padding


def crop_padding(image: np.ndarray, padding: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if padding[0] < 0:
        image = image[-padding[0]:, :]
        padding = (0, padding[1])
    if padding[1] < 0:
        image = image[:, -padding[1]:]
        padding = (padding[0], 0)
    return image, padding


def scale_image(
        image: np.ndarray,
        channel_name: str,
        scale_factor: float,
        filename: str = None) -> np.ndarray:
    new_dim = (int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor))
    scaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    if DEBUG and filename is not None and channel_name is not None:
        image_utils.save_image(scaled_image, channel_name, filename, ImageOp.SCALE)
    return scaled_image


def get_references(
        optional_vertices: List[Optional[np.ndarray]]) -> Tuple[Tuple[np.int64, np.int64], Tuple[np.int64, np.int64]]:
    vertices = [v for v in optional_vertices if v is not None]

    reference_position: Tuple[np.int64, np.int64] = (np.int64(0), np.int64(0))
    reference_size: Tuple[np.int64, np.int64] = (np.int64(0), np.int64(0))

    # 1.6 approximate ratio
    top_bottom_ref = ((vertices[0][1] - vertices[0][0])[1] / (vertices[0][3] - vertices[0][2])[0]) < 1.6
    # top_bottom_ref = True

    # find reference position
    for vertex_i in vertices:
        dx = vertex_i[0][0] - vertex_i[2][0]
        dy = vertex_i[2][1] - vertex_i[0][1]
        if dx > reference_position[0]:
            reference_position = (dx, reference_position[1])
        if dy > reference_position[1]:
            reference_position = (reference_position[0], dy)

    # find reference size
    for vertex_i in vertices:
        if top_bottom_ref:
            d_h = vertex_i[1][1] - vertex_i[0][1]
            d_w = max(vertex_i[3][0] - vertex_i[0][0], vertex_i[0][0] - vertex_i[2][0]) * 2
            if d_h > reference_size[1]:
                reference_size = (d_w, d_h)
        else:
            d_h = max(vertex_i[2][1] - vertex_i[0][1], vertex_i[1][1] - vertex_i[2][1]) * 2
            d_w = vertex_i[3][0] - vertex_i[2][0]
            if d_w > reference_size[0]:
                reference_size = (d_w, d_h)

    if DEBUG:
        print("top-bottom ref: ", top_bottom_ref,
              (vertices[0][1] - vertices[0][0])[1] / (vertices[0][3] - vertices[0][2])[0])
        print("reference position: ", reference_position)
        print("reference size: ", reference_size)
    return reference_position, reference_size


def get_lines(vertices_i: List[np.ndarray]) -> pd.DataFrame:

    delta_list: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = [
        (
            j,
            vertices_i[j],
            vertices_i[(j + 1) % len(vertices_i)],
            vertices_i[(j + 1) % len(vertices_i)] - vertices_i[j]
        ) for j in range(len(vertices_i))]

    lines = [
        (
            j,
            np.multiply(delta, delta).sum(),
            (delta[1] / delta[0]) if delta[0] != 0 else np.sign(delta[0]) * np.sign(delta[1]) * 100000,
            np.sign(delta[0]), np.sign(delta[1]), p0.tolist(), p1.tolist()
        ) for j, p0, p1, delta in delta_list]

    if DEBUG:
        print("all lines\n", pd.DataFrame(lines))
    # very permissive: slopes of selected lines are within 0.55 < abs(slope) < 0.65
    selected_lines = sorted([(j, length, slope, sign_x, sign_y, p0, p1)
                             for (j, length, slope, sign_x, sign_y, p0, p1)
                             in lines if 10 > abs(slope) > 0.2], key=lambda x: -x[1])

    lines_df = pd.DataFrame(selected_lines)

    filtered_lines = lines_df.groupby([3, 4]).apply(lambda group: group.iloc[0])

    if DEBUG:
        print("filtered lines\n", filtered_lines)

    return filtered_lines


def get_intersection(lines: pd.DataFrame) -> Optional[List[Tuple[int, int]]]:
    intersections = []
    if len(lines) == 4:

        y0 = np.stack(lines[5].to_numpy())[:, 1]
        y1 = np.stack(lines[6].to_numpy())[:, 1]
        lines[7] = y0 + y1
        lines.sort_values(7, inplace=True)

        for i in range(2):
            points = lines[i * 2: i * 2 + 2]
            point_0 = np.stack(points[5].to_numpy())

            x1, y1 = point_0[0]
            p1 = points[2].iloc[0]
            h1 = y1 - p1 * x1

            x2, y2 = point_0[1]
            p2 = points[2].iloc[1]
            h2 = y2 - p2 * x2

            x = int((h2 - h1) / (p1 - p2))
            y = int(p1 * x + h1)
            intersections.append((x, y))

        x0 = np.stack(lines[5].to_numpy())[:, 0]
        x1 = np.stack(lines[6].to_numpy())[:, 0]
        lines[7] = x0 + x1
        lines.sort_values(7, inplace=True)

        for i in range(2):
            points = lines[i * 2: i * 2 + 2]
            point_0 = np.stack(points[5].to_numpy())

            x1, y1 = point_0[0]
            p1 = points[2].iloc[0]
            h1 = y1 - p1 * x1

            x2, y2 = point_0[1]
            p2 = points[2].iloc[1]
            h2 = y2 - p2 * x2

            x = int((h2 - h1) / (p1 - p2))
            y = int(p1 * x + h1)
            intersections.append((x, y))

    elif len(lines) == 2:
        return None
    if DEBUG:
        print(intersections)
    return intersections


def get_orientation(vertices: np.ndarray) -> bool:
    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]
    print("orientation", d_w / d_h < 2, d_w / d_h)
    return True  # TODO: what the heck


def process_raw_map(
        image: np.ndarray,
        filename_i: str,
        channel_name: str,
        kernel_size: int = 5,
        sigma: int = 5) -> Tuple[MapPatchingErrors, Union[str, Tuple[np.ndarray, bool]]]:
    print("map_patching_utils", filename_i)
    if image is None:
        if DEBUG:
            print("image not found:", filename_i)
        return MapPatchingErrors.MISSING_MAP_INPUT, filename_i

    edges = image_processing_utils.get_three_color_edges(image, channel_name, filename_i, border=50)
    blur = cv2.GaussianBlur(edges, (kernel_size, kernel_size), sigmaX=sigma)
    contour = select_contours(blur)

    if contour is None:
        return MapPatchingErrors.MAP_NOT_RECOGNIZED, filename_i

    polygon = draw_contour(edges, contour, channel_name, filename=filename_i)

    if polygon is None:
        return MapPatchingErrors.MAP_NOT_RECOGNIZED, filename_i

    vertices_i = compute_vertices(polygon)
    is_vertical_i = get_orientation(vertices_i)
    lines = get_lines([p[0] for p in polygon])

    if lines is None or len(lines) != 4:
        return MapPatchingErrors.MAP_NOT_RECOGNIZED, filename_i

    intersections = get_intersection(lines)

    if DEBUG:
        print()
        print("intersections:\n", intersections)
        if intersections is not None:
            print(len(intersections), len(intersections[0]), type(intersections))
        print(vertices_i.shape, type(vertices_i))

    if intersections is None or len(intersections) < 2:
        return MapPatchingErrors.MAP_NOT_RECOGNIZED, filename_i

    vertices_i = np.array(intersections)

    if DEBUG:
        debug_process_raw_map(vertices_i, edges, channel_name, filename_i)

    return MapPatchingErrors.SUCCESS, (vertices_i, is_vertical_i)


def debug_process_raw_map(
        vertices_i: np.ndarray,
        edges: np.ndarray,
        channel_name: str,
        filename_i: str) -> None:
    print("vertices after:\n", vertices_i)
    print_vertices = [vertices_i[0], vertices_i[3], vertices_i[1], vertices_i[2], vertices_i[0]]
    for i in range(len(print_vertices) - 1):
        cv2.line(edges, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
        cv2.putText(edges, "%s" % i, print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6,
                    (255, 255, 255), 3, cv2.LINE_AA)
    image_utils.save_image(edges, channel_name, filename_i, ImageOp.DEBUG_VERTICES)


def transform_image(
        image_i: np.ndarray,
        vertices_i: np.ndarray,
        is_vertical: bool,
        reference_position: Tuple[np.int64, np.int64],
        reference_size: Tuple[np.int64, np.int64],
        channel_name: str,
        filename_i: str,
        map_size: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float, np.ndarray]:
    transformation_dimensions = get_transformation_dimensions(
        vertices_i, is_vertical, reference_position, reference_size)
    padding, scale_factor, padded_and_scaled = transformation_dimensions

    scaled_image = scale_image(image_i, channel_name, scale_factor, filename_i)
    cropped_image, padding = crop_padding_(scaled_image, padding, filename_i, channel_name)

    scale = int((reference_size[1] + 100) / math.sqrt(int(map_size)))
    oppacity = remove_clouds(cropped_image, ksize=7, sigma=15, template_height=scale)

    flipped_padding = padding[1], padding[0]

    return cropped_image, oppacity, flipped_padding, scale_factor, padded_and_scaled


def patch_output(
        patch_work: np.ndarray,
        scaled_padding: Tuple[int, int],
        oppacity: np.ndarray,
        size: Tuple[int, int],
        bit: np.ndarray,
        i: int) -> np.ndarray:
    background = patch_work[
        scaled_padding[0]:scaled_padding[0] + size[0],
        scaled_padding[1]:scaled_padding[1] + size[1], :]
    if DEBUG:
        print(background.shape, oppacity.shape)
        print("scaled_padding", scaled_padding)
        print("size", size)
        print("bit size", bit.shape)
    if i == 0:  # background
        result = bit
    else:
        result = cv2.multiply(background, 1 - oppacity) + cv2.multiply(bit, oppacity)
    patch_work[
        scaled_padding[0]: scaled_padding[0] + size[0],
        scaled_padding[1]: scaled_padding[1] + size[1], :] = result
    return patch_work


def crop_output(
        image: np.ndarray,
        position: Tuple[np.int64, np.int64],
        size: Tuple[np.int64, np.int64]) -> np.ndarray:

    return image[
        (position[1] - 20).clip(0): (position[1] + size[1] + 20).clip(0, image.shape[1]),
        (position[0] - size[0] - 20).clip(0): (position[0] + size[0] + 20).clip(0, image.shape[0])]


def patch_partial_maps(
        channel_name: str,
        images: list,
        files: list,
        map_size: str,
        patch_uuid: str,
        database_client: DatabaseClient = None,
        message: discord.Message = None) -> Tuple[str, str, list]:

    patching_errors = []
    vertices: List[Optional[np.ndarray]] = []
    vertical: List[Optional[bool]] = []
    bits = []
    transparency_masks = []
    scaled_paddings = []
    sizes: List[Tuple[int, int]] = []
    scaled_vertices = []

    for i, image_i in enumerate(images):
        filename_i = files[i]
        status, processed_raw_map = process_raw_map(image_i, filename_i, channel_name)

        if status == MapPatchingErrors.SUCCESS and not isinstance(processed_raw_map, str):
            vertices_i, is_vertical_i = processed_raw_map
            vertices.append(vertices_i)
            vertical.append(is_vertical_i)
        else:
            if database_client is not None:
                database_client.update_patching_process_input_status(patch_uuid, filename_i, status.name)
            patching_errors.append((status, processed_raw_map))
            vertices.append(None)
            vertical.append(None)

    reference_position, reference_size = get_references(vertices)

    for i in range(len(images)):
        filename_i = files[i]
        image_i = images[i]
        loaded_vertices_i = vertices[i]
        loaded_is_vertical_i = vertical[i]
        if loaded_vertices_i is not None and loaded_is_vertical_i is not None:
            transformation = transform_image(
                image_i, loaded_vertices_i, loaded_is_vertical_i, reference_position, reference_size, channel_name,
                filename_i, map_size)
            scaled_image, oppacity, padding, scale_factor, padded_and_scaled = transformation

            transparency_masks.append(oppacity)
            bits.append(scaled_image)
            scaled_paddings.append(padding)
            scaled_vertices.append(padded_and_scaled)
            sizes.append(scaled_image.shape[0:2])  # type: ignore

    output_size = np.max(np.array(scaled_paddings) + np.array(sizes), axis=0)

    patch_work = np.zeros([output_size[0], output_size[1], 3], np.uint8)

    for i in range(len(bits)):
        bit = bits[i]
        scaled_padding = scaled_paddings[i]
        size = sizes[i]
        oppacity = transparency_masks[i]
        patch_work = patch_output(patch_work, scaled_padding, oppacity, size, bit, i)

    cropped_patch_work = crop_output(patch_work, reference_position, reference_size)

    if DEBUG:
        debug_patch_partial_maps(scaled_vertices, patch_work, channel_name)

    if message is not None and database_client is not None:
        filename = database_client.add_resource(
            message.guild.id, message.channel.id, message.id, message.author.id, message.author.name,
            ImageOp.MAP_PATCHING_OUTPUT)
    else:
        filename = 'map_patching_debug'
    file_path = image_utils.save_image(cropped_patch_work, channel_name, filename, ImageOp.MAP_PATCHING_OUTPUT)
    return file_path, filename, patching_errors


def debug_patch_partial_maps(
        scaled_vertices: List[np.ndarray],
        patch_work: np.ndarray,
        channel_name: str) -> str:
    print(scaled_vertices)
    vertex_lines = np.zeros_like(patch_work)
    for j, vertices in enumerate(scaled_vertices):
        print_vertices = [vertices[0], vertices[3], vertices[1], vertices[2], vertices[0]]
        for i in range(len(print_vertices) - 1):
            cv2.line(vertex_lines, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
            cv2.putText(vertex_lines, "%s, %s" % (j, i), print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6,
                        (255, 255, 255), 3, cv2.LINE_AA)

    return image_utils.save_image(vertex_lines, channel_name, 'map_patching_debut', ImageOp.DEBUG_VERTICES)


def is_map_patching_request(
        message: discord.Message,
        attachment: discord.Attachment,
        filename: str) -> bool:
    return "🖼️" in [r.emoji for r in message.reactions]


def remove_clouds(
        img: np.ndarray,
        ksize: int = 31,
        sigma: int = 25,
        template_height: int = 1) -> np.ndarray:
    print("img shape", img.shape)
    original_template = image_utils.get_cloud_template()

    template_shape = original_template.shape

    scale = template_height * math.sqrt(2) / template_shape[1]
    template_width = int(template_shape[0] * scale)
    template_height = int(template_shape[1] * scale)

    new_dim = (template_height, template_width)
    scaled_template = cv2.resize(original_template, new_dim, interpolation=cv2.INTER_AREA)

    img_border_width = max(template_width, template_height)

    result = cv2.copyMakeBorder(
        img, img_border_width, img_border_width, img_border_width, img_border_width, cv2.BORDER_CONSTANT, value=0)

    img_alpha = np.ones((result.shape[0:2]))

    left_template = scaled_template[:, :int(template_height / 2), :]
    right_template = scaled_template[:, int(template_height / 2):, :]

    for i, template in enumerate([scaled_template, left_template, right_template]):  # , ]:
        img_alpha = match_cloud(result, template, img_alpha, ksize, sigma, 50 + img_border_width, i)

    img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8)
    mask = cv2.merge([img_alpha_c, img_alpha_c, img_alpha_c]).astype(np.uint8)
    cropped_mask = mask[img_border_width:-img_border_width, img_border_width:-img_border_width, :]
    return cropped_mask


def match_cloud(
        result: np.ndarray,
        template: np.ndarray,
        img_alpha: np.ndarray,
        ksize: int,
        sigma: int,
        border_width: int,
        i: int) -> np.ndarray:
    hh, ww = template.shape[:2]

    # extract pawn base image and alpha channel and make alpha 3 channels
    pawn = template[:, :, 0:3]
    alpha_template = template[:, :, 3]
    alpha = cv2.merge([alpha_template, alpha_template, alpha_template])

    # do masked template matching and save correlation image
    corr_img = cv2.matchTemplate(result, pawn, cv2.TM_CCOEFF_NORMED, mask=alpha)
    correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)
    corr_shape = corr_img.shape

    blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)

    threshold = {
        0: 80,
        1: 110,
        2: 110
    }
    border_threshold = {
        0: 60,
        1: 40,
        2: 40
    }

    max_val = np.max(blur)
    rad = int(math.sqrt(hh * hh + ww * ww) / 4)

    while max_val > threshold[i]:

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

        if max_val > threshold[i]:
            img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] = \
                img_alpha[max_loc[1]: max_loc[1] + hh, max_loc[0]: max_loc[0] + ww] - alpha_template / 255.0

            cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)

        else:
            break

    if corr_shape[0] > corr_shape[1]:
        borders = [((0, 0), blur[:, :border_width]), ((0, corr_shape[1] - border_width), blur[:, -border_width:])]
    else:
        borders = [((0, 0), blur[:border_width, :]), ((corr_shape[0] - border_width, 0), blur[-border_width:, :])]
        # ((loc[0] < width) or ((shape[1] - loc[0]) < width))

    for padding, border_image in borders:

        max_val = np.max(border_image)

        while max_val > border_threshold[i]:

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(border_image)

            padded_max_loc = max_loc[0] + padding[1], max_loc[1] + padding[0]

            if max_val > border_threshold[i]:
                img_alpha[padded_max_loc[1]: padded_max_loc[1] + hh, padded_max_loc[0]: padded_max_loc[0] + ww] = \
                    img_alpha[padded_max_loc[1]: padded_max_loc[1] + hh, padded_max_loc[0]: padded_max_loc[0] + ww] \
                    - alpha_template / 255.0

                cv2.circle(border_image, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)
            else:
                break

    return img_alpha
