import os
import cv2
import math
import discord
import asyncio
import numpy as np
import pandas as pd

from common import image_utils
from common.image_utils import ImageOp
from common import image_processing_utils
from database_interaction import database_client

import nest_asyncio
nest_asyncio.apply()

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


def getLines(image, minLineLength=500, channel=None, filename=None):
    processed = image.copy()
    edges = image_processing_utils.get_one_color_edges(image, channel=channel)
    lines = cv2.HoughLinesP(image=edges, rho=0.1, theta=np.pi/180, threshold=150, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=100)

    for i in range(lines.shape[0]):
        cv2.line(processed, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                 cv2.LINE_AA)
        if filename:
            image_utils.save_image(image, channel.name, filename, ImageOp.HOUGH_LINES)
    return processed


def select_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    max_half_perimeter = 0
    output = None

    for contour_ in contours:

        x, y, w, h = cv2.boundingRect(contour_)

        if w > int(image.shape[0] / 7) and h > int(image.shape[1] / 7):
            if w + h > max_half_perimeter:
                max_half_perimeter = w + h
                output = contour_

    return output


# image: edges
def draw_contour(image, contour_, channel_name=None, epsilon=None, epsilonFactor=0.005, filename=None):
    mask = np.zeros(image.shape[:2], np.uint8)

    convex_hull = cv2.convexHull(contour_, returnPoints=True)

    if epsilon is None:
        epsilon = epsilonFactor * cv2.arcLength(convex_hull, True)

    approx = cv2.approxPolyDP(convex_hull, epsilon, True)

    while(len(approx) > 12):
        if DEBUG:
            print("approx too long: ", len(approx))
        epsilon /= 0.9
        approx = cv2.approxPolyDP(contour_, epsilon, True)

    if len(approx) < 4:
        return

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


def compute_vertices(c):
    """
    Returns the vertices of the approximate diamond: up, down, left, right
    """
    c = [i[0] for i in c]

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


def get_transformation_dimensions(vertices, is_vertical, reference_position, reference_size):

    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]

    if is_vertical:  # d_w / d_h < ratio:  # smaller width or larger height proportionally
        # missing width; scale on height
        scale_factor = d_h / reference_size[1]  # then divide by scale factor to find right size
        scaled_padding = [
            int(reference_position[0] - (vertices[0][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[0][1] - 50) / scale_factor)
        ]
    else:
        # missing height; scale on width
        scale_factor = d_w / reference_size[0]
        scaled_padding = [
            int(reference_position[0] - (vertices[2][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[2][1] - 50) / scale_factor)
        ]
    padded_and_scaled_vertices = vertices / scale_factor + scaled_padding
    padded_and_scaled_vertices = [[int(a) for a in b] for b in padded_and_scaled_vertices]

    if DEBUG:
        print("scale factor:", scale_factor)
        print("padding:", scaled_padding)
        print("cropped and scaled vertices:\n", padded_and_scaled_vertices)
    return scaled_padding, scale_factor, padded_and_scaled_vertices


def crop_padding_(image, padding, filename, channel_name):
    if padding[0] < 0:
        image = image[:, -padding[0]:]
        padding[0] = 0
    if padding[1] < 0:
        image = image[-padding[1]:, :]
        padding[1] = 0
    if DEBUG and filename is not None and channel_name is not None:
        image_utils.save_image(image, channel_name, filename, ImageOp.PADDING)
    return image, padding


def crop_padding(image, padding):
    if padding[0] < 0:
        image = image[-padding[0]:, :]
        padding[0] = 0
    if padding[1] < 0:
        image = image[:, -padding[1]:]
        padding[1] = 0
    return image, padding


def scale_image(image, channel_name, scale_factor, filename=None):
    new_dim = (int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor))
    scaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    if DEBUG and filename is not None and channel_name is not None:
        image_utils.save_image(scaled_image, channel_name, filename, ImageOp.SCALE)
    return scaled_image


def get_references(vertices):
    reference_position = [0, 0]
    reference_size = [0, 0]

    # 1.6 approximate ratio
    top_bottom_ref = ((vertices[0][1] - vertices[0][0])[1] / (vertices[0][3] - vertices[0][2])[0]) < 1.6
    # top_bottom_ref = True

    # find reference position
    for vertex_i in vertices:
        dx = vertex_i[0][0] - vertex_i[2][0]
        dy = vertex_i[2][1] - vertex_i[0][1]
        if dx > reference_position[0]:
            reference_position[0] = dx
        if dy > reference_position[1]:
            reference_position[1] = dy

    # find reference size
    for vertex_i in vertices:
        if top_bottom_ref:
            d_h = vertex_i[1][1] - vertex_i[0][1]
            d_w = max(vertex_i[3][0] - vertex_i[0][0], vertex_i[0][0] - vertex_i[2][0]) * 2
            if d_h > reference_size[1]:
                reference_size = [d_w, d_h]
        else:
            d_h = max(vertex_i[2][1] - vertex_i[0][1], vertex_i[1][1] - vertex_i[2][1]) * 2
            d_w = vertex_i[3][0] - vertex_i[2][0]
            if d_w > reference_size[0]:
                reference_size = [d_w, d_h]

    if DEBUG:
        print("top-bottom ref: ", top_bottom_ref,
              (vertices[0][1] - vertices[0][0])[1] / (vertices[0][3] - vertices[0][2])[0])
        print("reference position: ", reference_position)
        print("reference size: ", reference_size)
    return reference_position, reference_size


def get_lines(vertices_i):

    delta = [(j, vertices_i[j],
              vertices_i[(j + 1) % len(vertices_i)],
              vertices_i[(j + 1) % len(vertices_i)] - vertices_i[j]) for j in range(len(vertices_i))]

    lines = [(j, (delta * delta).sum(),
             (delta[1] / delta[0]) if delta[0] != 0 else np.sign(delta[0]) * np.sign(delta[1]) * 100000,
             np.sign(delta[0]), np.sign(delta[1]), p0.tolist(), p1.tolist()) for j, p0, p1, delta in delta]

    if DEBUG:
        print("all lines\n", pd.DataFrame(lines))
    # very permissive: slopes of selected lines are within 0.55 < abs(slope) < 0.65
    selected_lines = sorted([(j, length, slope, sign_x, sign_y, p0, p1)
                            for (j, length, slope, sign_x, sign_y, p0, p1)
                            in lines if abs(slope) < 10 and abs(slope) > 0.2],  key=lambda x: -x[1])

    lines_df = pd.DataFrame(selected_lines)

    filtered_lines = lines_df.groupby([3, 4]).apply(lambda group: group.iloc[0])

    if DEBUG:
        print("filtered lines\n", filtered_lines)

    return filtered_lines


def get_intersection(lines):
    intersections = []
    if len(lines) == 4:

        y0 = np.stack(lines[5].to_numpy())[:, 1]
        y1 = np.stack(lines[6].to_numpy())[:, 1]
        lines[7] = y0 + y1
        lines.sort_values(7, inplace=True)

        for i in range(2):
            points = lines[i*2:i*2+2]
            point_0 = np.stack(points[5].to_numpy())

            x1, y1 = point_0[0]
            p1 = points[2].iloc[0]
            h1 = y1 - p1 * x1

            x2, y2 = point_0[1]
            p2 = points[2].iloc[1]
            h2 = y2 - p2 * x2

            x = int((h2 - h1) / (p1 - p2))
            y = int(p1 * x + h1)
            intersections.append([x, y])

        x0 = np.stack(lines[5].to_numpy())[:, 0]
        x1 = np.stack(lines[6].to_numpy())[:, 0]
        lines[7] = x0 + x1
        lines.sort_values(7, inplace=True)

        for i in range(2):
            points = lines[i*2:i*2+2]
            point_0 = np.stack(points[5].to_numpy())

            x1, y1 = point_0[0]
            p1 = points[2].iloc[0]
            h1 = y1 - p1 * x1

            x2, y2 = point_0[1]
            p2 = points[2].iloc[1]
            h2 = y2 - p2 * x2

            x = int((h2 - h1) / (p1 - p2))
            y = int(p1 * x + h1)
            intersections.append([x, y])

    elif len(lines) == 2:
        return
    if DEBUG:
        print(intersections)
    return intersections


def get_orientation(vertices):
    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]
    print("orientation", d_w / d_h < 2, d_w / d_h)
    return True


async def process_raw_map(
        filename_i, i, channel_name, map_size, database_client, message=None, kernel_size=5, sigma=5):
    print("map_patching_utils", filename_i)
    if i == 0:
        image = image_utils.get_background_template(map_size)
    else:
        image = await image_utils.load_image(database_client, channel_name, message, filename_i, ImageOp.INPUT)

    if image is None:
        if DEBUG:
            print("image not found:", filename_i)
        print("EXIT LOOP")
        return

    edges = image_processing_utils.get_three_color_edges(image, channel_name, filename_i, border=50)
    blur = cv2.GaussianBlur(edges, (kernel_size, kernel_size), sigmaX=sigma)
    contour = select_contours(blur)
    polygon = draw_contour(edges, contour, channel_name, filename=filename_i)
    vertices_i = compute_vertices(polygon)
    is_vertical_i = get_orientation(vertices_i)
    lines = get_lines([p[0] for p in polygon])

    if lines is None or len(lines) != 4:
        if DEBUG:
            print("lines not detected for file:", filename_i)
        return

    intersections = get_intersection(lines)

    if DEBUG:
        print()
        print("intersections:\n", intersections)
        print(len(intersections), len(intersections[0]), type(intersections))
        print(vertices_i.shape, type(vertices_i))

    if len(intersections) < 2:
        print("intersections not detected for file:", filename_i)
        return

    vertices_i = np.array(intersections)

    if DEBUG:
        print("vertices after:\n", vertices_i)
        print_vertices = [vertices_i[0], vertices_i[3], vertices_i[1], vertices_i[2], vertices_i[0]]
        for i in range(len(print_vertices) - 1):
            cv2.line(edges, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
            cv2.putText(edges, "%s" % (i), print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6,
                        (255, 255, 255), 3, cv2.LINE_AA)
        image_utils.save_image(edges, channel_name, filename_i, ImageOp.DEBUG_VERTICES)
    return image, vertices_i, is_vertical_i


async def transform_image(
        image_i, vertices_i, is_vertical, reference_position, reference_size, channel_name, filename_i, i, loop,
        map_size):
    transformation_dimensions = get_transformation_dimensions(
        vertices_i, is_vertical, reference_position, reference_size)
    padding, scale_factor, padded_and_scaled = transformation_dimensions

    scaled_image = scale_image(image_i, channel_name, scale_factor, filename_i)
    cropped_image, padding = crop_padding_(scaled_image, padding, filename_i, channel_name)

    scale = reference_size[1] / math.sqrt(int(map_size))
    coroutine = remove_clouds(cropped_image, ksize=7, sigma=15, template_height=scale)
    oppacity = loop.run_until_complete(coroutine)
    padding = np.flip(padding)

    return cropped_image, oppacity, padding, scale_factor, padded_and_scaled


async def patch_output(patch_work, scaled_padding, oppacity, size, bit, i):
    background = patch_work[
        scaled_padding[0]:scaled_padding[0]+size[0],
        scaled_padding[1]:scaled_padding[1]+size[1], :]
    if DEBUG:
        print(background.shape, oppacity.shape)
        print("scaled_padding", scaled_padding)
        print("size", size)
        print("bit size", bit.shape)
    if i == 0:  # background
        result = bit
    else:
        result = cv2.multiply(background, 1 - oppacity) + cv2.multiply(bit, oppacity)
    patch_work[scaled_padding[0]:scaled_padding[0]+size[0], scaled_padding[1]:scaled_padding[1]+size[1], :] = result
    return patch_work


def crop_output(image, position, size):
    return image[
        (position[1] - 20).clip(0): (position[1]+size[1] + 20).clip(0, image.shape[1]),
        (position[0] - size[0] - 20).clip(0): (position[0]+size[0] + 20).clip(0, image.shape[0])]


async def patch_partial_maps(
        channel_name: str,
        files: list,
        map_size: str,
        database_client: database_client.DatabaseClient = None,
        message=None):

    files.insert(0, "background")

    vertices = []
    vertical = []
    bits = []
    transparency_masks = []
    scaled_paddings = []
    sizes = []
    images = []
    scaled_vertices = []

    for i, filename_i in enumerate(files):
        processed_raw_map = await process_raw_map(filename_i, i, channel_name, map_size, database_client, message)
        if processed_raw_map is None:
            continue

        image, vertices_i, is_vertical_i = processed_raw_map
        images.append(image)
        vertices.append(vertices_i)
        vertical.append(is_vertical_i)

    reference_position, reference_size = get_references(vertices)
    loop = asyncio.get_running_loop()  # TODO look into timeout

    for i in range(len(images)):
        image_i = images[i]
        vertices_i = vertices[i]
        is_vertical_i = vertical[i]
        transformation = await transform_image(
            image_i, vertices_i, is_vertical_i, reference_position, reference_size, channel_name, filename_i, i, loop,
            map_size)
        scaled_image, oppacity, padding, scale_factor, padded_and_scaled = transformation

        transparency_masks.append(oppacity)
        bits.append(scaled_image)
        scaled_paddings.append(padding)
        scaled_vertices.append(padded_and_scaled)
        sizes.append(scaled_image.shape[0:2])

    output_size = np.max(np.array(scaled_paddings) + np.array(sizes), axis=0)

    patch_work = np.zeros([output_size[0], output_size[1], 3], np.uint8)

    for i in range(len(bits)):
        bit = bits[i]
        scaled_padding = scaled_paddings[i]
        size = sizes[i]
        oppacity = transparency_masks[i]
        patch_work = await patch_output(patch_work, scaled_padding, oppacity, size, bit, i)

    cropped_patch_work = crop_output(patch_work, reference_position, reference_size)

    if DEBUG:
        print(scaled_vertices)
        vertex_lines = np.zeros_like(patch_work)
        for j, vertices in enumerate(scaled_vertices):
            print_vertices = [vertices[0], vertices[3], vertices[1], vertices[2], vertices[0]]
            for i in range(len(print_vertices) - 1):
                cv2.line(vertex_lines, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
                cv2.putText(vertex_lines, "%s, %s" % (j, i), print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6,
                            (255, 255, 255), 3, cv2.LINE_AA)

        image_utils.save_image(vertex_lines, channel_name, 'map_patching_debut', ImageOp.DEBUG_VERTICES)

    if message is not None and database_client is not None:
        filename = database_client.add_resource(message, message.author, ImageOp.MAP_PATCHING_OUTPUT)
    else:
        filename = 'map_patching_debug'
    file_path = image_utils.save_image(cropped_patch_work, channel_name, filename, ImageOp.MAP_PATCHING_OUTPUT)
    return file_path, filename


def is_map_patching_request(message, attachment, filename):
    print(
        "is_map_patching_request",
        discord.PartialEmoji(name="🖼️") in [r.emoji for r in message.reactions],
        "🖼️" in [r.emoji for r in message.reactions],
        [r.emoji for r in message.reactions],
        message.reactions)
    return "🖼️" in [r.emoji for r in message.reactions]


async def remove_clouds(img, ksize=31, sigma=25, template_height=1):
    # read chessboard image
    # img = cv2.imread(image_path)
    img_alpha = np.ones((img.shape[0:2]))

    # read pawn image template
    template = image_utils.get_cloud_template()
    template_shape = template.shape

    scale = template_height / template_shape[1] * math.sqrt(2)

    new_dim = (int(template_shape[1] * scale), int(template_shape[0] * scale))

    template = cv2.resize(template, new_dim, interpolation=cv2.INTER_AREA)
    print("template new dim", template_shape, template.shape)

    hh, ww = template.shape[:2]

    # extract pawn base image and alpha channel and make alpha 3 channels
    pawn = template[:, :, 0:3]
    alpha_template = template[:, :, 3]
    alpha = cv2.merge([alpha_template, alpha_template, alpha_template])

    # do masked template matching and save correlation image
    corr_img = cv2.matchTemplate(img, pawn, cv2.TM_CCOEFF_NORMED, mask=alpha)
    correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)

    threshold = 60
    max_val = np.max(blur)
    rad = int(math.sqrt(hh * hh + ww * ww) / 4)

    while max_val > threshold:

        # find max value of correlation image
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        # print(max_val, max_loc)

        if max_val > threshold:
            # draw match on copy of input
            # cv2.rectangle(result, max_loc, (max_loc[0]+ww, max_loc[1]+hh), (0, 0, 255), 2)
            img_alpha[max_loc[1]: max_loc[1]+hh, max_loc[0]: max_loc[0]+ww] = \
                img_alpha[max_loc[1]: max_loc[1]+hh, max_loc[0]: max_loc[0]+ww] - alpha_template / 255.0

            # write black circle at max_loc in corr_img
            cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)

        else:
            break

    img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8)
    mask = cv2.merge([img_alpha_c, img_alpha_c, img_alpha_c]).astype(np.uint8)

    return mask


async def remove_clouds_scaled(img, ksize=7, sigma=15, template_height=None):
    template = image_utils.get_cloud_template()
    hh, ww = template.shape[:2]

    pawn = template[:, :, 0:3]
    alpha_template = template[:, :, 3]
    alpha = cv2.merge([alpha_template, alpha_template, alpha_template])

    result_set = []

    # TODO Optimisation possibilities:
    # - not go linear, look for the point
    # - start in the middle and go out, based on gradient
    # - stop when it goes down again
    # - change spacing
    # - move range based on estiamted scaling
    for scale in range(90, 110, 2):
        scale = scale / 100.0
        new_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break

        img_alpha = np.ones((resized.shape[0:2]))

        corr_img = cv2.matchTemplate(resized, pawn, cv2.TM_CCOEFF_NORMED, mask=alpha)
        correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)
        threshold = 100

        # cv2.imwrite('./cloud/blur_%.2f.png' % scale, blur)

        maximum = np.max(blur)

        max_val = np.max(blur)
        rad = int(math.sqrt(hh * hh + ww * ww) / 4)

        count = 0

        while max_val > threshold:

            # find max value of correlation image
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

            if max_val > threshold:
                # draw match on copy of input
                img_alpha[max_loc[1]: max_loc[1]+hh, max_loc[0]: max_loc[0]+ww] = \
                    img_alpha[max_loc[1]: max_loc[1]+hh, max_loc[0]: max_loc[0]+ww] - alpha_template / 255.0

                # write black circle at max_loc in corr_img
                cv2.circle(blur, (max_loc), radius=rad, color=0, thickness=cv2.FILLED)
                count += 1

            else:
                break
        img_alpha_c = img_alpha.clip(0, 1).astype(np.uint8)
        mask = cv2.merge([img_alpha_c, img_alpha_c, img_alpha_c]).astype(np.uint8)
        alpha_area = np.sum(img_alpha_c == 0) / (img_alpha_c.shape[0] * img_alpha_c.shape[1])
        result_set.append([scale, maximum, maximum / scale, count, alpha_area, mask])

    selected_scaling = sorted(result_set, key=lambda x: -x[2])[0]
    print("selected_scaling", selected_scaling[:5])
    resized_original = cv2.resize(selected_scaling[5], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return resized_original
