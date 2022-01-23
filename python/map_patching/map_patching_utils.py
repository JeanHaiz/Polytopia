import re
import cv2
import math
import discord
import pytesseract
import numpy as np
import pandas as pd

from common import image_utils
from common.image_utils import ImageOp
from database_interaction import database_client

DEBUG = 1


def get_one_color_edges(image, channel=None, filename=None, low=50, high=150):
    edges = cv2.Canny(image, low, high, apertureSize=3)
    if filename and channel:
        image_utils.save_image(edges, channel.name, filename, ImageOp.ONE_COLOR_EDGES)
        # image_utils.write_img(edges, filename, 'one_colour_edges_%d_%d' % (low, high))
    return edges


def getLines(image, minLineLength=500, channel=None, filename=None):
    processed = image.copy()
    edges = get_one_color_edges(image, channel=channel)
    lines = cv2.HoughLinesP(image=edges, rho=0.1, theta=np.pi/180, threshold=150, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=100)

    a, b, c = lines.shape
    for i in range(a):
        cv2.line(processed, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                 cv2.LINE_AA)
        if filename:
            image_utils.save_image(image, channel.name, filename, ImageOp.HOUGH_LINES)
            # image_utils.write_img(image, filename, "houghlines5")
    return processed


def get_three_color_edges(image, channel_name=None, filename=None):
    img = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT)

    # Split out each channel
    blue, green, red = cv2.split(img)

    # Run canny edge detection on each channel
    blue_edges = cv2.Canny(blue, 200, 250)
    green_edges = cv2.Canny(green, 200, 250)
    red_edges = cv2.Canny(red, 200, 250)

    # Join edges back into image
    edges = blue_edges | green_edges | red_edges
    if filename is not None:
        image_utils.save_image(edges, channel_name, filename, ImageOp.THREE_COLOR_EDGES)
        # image_utils.write_img(edges, filename, 'three_color_edges_200_250')
    return edges


def select_contours(image, filename=None):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    keepers = []

    for contour_ in contours:

        x, y, w, h = cv2.boundingRect(contour_)

        if w > int(image.shape[0] / 7) and h > int(image.shape[1] / 7):
            keepers.append([contour_, [x, y, w, h]])

    # print("#contours kept: %d" % len(keepers))
    keepers.sort(key=lambda k: -(k[1][2] + k[1][3]))
    return keepers[0][0]


# image: edges
def draw_contour(image, contour_, channel_name=None, epsilon=None, epsilonFactor=0.005, filename=None):
    processed = image.copy()
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

    if filename is not None and channel_name is not None:
        cv2.drawContours(processed, [approx], 0, (255, 255, 255), 3)
        cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

        image_utils.save_image(processed, channel_name, filename, ImageOp.OVERLAY)
        image_utils.save_image(mask, channel_name, filename, ImageOp.MASK)
        # image_utils.write_img(processed, filename, "overlay")
        # image_utils.write_img(mask, filename, "mask")
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

    # ratio = reference_size[0] / reference_size[1]  # width / heigth = 1.6108...

    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]

    # print(i, "%.2f" % (d_w / d_h), estimated_size)

    # print("here", d_w / d_h < ratio, d_w, d_h, ratio)
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

    # scaled_padding = np.flip(scaled_padding)
    if DEBUG:
        print("scale factor:", scale_factor)
        print("padding:", scaled_padding)
        print("cropped and scaled vertices:\n",
              # (vertices + padding - reference_positions[0]) / scale_factor + reference_positions[0])
              # (vertices + padding) / scale_factor)
              padded_and_scaled_vertices)
    return scaled_padding, scale_factor, padded_and_scaled_vertices


def crop_padding_(image, padding):
    if padding[0] < 0:
        image = image[:, -padding[0]:]
        padding[0] = 0
    if padding[1] < 0:
        image = image[-padding[1]:, :]
        padding[1] = 0
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
    # print("new dimenstion", new_dim, np.flip(image.shape))
    scaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    # print("image shape", np.flip(scaled_bit.shape[0:2]), reference_padding)
    if filename is not None and channel_name is not None:
        image_utils.save_image(scaled_image, channel_name, filename, ImageOp.SCALE)
        # image_utils.write_img(scaled_image, filename, "position&scale")
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


async def process_raw_map(filename_i, i, channel_name, map_size):
    print("map_patching_utils", filename_i)
    if i == 0:
        image = image_utils.get_background_template(map_size)
    else:
        image = await image_utils.load_image(database_client, channel_name, None, filename_i, ImageOp.INPUT)

    print("size for i", i, image.shape, type(image))
    if image is None:
        if DEBUG:
            print("image not found:", filename_i)
        print("EXIT LOOP")
        return None

    edges = get_three_color_edges(image, channel_name, filename_i)
    blur = cv2.GaussianBlur(edges, (15, 15), sigmaX=25)
    contour = select_contours(blur, filename_i)
    polygon = draw_contour(edges, contour, channel_name, filename=filename_i)
    vertices_i = compute_vertices(polygon)
    is_vertical_i = get_orientation(vertices_i)
    lines = get_lines([p[0] for p in polygon])

    if lines is None or len(lines) != 4:
        print("lines not detected for file:", filename_i)
        return

    intersections = get_intersection(lines)

    if DEBUG:
        print()
        print("intersections:\n", intersections)

    if len(intersections) != 2:
        print("intersections not detected for file:", filename_i)
        return
    if is_vertical_i:
        vertices_i = np.concatenate((vertices_i[0:2], intersections))
    else:
        vertices_i = np.concatenate((intersections, vertices_i[2:4]))

    if DEBUG:
        print("vertices after:\n", vertices_i)
        print_vertices = [vertices_i[0], vertices_i[3], vertices_i[1], vertices_i[2], vertices_i[0]]
        for i in range(len(print_vertices) - 1):
            cv2.line(edges, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
            cv2.putText(edges, "%s" % (i), print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6, 
                        (255, 255, 255), 3, cv2.LINE_AA)
        image_utils.save_image(edges, channel_name, filename_i, ImageOp.DEBUG_VERTICES)
    return image, vertices_i, is_vertical_i


async def transform_image(image_i, vertices_i, is_vertical, reference_position, reference_size, channel_name):
    transformation_dimensions = get_transformation_dimensions(
        vertices_i, is_vertical, reference_position, reference_size)
    padding, scale_factor, padded_and_scaled = transformation_dimensions

    # print("pre-translation padding", reference_padding, np.flip(image_i.shape[0:2]))
    # print(np.flip(image.shape[0:2]),
    #       (image.shape[1] + reference_padding[0], image_i.shape[0] + reference_padding[1]))

    cropped_image, padding = crop_padding_(image_i, padding)
    scaled_image = scale_image(cropped_image, channel_name, scale_factor)
    oppacity = remove_clouds(scaled_image)

    padding = np.flip(padding)

    return scaled_image, oppacity, padding, scale_factor, padded_and_scaled


async def patch_output(patch_work, scaled_padding, oppacity, size, bit, i):
    background = patch_work[
        scaled_padding[0]:scaled_padding[0]+size[0],
        scaled_padding[1]:scaled_padding[1]+size[1], :]
    print(background.shape, oppacity.shape)
    print("scaled_padding", scaled_padding)
    print("size")
    print("bit size", bit.shape)
    if i == 0:
        result = bit
    else:
        result = cv2.multiply(background, 1 - oppacity) + cv2.multiply(bit, oppacity)
    # result = cv2.multiply(bit, oppacity)
    patch_work[scaled_padding[0]:scaled_padding[0]+size[0], scaled_padding[1]:scaled_padding[1]+size[1], :] = result
    return patch_work


async def patch_partial_maps(
        channel_name: str,
        files: list,
        map_size: str,
        database_client: database_client.DatabaseClient = None,
        message=None):
    # reference_size = (2028, 1259)
    # reference_positions = [(1121 - 56, 274 - 221), (105 - 56, 880 - 221)]

    # reference_positions = [(2242, 548), (210, 1760)]

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
        image, vertices_i, is_vertical_i = await process_raw_map(filename_i, i, channel_name, map_size)
        images.append(image)
        vertices.append(vertices_i)
        vertical.append(is_vertical_i)

    reference_position, reference_size = get_references(vertices)

    for i in range(len(images)):
        image_i = images[i]
        vertices_i = vertices[i]
        is_vertical_id = vertical[i]
        transformation = await transform_image(image_i, vertices_i, is_vertical_i, reference_position, reference_size, channel_name)
        scaled_image, oppacity, padding, scale_factor, padded_and_scaled = transformation

        transparency_masks.append(oppacity)
        bits.append(scaled_image)
        scaled_paddings.append(padding)
        scaled_vertices.append(padded_and_scaled)
        # scaled_vertices.append([[int(a) for a in b] for b in (vertices_i / scale_factor).tolist()])
        sizes.append(scaled_image.shape[0:2])

        print("sizes:", scaled_image.shape, oppacity.shape)

    output_size = np.max(np.array(scaled_paddings) + np.array(sizes), axis=0)
    if DEBUG:
        print("output size:", output_size)
    patch_work = np.zeros([output_size[0], output_size[1], 3], np.uint8)

    for i in range(len(bits)):
        bit = bits[i]
        scaled_padding = scaled_paddings[i]
        # scaled_padding = [scaled_padding[1], scaled_padding[0]]
        # scaled_padding = np.clip(scaled_padding, 0, None)
        size = sizes[i]
        oppacity = transparency_masks[i]
        patch_work = await patch_output(patch_work, scaled_padding, oppacity, size, bit, i)

    if DEBUG:
        print(scaled_vertices)
        vertex_lines = np.zeros_like(patch_work)
        for j, vertices in enumerate(scaled_vertices):
            print_vertices = [vertices[0], vertices[3], vertices[1], vertices[2], vertices[0]]
            for i in range(len(print_vertices) - 1):
                cv2.line(vertex_lines, print_vertices[i], print_vertices[i + 1], (255, 255, 255), 2)
                cv2.putText(vertex_lines, "%s, %s" % (j, i), print_vertices[i], cv2.FONT_HERSHEY_COMPLEX, 6, 
                            (255, 255, 255), 3, cv2.LINE_AA)

                # cv2.line(patch_work, vertices[i], vertices[i+1])
                # image = cv2.polylines(patch_work, [scaled_vertices], True, (255, 255, 255), 10)
        image_utils.save_image(vertex_lines, channel_name, 'map_patching_debut', ImageOp.DEBUG_VERTICES)

    if message is not None and database_client is not None:
        filename = database_client.add_resource(message, message.author, ImageOp.MAP_PATCHING_OUTPUT)
        file_path = image_utils.save_image(patch_work, channel_name, filename, ImageOp.MAP_PATCHING_OUTPUT)
    else:
        file_path = image_utils.save_image(patch_work, channel_name, 'map_patching_debug', ImageOp.MAP_PATCHING_OUTPUT)
    return file_path


def is_map_patching_request(message, attachment, filename):
    print(
        "is_map_patching_request",
        discord.PartialEmoji(name="🖼️") in [r.emoji for r in message.reactions],
        "🖼️" in [r.emoji for r in message.reactions],
        [r.emoji for r in message.reactions],
        message.reactions)
    return "🖼️" in [r.emoji for r in message.reactions]


def remove_clouds(img, ouptut_prefix=None, ksize=31, sigma=25):
    # read chessboard image
    # img = cv2.imread(image_path)
    img_alpha = np.ones((img.shape[0:2]))

    # read pawn image template
    template = image_utils.get_cloud_template()
    hh, ww = template.shape[:2]

    # extract pawn base image and alpha channel and make alpha 3 channels
    pawn = template[:, :, 0:3]
    alpha_template = template[:, :, 3]
    alpha = cv2.merge([alpha_template, alpha_template, alpha_template])

    # do masked template matching and save correlation image
    corr_img = cv2.matchTemplate(img, pawn, cv2.TM_CCOEFF_NORMED, mask=alpha)
    correlation_raw = (255 * corr_img).clip(0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(correlation_raw, (ksize, ksize), sigmaX=sigma)
    blur_copy = blur.copy()

    threshold = 40
    result = img.copy()
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

    if False:
        # image_utils.save_image(correlation_raw, channel)
        cv2.imwrite(ouptut_prefix + '_correlation_3.png', correlation_raw)
        cv2.imwrite(ouptut_prefix + '_blur.jpg', blur_copy)
        cv2.imwrite(ouptut_prefix + '_correlation_2.png', corr_img)
        cv2.imwrite(ouptut_prefix + "_img_alpha.png", (255*img_alpha).clip(0, 255).astype(np.uint8))
        # cv2.imwrite(ouptut_prefix + "_with_alpha.png", with_alpha)
        cv2.imwrite(ouptut_prefix + '_pawn.png', pawn)
        cv2.imwrite(ouptut_prefix + '_alpha.png', alpha)
        cv2.imwrite(ouptut_prefix + '_correlation.png', blur)
        cv2.imwrite(ouptut_prefix + '_matches2.jpg', result)
    return mask


def get_turn(image):
    crop = image[:int(image.shape[0] / 6), :]
    grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    (_, blackAndWhiteImage) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
    selected_image = blackAndWhiteImage
    # map_text = pytesseract.image_to_string(selected_image, config='--oem 3 --psm 6').split("\n")

    contours, _ = cv2.findContours(selected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
                    elif x >= row_min_i and x <= row_max_i and x + w >= row_max_i:
                        row_max_i = x + w
                        found_row = True
                        rows[i] = rows[i] + [c]
                    elif x <= row_min_i and x + w >= row_min_i and x + w <= row_max_i:
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

    delta = 15
    turn = None
    for i in rows.keys():
        row_min_i = max(row_mins[i] - delta, 0)
        row_max_i = min(row_maxes[i] + delta, selected_image.shape[0])
        row_image = selected_image[row_min_i:row_max_i]
        cv2.imwrite("binary_%d.png" % i, row_image)
        row_text = pytesseract.image_to_string(row_image, config='--psm 6')

        cleared_row_text = row_text.replace("\n", "").replace("\x0c", "")
        cleared_row_text = re.sub(r"[^a-zA-Z0-9 ]", "", cleared_row_text)
        if 'Scores' not in cleared_row_text and 'Stars' not in cleared_row_text:
            cleared_row_text_split = cleared_row_text.split(" ")
            if len(cleared_row_text_split) > 2 and cleared_row_text_split[2].isnumeric():
                turn = cleared_row_text_split[2]
    if turn is None:
        print("turn not recognised")
    else:
        print("turn %s" % turn)
    return turn
