import cv2
import discord
import numpy as np
import pandas as pd

from common import image_utils
from common.image_utils import ImageOperation
from database_interaction import database_client

DEBUG = 1


def get_one_color_edges(image, channel=None, filename=None, low=50, high=150):
    edges = cv2.Canny(image, low, high, apertureSize=3)
    if filename and channel:
        image_utils.save_image(edges, channel, filename, ImageOperation.ONE_COLOR_EDGES)
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
            image_utils.save_image(image, channel, filename, ImageOperation.HOUGH_LINES)
            # image_utils.write_img(image, filename, "houghlines5")
    return processed


def get_three_color_edges(image, channel=None, filename=None):
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
        image_utils.save_image(edges, channel, filename, ImageOperation.THREE_COLOR_EDGES)
        # image_utils.write_img(edges, filename, 'three_color_edges_200_250')
    return edges


def select_contours(image, filename=None):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    keepers = []

    for index_, contour_ in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour_)

        if w > int(image.shape[0] / 7) and h > int(image.shape[1] / 7):
            keepers.append([contour_, [x, y, w, h]])

    # print("#contours kept: %d" % len(keepers))
    keepers.sort(key=lambda k: -(k[1][2] + k[1][3]))
    return keepers[0][0]


# image: edges
def draw_contour(image, channel, contour_, epsilon=None, epsilonFactor=0.005, filename=None):
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

    if filename is not None:
        cv2.drawContours(processed, [approx], 0, (255, 255, 255), 3)
        cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

        image_utils.save_image(processed, channel, filename, ImageOperation.OVERLAY)
        image_utils.save_image(mask, channel, filename, ImageOperation.MASK)
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


def get_transformation_dimensions(vertices, reference_position, reference_size):

    ratio = reference_size[0] / reference_size[1]  # width / heigth = 1.6108...

    d_h = vertices[1][1] - vertices[0][1]
    d_w = vertices[3][0] - vertices[2][0]

    # print(i, "%.2f" % (d_w / d_h), estimated_size)

    if d_w / d_h < ratio:  # smaller width or larger height proportionally
        # missing width; scale on height
        scale_factor = d_h / reference_size[1]  # then divide by scale factor to find right size
        padding = [
            int(reference_position[0] - (vertices[0][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[0][1] - 50) / scale_factor)
        ]
    else:
        # missing height; scale on width
        scale_factor = d_w / reference_size[0]
        padding = [
            int(reference_position[0] - (vertices[2][0] - 50) / scale_factor),
            int(reference_position[1] - (vertices[2][1] - 50) / scale_factor)
        ]
    if DEBUG:
        print("scale factor:", scale_factor)
        print("padding:", padding)
        print("cropped and scaled vertices:\n",
              # (vertices + padding - reference_positions[0]) / scale_factor + reference_positions[0])
              # (vertices + padding) / scale_factor)
              vertices / scale_factor + padding)
    return scale_factor, padding


def crop_padding(image, padding):
    if padding[0] < 0:
        image = image[:, -padding[0]:]
        padding[0] = 0
    if padding[1] < 0:
        image = image[-padding[1]:, :]
        padding[1] = 0
    return image


def scale_image(image, channel, scale_factor, filename=None):
    new_dim = (int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor))
    # print("new dimenstion", new_dim, np.flip(image.shape))
    scaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    # print("image shape", np.flip(scaled_bit.shape[0:2]), reference_padding)
    if filename is not None:
        image_utils.save_image(scaled_image, channel, filename, ImageOperation.SCALE)
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
        if top_bottom_ref and dx > reference_position[0]:
            reference_position = [dx, dy]
        if not top_bottom_ref and dy > reference_position[1]:
            reference_position = [dx, dy]

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
             (delta[1] / delta[0]) if delta[0] != 0 else np.sign(delta[0]) * np.sign(delta[1]) * np.inf,
             np.sign(delta[0]), np.sign(delta[1]), p0.tolist(), p1.tolist()) for j, p0, p1, delta in delta]

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

            x = (h2 - h1) / (p1 - p2)
            y = p1 * x + h1
            intersections.append([x, y])

    elif len(lines) == 2:
        return
    if DEBUG:
        print(intersections)
    return intersections


async def patch_partial_maps(message, files, database_client: database_client.DatabaseClient):

    # reference_size = (2028, 1259)
    # reference_positions = [(1121 - 56, 274 - 221), (105 - 56, 880 - 221)]

    # reference_positions = [(2242, 548), (210, 1760)]

    vertices = []
    bits = []
    scaled_paddings = []
    sizes = []
    images = []
    scaled_vertices = []

    for i, filename_i in enumerate(files):
        print("map_patching_utils", filename_i)
        image = await image_utils.load_image(database_client, message, filename_i, image_utils.ImageOperation.INPUT)
        if image is None:
            if DEBUG:
                print("image not found:", filename_i)
            print("EXIT LOOP")
            return None

        images.append(image)

        edges = get_three_color_edges(image, filename=filename_i, channel=message.channel)

        contour = select_contours(edges, filename_i)
        polygon = draw_contour(edges, message.channel, contour, filename=filename_i)
        vertices_i = compute_vertices(polygon)
        lines = get_lines([p[0] for p in polygon])
        intersections = get_intersection(lines)

        if DEBUG:
            print()
            print("intersections:\n", intersections)
            print("vertices:\n", vertices_i)

        vertices_i = np.concatenate((vertices_i[0:2], intersections))

        if DEBUG:
            print("vertices after:\n", vertices_i)

        vertices.append(vertices_i)

    reference_position, reference_size = get_references(vertices)

    for i, filename_i in enumerate(files):
        image_i = images[i]
        vertices_i = vertices[i]
        scale_factor, padding = get_transformation_dimensions(vertices_i, reference_position, reference_size)

        # print("pre-translation padding", reference_padding, np.flip(image_i.shape[0:2]))
        # print(np.flip(image.shape[0:2]),
        #       (image.shape[1] + reference_padding[0], image_i.shape[0] + reference_padding[1]))

        cropped_image = crop_padding(image_i, padding)
        scaled_image = scale_image(cropped_image, message.channel, scale_factor)
        # scaled_image = scale_image(image, scale_factor)

        bits.append(scaled_image)
        # scaled_paddings.append((padding).astype(int))
        scaled_paddings.append(padding)
        scaled_vertices.append((vertices_i / scale_factor).tolist())
        sizes.append(np.flip(scaled_image.shape[0:2]))

        if DEBUG:
            print()

    output_size = np.max(np.array(scaled_paddings) + np.array(sizes), axis=0)
    if DEBUG:
        print("output size:", output_size)
    patch_work = np.zeros([output_size[1], output_size[0], 3], np.uint8)

    for i in range(len(bits)):
        bit = bits[i]
        scaled_padding = scaled_paddings[i]
        size = sizes[i]
        patch_work[scaled_padding[1]:scaled_padding[1]+size[1], scaled_padding[0]:scaled_padding[0]+size[0], :] = bit
        if DEBUG:
            print(scaled_vertices)
            # image = cv2.polylines(patch_work, [scaled_vertices], True, (255, 255, 255), 10)

    filename = database_client.add_resource(message, message.author, ImageOperation.MAP_PATCHING_OUTPUT)
    file_path = image_utils.save_image(patch_work, message.channel, filename, ImageOperation.MAP_PATCHING_OUTPUT)
    return file_path


def is_map_patching_request(message, attachment, filename):
    print(
        "is_map_patching_request",
        discord.PartialEmoji(name="🖼️") in [r.emoji for r in message.reactions],
        "🖼️" in [r.emoji for r in message.reactions],
        [r.emoji for r in message.reactions],
        message.reactions)
    return "🖼️" in [r.emoji for r in message.reactions]
