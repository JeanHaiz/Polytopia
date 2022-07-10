import cv2

import numpy as np

from typing import List
from typing import Tuple
from typing import Optional

from common import image_utils
from common.image_operation import ImageOp
from map_patching.image_param import ImageParam
from map_patching.corner_orientation import CornerOrientation
from database_interaction.database_client import DatabaseClient

from map_patching import map_patching_analysis


def patch_processed_images(
        database_client: DatabaseClient,
        image_filenames: List[str],
        map_size: str,
        guild_id: str,
        channel_id: str,
        channel_name: str,
        message_id: str,
        author_id: str,
        author_name: str) -> Tuple[str, str, list]:

    background_image = load_processed_background(map_size)
    background_params = load_background_params(database_client, map_size)

    if background_image is None or background_params is None:
        print("ANALYSING BACKGROUND")
        _, background_params = map_patching_analysis.analyse_background(database_client, map_size)
        background_image = load_processed_background(map_size)

    images_n_filenames = load_processed_images(image_filenames, channel_name)
    image_params = load_image_params(database_client, image_filenames)
    # missing_filenames = []

    processed_images = []
    processed_params = []

    for filename, image_entry in images_n_filenames:

        current_params = [ip for ip in image_params if ip.filename == filename]

        if image_entry is None or len(current_params) == 0:
            print("ANALYSING IMAGE:", filename, image_entry is None, current_params)
            raw_image = image_utils.load_image(channel_name, filename, ImageOp.MAP_INPUT)
            _, image_entry_params = map_patching_analysis.analyse_map(
                raw_image, database_client, channel_name, filename)
            analysed_map_image = image_utils.load_image(
                channel_name, filename, ImageOp.MAP_PROCESSED_IMAGE)
            if analysed_map_image is None or image_entry_params is None:
                print("Could not analyse image %s:" % filename, analysed_map_image, image_entry_params)
                continue
            processed_images.append(analysed_map_image)
            processed_params.append(image_entry_params)
        else:
            processed_images.append(image_entry)
            processed_params.append(current_params[0])

    # if len(images) != len(image_filenames) or image_params != len(image_filenames) or len(missing_filenames) != 0:

    patch_image = patch_processed_image_files(background_image, processed_images, background_params, processed_params)

    print("output before crop", patch_image.shape, background_params.getPosition(), background_params.getSize())
    output = crop_output(patch_image, background_params.getPosition(), background_params.getSize())
    print("after crop", output.shape)
    filename = database_client.add_resource(
        guild_id, channel_id, message_id, author_id, author_name, ImageOp.MAP_PATCHING_OUTPUT)
    return image_utils.save_image(output, channel_name, filename, ImageOp.MAP_PATCHING_OUTPUT), filename, []


def crop_output(
        image: np.ndarray,
        position: Tuple[np.int64, np.int64],
        size: Tuple[np.int64, np.int64]) -> np.ndarray:

    return image[
        (position[1] - 20).clip(0): (position[1] + size[1] + 20).clip(0, image.shape[0]),
        (position[0] - 20).clip(0): (position[0] + size[0] + 20).clip(0, image.shape[1])]
    return image[
        (position[1] - 20).clip(0): (position[1] + size[1] + 20).clip(0, image.shape[0]),
        (position[0] - size[0] - 20).clip(0): (position[0] + size[0] + 20).clip(0, image.shape[1])]


def load_processed_background(map_size: str) -> np.ndarray:
    return image_utils.get_processed_background_template(map_size)


def load_processed_images(
        image_filenames: List[str],
        channel_name: str) -> List[Tuple[str, np.ndarray]]:

    return [(filename, image_utils.load_image(
        channel_name,
        filename,
        image_utils.ImageOp.MAP_PROCESSED_IMAGE
    )) for filename in image_filenames]


def patch_processed_image_files(
        background_image: np.ndarray,
        images: List[np.ndarray],
        background_params: ImageParam,
        image_params: List[ImageParam]) -> np.ndarray:

    background_height = get_height(background_params.corners)
    background_width = get_width(background_params.corners)

    print("background shape (y, x)", background_height, background_width)

    output = background_image

    if background_height is None or background_width is None:
        print("background invalid: " + str(background_params.corners))
        return output
    else:
        for image_i, image_param_i in zip(images, image_params):
            if len(image_param_i.corners) == 0:
                print("no corner found for image", image_param_i)
                continue

            corner_orientations = [CornerOrientation(c[2]) for c in image_param_i.corners]

            if False:  # CornerOrientation.TOP in corner_orientations and CornerOrientation.BOTTOM in corner_orientations:
                image_i_height = get_height(image_param_i.corners) or 0
                scale_ratio = image_i_height / background_height

                # new_dim = image_i.shape * scale_ratio
                # new_dim = (int(image_i.shape[0] * scale_ratio), int(image_i.shape[1] * scale_ratio))
                new_dim = (int(image_i.shape[1] * scale_ratio), int(image_i.shape[0] * scale_ratio))
                scaled_image_i = cv2.resize(image_i, new_dim, interpolation=cv2.INTER_AREA)

                image_i_top = get_corner(image_param_i.corners, CornerOrientation.TOP) or [0, 0]
                background_top = get_corner(background_params.corners, CornerOrientation.TOP) or [0, 0]

                scaled_top_i = [int(i * scale_ratio) for i in image_i_top]

                padding_i = (background_top[1] - scaled_top_i[1], background_top[0] - scaled_top_i[0])

            elif False:  # CornerOrientation.RIGHT in corner_orientations and CornerOrientation.LEFT in corner_orientations:
                image_i_width = get_width(image_param_i.corners) or 0
                if image_i_width == 0:
                    print("image width is null", image_param_i.corners)
                    continue

                print("widths", image_i_width, background_width)

                scale_ratio = image_i_width / background_width

                print("scale ratio", scale_ratio, image_i_width)
                # new_dim = image_i.shape * scale_ratio
                # new_dim = (int(image_i.shape[0] * scale_ratio), int(image_i.shape[1] * scale_ratio))
                new_dim = (int(image_i.shape[1] * scale_ratio), int(image_i.shape[0] * scale_ratio))
                print(new_dim, image_i.shape, scale_ratio)
                scaled_image_i = cv2.resize(image_i, new_dim, interpolation=cv2.INTER_AREA)

                image_i_left = get_corner(image_param_i.corners, CornerOrientation.LEFT) or [0, 0]
                background_left = get_corner(background_params.corners, CornerOrientation.LEFT) or [0, 0]

                scaled_left_i = [int(i * scale_ratio) for i in image_i_left]

                padding_i = (background_left[1] - scaled_left_i[1], background_left[0] - scaled_left_i[0])

            else:
                print("third way")
                scaled_image_i = image_i
                corner_orientations.sort(key=lambda x: x.value)
                selected_corner_orientation_i = corner_orientations[0]
                print("Selected Corner", CornerOrientation(selected_corner_orientation_i))

                selected_corner_i = get_corner(image_param_i.corners, selected_corner_orientation_i) or (0, 0)
                background_selected_corner = get_corner(background_params.corners, selected_corner_orientation_i) \
                    or (0, 0)

                padding_i = (
                    background_selected_corner[0] - selected_corner_i[0],
                    background_selected_corner[1] - selected_corner_i[1])

            cropped_image_i, padding_i = crop_padding(scaled_image_i, padding_i)

            output = patch_image(
                patch_work=output,
                scaled_padding=padding_i,
                cropped_image_i=cropped_image_i)

    return output


def load_background_params(database_client: DatabaseClient, map_size: str) -> ImageParam:
    return database_client.get_background_image_params(int(map_size))


def load_image_params(
        database_client: DatabaseClient,
        image_filenames: List[str]) -> List[ImageParam]:
    return database_client.get_bulk_image_params(image_filenames)


def get_corner(corners: List[Tuple[int, int, int]], orientation: CornerOrientation) -> Optional[Tuple[int, int]]:
    for c in corners:
        if c[2] == orientation.value:
            return (c[0], c[1])
    return None


def get_height(corners: List[Tuple[int, int, int]]) -> Optional[int]:
    bottom = get_corner(corners, CornerOrientation.BOTTOM)
    top = get_corner(corners, CornerOrientation.TOP)
    if top is not None and bottom is not None:
        return bottom[1] - top[1]
    else:
        return None


def get_width(corners: List[Tuple[int, int, int]]) -> Optional[int]:
    right = get_corner(corners, CornerOrientation.RIGHT)
    left = get_corner(corners, CornerOrientation.LEFT)
    if right is not None and left is not None:
        return right[0] - left[0]
    else:
        return None


def crop_padding(
        image: np.ndarray,
        padding: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:

    if padding[0] < 0:
        image = image[:, -padding[0]:]
        padding = (0, padding[1])
    if padding[1] < 0:
        image = image[-padding[1]:, :]
        padding = (padding[0], 0)
    return image, padding


def crop_padding_(
        image: np.ndarray,
        padding: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:

    if padding[0] < 0:
        image = image[-padding[0]:, :]
        padding = (0, padding[1])
    if padding[1] < 0:
        image = image[:, -padding[1]:]
        padding = (padding[0], 0)
    return image, padding


def patch_image(
        patch_work: np.ndarray,
        scaled_padding: Tuple[int, int],
        cropped_image_i: np.ndarray) -> np.ndarray:

    print("patch work", patch_work.shape)

    print("cropped_image_i shape", cropped_image_i.shape)
    # reshaped_cropped_image_i = np.reshape(cropped_image_i,
    #   (cropped_image_i.shape[1], cropped_image_i.shape[0], cropped_image_i.shape[2]))
    # print("reshaped cropped_image_i shape", reshaped_cropped_image_i.shape)
    reshaped_cropped_image_i = cropped_image_i
    oppacity: np.ndarray = reshaped_cropped_image_i[:, :, 3]
    size: Tuple[int, int] = reshaped_cropped_image_i.shape[0:2]
    bit: np.ndarray = reshaped_cropped_image_i[:, :, 0:3]

    # print(len(oppacity))
    print("oppacity", oppacity.shape)
    print("bit", bit.shape)
    print("padding", scaled_padding)

    print("patch types", type(patch_work), type(oppacity), type(bit))
    print(patch_work.shape, scaled_padding, oppacity.shape, size, bit.shape)

    print("shape for background", patch_work.shape)
    print(scaled_padding[0], scaled_padding[0] + size[1], size[1])
    print(scaled_padding[1], scaled_padding[1] + size[0], size[0])

    background = patch_work[
        scaled_padding[1]:scaled_padding[1] + size[0],
        scaled_padding[0]:scaled_padding[0] + size[1],
        :]

    cv2.imwrite("./patch-work-piece.png", background)
    print("backgound shape", background.shape)

    print("shape for cropped bit", bit.shape)
    print(
        min(
            bit.shape[0],
            scaled_padding[0] + size[0],
            background.shape[1]
        ), bit.shape[0], scaled_padding[0] + size[0], background.shape[1])
    print(
        min(
            bit.shape[1],
            scaled_padding[1] + size[1],
            background.shape[0]
        ), bit.shape[1], scaled_padding[1] + size[1], background.shape[0])
    cropped_bit = bit[
        :min(bit.shape[0], scaled_padding[0] + size[0], background.shape[0]),
        :min(bit.shape[1], scaled_padding[1] + size[1], background.shape[1]),
        :]
    cv2.imwrite("./cropped_bit.png", cropped_bit)
    print("cropped bit shape", cropped_bit.shape)

    print("shape for cropped oppacity", oppacity.shape)
    print(
        min(
            oppacity.shape[0],
            scaled_padding[0] + size[0],
            background.shape[1]
        ), oppacity.shape[0], scaled_padding[0] + size[0], background.shape[1])
    print(
        min(
            oppacity.shape[1],
            scaled_padding[1] + size[1],
            background.shape[0]
        ), oppacity.shape[1], scaled_padding[1] + size[1], background.shape[0])
    cropped_oppacity = (oppacity[
        :min(oppacity.shape[0], scaled_padding[0] + size[0], background.shape[0]),
        :min(oppacity.shape[1], scaled_padding[1] + size[1], background.shape[1])] / 255).astype('uint8')

    print("cropped oppacity shape", cropped_oppacity.shape)
    cv2.imwrite("./cropped-oppacity.png", cropped_oppacity)

    print("should match", background.shape, cv2.merge((cropped_oppacity, cropped_oppacity, cropped_oppacity)).shape)

    # background = np.reshape(background, (background.shape[1], background.shape[0], background.shape[2]))
    # print("do match", background.shape, cv2.merge((cropped_oppacity, cropped_oppacity, cropped_oppacity)).shape)
    # print("cropped max", np.max(cv2.merge((cropped_oppacity, cropped_oppacity, cropped_oppacity))))
    opaque_bit = cv2.multiply(background, 1 - cv2.merge((cropped_oppacity, cropped_oppacity, cropped_oppacity)))
    transparent_bit = cv2.multiply(cropped_bit, cv2.merge((cropped_oppacity, cropped_oppacity, cropped_oppacity)))

    result = opaque_bit + transparent_bit

    patch_work[
        scaled_padding[1]: scaled_padding[1] + size[0],
        scaled_padding[0]: scaled_padding[0] + size[1],
        :] = result
    return patch_work
