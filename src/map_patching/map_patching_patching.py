import gc
import os
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
from map_patching.map_patching_errors import MapPatchingErrors

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


def patch_processed_images(
        image_filenames: List[str],
        map_size: str,
        guild_id: int,
        channel_id: int,
        channel_name: str,
        message_id: int,
        author_id: int,
        author_name: str,
        action_debug: bool) -> Tuple[str, str, list]:
    
    gc.set_threshold(50, 10, 10)

    database_client = DatabaseClient(
        user="discordBot", password="password123", port="5432", database="polytopiaHelper_dev",
        host="database")
    database_client.dispose()

    patching_errors = []

    images_n_check = check_processed_images(image_filenames, channel_name)
    
    image_params = load_image_params(database_client, image_filenames)

    if DEBUG:
        print("images_n_filenames:", images_n_check)
        print("image params:", image_params)

    processed_params = []

    for filename, image_check in images_n_check:

        current_params = [ip for ip in image_params if ip.filename == filename]

        if not image_check or len(current_params) == 0:
            print("ANALYSING IMAGE:", filename, current_params)
            raw_image = image_utils.load_image(channel_name, filename, ImageOp.MAP_INPUT)
            if raw_image is None:
                print("Image not found %s" % filename)
                patching_errors.append((MapPatchingErrors.NO_FILE_FOUND, filename))
                continue
            _, image_entry_params = map_patching_analysis.analyse_map(
                raw_image, database_client, channel_name, channel_id, filename, action_debug)
            analysed_map_image = image_utils.load_image(
                channel_name, filename, ImageOp.MAP_PROCESSED_IMAGE)
            if analysed_map_image is None or image_entry_params is None or len(image_entry_params.corners) == 0:
                patching_errors.append((MapPatchingErrors.MAP_NOT_RECOGNIZED, filename))
                print("Could not analyse image %s:" % filename, analysed_map_image is None, image_entry_params is None)
                continue
            processed_params.append(image_entry_params)
        else:
            processed_params.append(current_params[0])
        gc.collect()

    background_params = load_background_params(database_client, map_size)
    background_image = load_processed_background(map_size)

    if background_image is None or background_params is None:
        if DEBUG:
            print("ANALYSING BACKGROUND")
        _, background_params = map_patching_analysis.analyse_background(database_client, map_size, action_debug)
        background_image = load_processed_background(map_size)
    
    print("collector 4", gc.get_count())
    patched_image = patch_processed_image_files(background_image, background_params, channel_name, processed_params)
    print("collector 6", gc.get_count())
    
    if DEBUG:
        print("output before crop", patched_image.shape, background_params.get_position(), background_params.get_size())

    output_filename = database_client.add_resource(
        guild_id, channel_id, message_id, author_id, author_name, ImageOp.MAP_PATCHING_OUTPUT)
    if output_filename is None:
        output_path = None
        patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_SAVED, ""))
    else:
        output = crop_output(patched_image, background_params.get_position(), background_params.get_size())
    
        if DEBUG:
            print("after crop", output.shape)

        output_path = image_utils.save_image(output, channel_name, output_filename, ImageOp.MAP_PATCHING_OUTPUT)
        if output_path is None:
            patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_SAVED, ""))

    return output_path, output_filename, patching_errors


def crop_output(
        image: np.ndarray,
        position: Tuple[np.int64, np.int64],
        size: Tuple[np.int64, np.int64]) -> np.ndarray:

    return image[
        (position[1] - 20).clip(0): (position[1] + size[1] + 20).clip(0, image.shape[0]),
        (position[0] - 20).clip(0): (position[0] + size[0] + 20).clip(0, image.shape[1])]


def load_processed_background(map_size: str) -> np.ndarray:
    return image_utils.get_processed_background_template(map_size)


def check_processed_images(
        image_filenames: List[str],
        channel_name: str) -> List[Tuple[str, bool]]:

    return [(filename, image_utils.load_image(
        channel_name,
        filename,
        image_utils.ImageOp.MAP_PROCESSED_IMAGE
    ) is not None) for filename in image_filenames]


def patch_processed_image_files(
        background_image: np.ndarray,
        background_params: ImageParam,
        channel_name: str,
        image_params: List[ImageParam]) -> np.ndarray:

    background_height = get_height(background_params.corners)
    background_width = get_width(background_params.corners)

    if DEBUG:
        print("background shape (y, x)", background_height, background_width)

    if background_height is None or background_width is None:
        if DEBUG:
            print("background invalid: " + str(background_params.corners))
        return background_image
    else:
        for image_param_i in image_params:
            image_i = image_utils.load_image(
                channel_name, image_param_i.filename, image_utils.ImageOp.MAP_PROCESSED_IMAGE)

            if len(image_param_i.corners) == 0:
                print("no corner found for image", image_param_i)
                continue

            corner_orientations = [CornerOrientation(c[2]) for c in image_param_i.corners]

            corner_orientations.sort(key=lambda x: x.value)
            selected_corner_orientation_i = corner_orientations[0]
            if DEBUG:
                print("Selected Corner", CornerOrientation(selected_corner_orientation_i))

            selected_corner_i = get_corner(image_param_i.corners, selected_corner_orientation_i) or (0, 0)
            background_selected_corner = get_corner(background_params.corners, selected_corner_orientation_i) \
                or (0, 0)

            padding_i = (
                background_selected_corner[0] - selected_corner_i[0],
                background_selected_corner[1] - selected_corner_i[1])

            cropped_image_i, padding_i = crop_padding(image_i, padding_i)

            background_image = patch_image(
                patch_work=background_image,
                scaled_padding=padding_i,
                reshaped_cropped_image_i=cropped_image_i)
            print("collector 5", gc.get_count())
            del image_i, cropped_image_i,
            gc.collect()

        return background_image


def load_background_params(database_client: DatabaseClient, map_size: str) -> ImageParam:
    return database_client.get_background_image_params(int(map_size))


def load_image_params(
        database_client: DatabaseClient,
        image_filenames: List[str]) -> List[ImageParam]:
    return database_client.get_bulk_image_params(image_filenames)


def get_corner(corners: List[Tuple[int, int, int]], orientation: CornerOrientation) -> Optional[Tuple[int, int]]:
    for c in corners:
        if c[2] == orientation.value:
            return c[0], c[1]
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
        reshaped_cropped_image_i: np.ndarray) -> np.ndarray:

    if DEBUG:
        print("patch work", patch_work.shape)
        print("cropped_image_i shape", reshaped_cropped_image_i.shape)

    opacity: np.ndarray = reshaped_cropped_image_i[:, :, 3]
    size: Tuple[int, int] = reshaped_cropped_image_i.shape[0:2]
    bit: np.ndarray = reshaped_cropped_image_i[:, :, 0:3]

    if DEBUG:
        # print(len(opacity))
        print("opacity", opacity.shape)
        print("bit", bit.shape)
        print("padding", scaled_padding)
    
        print("patch types", type(patch_work), type(opacity), type(bit))
        print(patch_work.shape, scaled_padding, opacity.shape, size, bit.shape)
    
        print("shape for background", patch_work.shape)
        print(scaled_padding[0], scaled_padding[0] + size[1], size[1])
        print(scaled_padding[1], scaled_padding[1] + size[0], size[0])

    background = patch_work[
        scaled_padding[1]:scaled_padding[1] + size[0],
        scaled_padding[0]:scaled_padding[0] + size[1],
        :]

    if DEBUG:
        cv2.imwrite("./patch-work-piece.png", background)
        print("background shape", background.shape)
    
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
    
    if DEBUG:
        cv2.imwrite("./cropped_bit.png", cropped_bit)
        print("cropped bit shape", cropped_bit.shape)
    
        print("shape for cropped opacity", opacity.shape)
        print(
            min(
                opacity.shape[0],
                scaled_padding[0] + size[0],
                background.shape[1]
            ), opacity.shape[0], scaled_padding[0] + size[0], background.shape[1])
        print(
            min(
                opacity.shape[1],
                scaled_padding[1] + size[1],
                background.shape[0]
            ), opacity.shape[1], scaled_padding[1] + size[1], background.shape[0])

    cropped_opacity = (opacity[
        :min(opacity.shape[0], scaled_padding[0] + size[0], background.shape[0]),
        :min(opacity.shape[1], scaled_padding[1] + size[1], background.shape[1])] / 255).astype('uint8')

    if DEBUG:
        print("cropped opacity shape", cropped_opacity.shape)
        cv2.imwrite("./cropped-opacity.png", cropped_opacity)
    
        print("should match", background.shape, cv2.merge((cropped_opacity, cropped_opacity, cropped_opacity)).shape)

    opaque_bit = cv2.multiply(background, 1 - cv2.merge((cropped_opacity, cropped_opacity, cropped_opacity)))
    transparent_bit = cv2.multiply(cropped_bit, cv2.merge((cropped_opacity, cropped_opacity, cropped_opacity)))

    result = opaque_bit + transparent_bit

    patch_work[
        scaled_padding[1]: scaled_padding[1] + size[0],
        scaled_padding[0]: scaled_padding[0] + size[1],
        :] = result
    return patch_work
