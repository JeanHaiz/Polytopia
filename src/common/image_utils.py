import os
import re
import cv2
import time
import pathlib

import numpy as np

from io import BytesIO

from typing import Tuple
from typing import Optional
from typing import Coroutine
from typing import Callable
from typing import Any
from common.logger_utils import logger

from common.image_operation import ImageOp
from database.database_client import DatabaseClient

REPO_ROOT = pathlib.Path(__file__).parent.parent.absolute()

MAX_FILE_SIZE = 8000000

DEBUG = os.getenv("POLYTOPIA_DEBUG", 0)


async def get_or_fetch_image_check(
        database_client: DatabaseClient,
        download_fct: Callable[[int], Coroutine[Any, Any, BytesIO]],
        channel_name: str,
        message_id: int,
        filename: str,
        operation: ImageOp) -> bool:
    print("filepath", channel_name, filename, type(channel_name), type(filename), flush=True)
    file_path = get_file_path(channel_name, operation, filename)
    image = cv2.imread(file_path)
    
    if image is None:
        if DEBUG:
            print("Unknown image (%s): %s" % (operation, file_path), flush=True)
        
        if operation == ImageOp.INPUT:
            resource_number = database_client.get_resource_number(filename)
            await save_attachment(download_fct, channel_name, operation, filename, resource_number)
            if DEBUG:
                print("Saved image (%s): %s" % (operation, file_path))
            image = cv2.imread(file_path)
        elif operation == ImageOp.MAP_INPUT or operation == ImageOp.SCORE_INPUT:
            check = await get_or_fetch_image_check(
                database_client,
                download_fct,
                channel_name,
                message_id,
                filename,
                ImageOp.INPUT
            )
            if check:
                move_input_image(channel_name, filename, operation)
                resource_number = database_client.get_resource_number(filename)
                database_client.set_resource_operation(message_id, operation, resource_number)
                image = cv2.imread(file_path)
        elif operation == ImageOp.MAP_PROCESSED_IMAGE:
            check = await get_or_fetch_image_check(
                database_client,
                download_fct,
                channel_name,
                message_id,
                filename,
                ImageOp.MAP_INPUT
            )
            return check
    else:
        if DEBUG:
            print("Existing image (%s): %s" % (operation, file_path))

    return image is not None


def load_image(channel_name: str, filename: str, operation: ImageOp) -> Optional[np.ndarray]:
    file_path = get_file_path(channel_name, operation, filename)
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("None image without retry (%s): %s" % (operation, file_path), flush=True)
    else:
        print("Image valid", len(image))
    return image


async def save_attachment(
        download_fct: Callable[[int], Coroutine[Any, Any, BytesIO]],
        channel_name: str,
        operation: ImageOp,
        filename: str,
        resource_number: int,
        allow_retry: bool = True
) -> None:
    
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = get_file_path(channel_name, operation, filename)
    
    print("paths", parent_path, file_path)
    try:
        print("saving image:", file_path)
        image = await download_fct(resource_number)
        with open(file_path, "wb") as outfile:
            outfile.write(image.getbuffer())
        image.close()
        outfile.close()

    except BaseException as be:
        if allow_retry:
            time.sleep(3)
            await save_attachment(download_fct, channel_name, operation, filename, resource_number, False)
        else:
            raise be


def save_image(image: np.ndarray, channel_name: str, filename: str, operation: ImageOp) -> Optional[str]:
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = get_file_path(channel_name, operation, filename)
    if DEBUG:
        logger.debug("writing image %s: %s" % (file_path, str(image.shape)))
        print("writing image %s: %s" % (file_path, str(image.shape)))
    
    if operation == ImageOp.MAP_PATCHING_OUTPUT:
        is_written = cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        file_size = os.path.getsize(file_path)
        if DEBUG:
            print(f"""Image file size: {file_size}""")
        if file_size > MAX_FILE_SIZE:
            if DEBUG:
                print(f"""Image file size exceeds {MAX_FILE_SIZE} limit: {file_size}""")
            compression_factor = min(file_size / MAX_FILE_SIZE, 0.95) - 0.05  # target: 8Mb - 5%
            compressed_image = cv2.resize(
                image, (int(image.shape[1] * compression_factor), int(image.shape[0] * compression_factor)))
            save_image(compressed_image, channel_name, filename, operation)
    else:
        is_written = cv2.imwrite(file_path, image)
    
    return filename if is_written else None


def move_input_image(channel_name: str, filename: str, target_operation: ImageOp) -> Optional[str]:
    file_path = get_file_path(channel_name, ImageOp.INPUT, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel_name, filename, target_operation)
    else:
        return None


def move_back_input_image(channel_name: str, filename: str, source_operation: ImageOp) -> Optional[str]:
    file_path = get_file_path(channel_name, source_operation, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel_name, filename, ImageOp.INPUT)
    else:
        return None


def __get_parent_path(channel_name: str, operation: ImageOp) -> str:
    return os.path.join(REPO_ROOT, "resources", __clean(channel_name), operation.name)


def get_file_path(channel_name: str, operation: ImageOp, filename: str) -> str:
    return os.path.join(__get_parent_path(channel_name, operation), filename + ".png")


def __get_template(template: str) -> Optional[np.ndarray]:
    return cv2.imread(os.path.join(REPO_ROOT, "templates", template), cv2.IMREAD_UNCHANGED)


def get_cloud_template() -> Optional[np.ndarray]:
    return __get_template("cloud_template_new_2.png")


def get_background_template(map_size: Optional[str]) -> Optional[np.ndarray]:
    if map_size is None or map_size == "0":
        map_size = "400"
    template = __get_template("background_template_%s.png" % map_size)
    if template is not None:
        return template[:, :, 0:3]
    else:
        return None


def get_processed_background_template(map_size: str) -> Optional[np.ndarray]:
    template = __get_template("processed_background_template_%s.png" % map_size)
    if template is not None:
        return template[:, :, 0:3]
    else:
        return None


def set_processed_background(processed_background: np.ndarray, map_size: str) -> Tuple[str, str]:
    filename = "processed_background_template_%s" % map_size
    path = os.path.join(REPO_ROOT, "templates", "%s.png" % filename)
    cv2.imwrite(path, processed_background)
    return path, filename


def get_plt_path(channel_name: str, filename: str) -> Tuple[str, str]:
    parent_path = __get_parent_path(channel_name, ImageOp.SCORE_PLT)
    file_path = get_file_path(channel_name, ImageOp.SCORE_PLT, filename)
    return parent_path, file_path


def __clean(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", path)
