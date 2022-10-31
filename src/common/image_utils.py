import os
import re
import cv2
import sys
import time
import interactions
import pathlib

import numpy as np

from discord import File
from discord import Message
from discord import Attachment
from typing import Tuple
from typing import Optional
from common.logger_utils import logger

from common.image_operation import ImageOp
from database_interaction.database_client import DatabaseClient

REPO_ROOT = pathlib.Path(__file__).parent.parent.absolute()
MAX_FILE_SIZE = 8000000

async def load_or_fetch_image(
        database_client: DatabaseClient,
        channel_name: str,
        message: Optional[Message],
        filename: str,
        operation: ImageOp) -> Optional[np.ndarray]:
    file_path = __get_file_path(channel_name, operation, filename)
    image = cv2.imread(file_path)
    if image is None:
        print("None image (%s): %s" % (operation, file_path))
        if operation == ImageOp.INPUT and message is not None:
            print("reload image")
            resource_number = database_client.get_resource_number(filename)
            await save_attachment(message.attachments[resource_number], channel_name, operation, filename)
            print("image saved", file_path)
            image = cv2.imread(file_path)
        if image is None:
            return None
    else:
        print("Image valid", len(image))
    return image


def load_image(channel_name: str, filename: str, operation: ImageOp) -> np.ndarray:
    file_path = __get_file_path(channel_name, operation, filename)
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("None image without retry (%s): %s" % (operation, file_path))
    else:
        print("Image valid", len(image))
    return image


async def save_attachment(
        attachment: Attachment,
        channel_name: str,
        operation: ImageOp,
        filename: str,
        allow_retry: bool = True) -> None:
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel_name, operation, filename)
    try:
        await attachment.save(file_path)
    except BaseException:
        if allow_retry:
            time.sleep(3)
            await save_attachment(attachment, channel_name, operation, filename, False)
        else:
            logger.error(sys.exc_info()[0])


def load_attachment(file_path: str, filename: str) -> File:
    logger.debug("loading attachment: %s" % file_path)
    print("loading attachment: %s" % file_path)

    with open(file_path, "rb") as fh:
        attachment = interactions.File(fp=fh, filename=filename + ".png")
        fh.close()

    return attachment


def save_image(image: np.ndarray, channel_name: str, filename: str, operation: ImageOp) -> Optional[str]:
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel_name, operation, filename)
    logger.debug("writing image %s: %s" % (file_path, str(image.shape)))
    
    if operation == ImageOp.MAP_PATCHING_OUTPUT:
        is_written = cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            compression_factor = min(file_size / MAX_FILE_SIZE, 0.95) - 0.05  # target: 8Mb - 5%
            compressed_image = cv2.resize(image, (image.shape[0] * compression_factor))
            save_image(compressed_image, channel_name, filename, operation)
    else:
        is_written = cv2.imwrite(file_path, image)
    
    return file_path if is_written else None


def move_input_image(channel_name: str, filename: str, target_operation: ImageOp) -> Optional[str]:
    file_path = __get_file_path(channel_name, ImageOp.INPUT, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel_name, filename, target_operation)
    else:
        return None


def move_back_input_image(channel_name: str, filename: str, source_operation: ImageOp) -> Optional[str]:
    file_path = __get_file_path(channel_name, source_operation, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel_name, filename, ImageOp.INPUT)
    else:
        return None


def __get_parent_path(channel_name: str, operation: ImageOp) -> str:
    return os.path.join(REPO_ROOT, "resources", __clean(channel_name), operation.name)


def __get_file_path(channel_name: str, operation: ImageOp, filename: str) -> str:
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
    # template = load_image("templates", )
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
    file_path = __get_file_path(channel_name, ImageOp.SCORE_PLT, filename)
    return parent_path, file_path


def __clean(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", path)
