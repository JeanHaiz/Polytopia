import os
import cv2
import sys
import time
import discord
import pathlib
from enum import Enum

from common.logger_utils import logger

# REPO_ROOT = "/Users/jean/Documents/Coding/Polytopia/"
REPO_ROOT = pathlib.Path(__file__).parent.parent.absolute()
print(REPO_ROOT)
INPUT_ROOT = os.path.join(REPO_ROOT, "resources")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")

# os.makedirs(INPUT_ROOT, exist_ok=True)
# os.makedirs(OUTPUT_ROOT, exist_ok=True)


class ImageOp(Enum):
    NONE = 0
    INPUT = 1
    DEBUG_VERTICES = 2
    MAP_STORY = 4
    MAP_PATCHING_OUTPUT = 8
    ONE_COLOR_EDGES = 16
    THREE_COLOR_EDGES = 32
    HOUGH_LINES = 64
    OVERLAY = 128
    MASK = 256
    SCALE = 512
    SCORE_INPUT = 1024
    MAP_INPUT = 2048
    PADDING = 4096
    TURN_PIECES = 8192


async def load_image(database_client, channel_name, message, filename, operation):
    # build path from parent directory
    # filename = database_client.get_resource_filename(message, operation)
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
            return
    else:
        print("Image valid", len(image))
    return image


async def save_attachment(attachment, channel_name, operation, filename, allow_retry=True):
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel_name, operation, filename)
    try:
        await attachment.save(file_path)
    except:
        if allow_retry:
            time.sleep(3)
            await save_attachment(attachment, channel_name, operation, filename, False)
        else:
            logger.error(sys.exc_info()[0])


def load_attachment(file_path, filename):
    logger.debug("loading attachment: %s" % file_path)
    print("loading attachment: %s" % file_path)

    if file_path is None or filename is None:
        return

    with open(file_path, "rb") as fh:
        attachment = discord.File(fh, filename=filename + ".png")
        fh.close()

    return attachment


def save_image(image, channel_name, filename, operation):
    parent_path = __get_parent_path(channel_name, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel_name, operation, filename)
    logger.debug("writing image: %s" % file_path)
    cv2.imwrite(file_path, image)
    return file_path


def move_input_image(channel, filename, target_operation):
    file_path = __get_file_path(channel.name, ImageOp.INPUT, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel.name, filename, target_operation)


def move_back_input_image(channel, filename, source_operation):
    file_path = __get_file_path(channel.name, source_operation, filename)
    image = cv2.imread(file_path)
    if image is not None:
        return save_image(image, channel.name, filename, ImageOp.INPUT)


def __read_img(filename):
    return cv2.imread(os.path.join(INPUT_ROOT, filename))


def __write_img(image, filename, transformation_name):
    split_name = filename.split(".")
    filename_part = split_name[0] if (len(split_name) == 1) else ".".join(split_name[:-1])
    return cv2.imwrite(os.path.join(OUTPUT_ROOT, filename_part + "_" + transformation_name + ".png"), image)


def __get_path(filename):
    return os.path.join(INPUT_ROOT, filename)


def __get_parent_path(channel_name, operation):
    return os.path.join(REPO_ROOT, "resources", channel_name, operation.name)


def __get_file_path(channel_name, operation, filename):
    return os.path.join(__get_parent_path(channel_name, operation), filename + ".png")


def __get_template(template):
    return cv2.imread(os.path.join(REPO_ROOT, "templates", template), cv2.IMREAD_UNCHANGED)


def get_cloud_template():
    return __get_template("cloud_template.png")


def get_background_template(map_size: str):
    print("map size", map_size)
    if map_size is None or map_size == "0":
        map_size = "400"
    return __get_template("background_template_%s.png" % map_size)[:, :, 0:3]
