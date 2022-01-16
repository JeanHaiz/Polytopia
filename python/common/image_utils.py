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


class ImageOperation(Enum):
    NONE = 0
    INPUT = 1
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


async def load_image(database_client, message, filename, operation):
    # build path from parent directory
    # filename = database_client.get_resource_filename(message, operation)
    file_path = __get_file_path(message.channel, operation, filename)
    image = cv2.imread(file_path)
    if image is None:
        print("None image (%s): %s" % (operation, file_path))
        if operation == ImageOperation.INPUT:
            print("reload image")
            resource_number = database_client.get_resource_number(filename)
            await save_attachment(message.attachments[resource_number], message, operation, filename)
            print("image saved", file_path)
            image = cv2.imread(file_path)
    else:
        print("Image valid", len(image))
    return image


async def save_attachment(attachment, message, operation, filename, allow_retry=True):
    parent_path = __get_parent_path(message.channel, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(message.channel, operation, filename)
    try:
        await attachment.save(file_path)
    except:
        if allow_retry:
            time.sleep(3)
            save_attachment(attachment, message, operation, filename, False)
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


def save_image(image, channel, filename, operation):
    parent_path = __get_parent_path(channel, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel, operation, filename)
    logger.debug("writing image: %s" % file_path)
    cv2.imwrite(file_path, image)
    return file_path


def move_input_image(channel, filename, target_operation):
    file_path = __get_file_path(channel, ImageOperation.INPUT, filename)
    image = cv2.imread(file_path)
    save_image(image, channel, filename, target_operation)


def __read_img(filename):
    return cv2.imread(os.path.join(INPUT_ROOT, filename))


def __write_img(image, filename, transformation_name):
    split_name = filename.split(".")
    filename_part = split_name[0] if (len(split_name) == 1) else ".".join(split_name[:-1])
    return cv2.imwrite(os.path.join(OUTPUT_ROOT, filename_part + "_" + transformation_name + ".png"), image)


def __get_path(filename):
    return os.path.join(INPUT_ROOT, filename)


def __get_parent_path(channel, operation):
    return os.path.join(REPO_ROOT, "resources", channel.name, operation.name)


def __get_file_path(channel, operation, filename):
    return os.path.join(__get_parent_path(channel, operation), filename + ".png")


def __get_template(template):
    return cv2.imread(os.path.join(REPO_ROOT, "templates", template), cv2.IMREAD_UNCHANGED)


def get_cloud_template():
    return __get_template("cloud_template.png")


def get_background_template():
    return __get_template("image32.png")[:, :, 0:3]
