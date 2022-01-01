import os
import cv2
import discord
import pathlib
from enum import Enum


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


def load_image(channel, filename, operation):
    # build path from parent directory
    # filename = database_client.get_resource_filename(message, operation)
    file_path = __get_file_path(channel, operation, filename)
    image = cv2.imread(file_path)
    if image is None:
        print("None image", file_path)
    else:
        print("Image valid", len(image))
    return image


async def save_attachment(attachment, message, operation, filename):
    parent_path = __get_parent_path(message.channel, operation)
    os.makedirs(parent_path, exist_ok=True)
    await attachment.save(__get_file_path(message.channel, operation, filename))


def load_attachment(file_path, filename):
    with open(file_path, "rb") as fh:
        attachment = discord.File(fh, filename=filename + ".png")
    return attachment


def save_image(image, channel, filename, operation):
    parent_path = __get_parent_path(channel, operation)
    os.makedirs(parent_path, exist_ok=True)
    file_path = __get_file_path(channel, operation, filename)
    cv2.imwrite(file_path, image)
    return file_path


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
