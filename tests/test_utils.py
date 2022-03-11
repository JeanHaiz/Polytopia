import os
import cv2
import numpy
import pathlib

from common import image_utils
from map_patching.map_patching_utils import ImageOp


REPO_ROOT = pathlib.Path(__file__).parent.absolute()


def get_resource(folder: str, filename: str) -> numpy.ndarray:
    path = os.path.join(REPO_ROOT, "resources", folder, filename)
    return cv2.imread(path)


def get_score_resource(filename: str) -> numpy.ndarray:
    return get_resource("score_recognition", filename + ".png")


def get_map_resource(filename: str) -> numpy.ndarray:
    return get_resource("map_patching", filename + ".png")


async def prepare_test_images(files, channel_name, map_size):
    images = [
        await image_utils.load_image(None, channel_name, None, filename_i, ImageOp.INPUT)
        for filename_i in files]
    images.insert(0, image_utils.get_background_template(map_size))
    files.insert(0, "background")
    return images
