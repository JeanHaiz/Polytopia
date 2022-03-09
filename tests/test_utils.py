import os
import cv2
import numpy
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.absolute()


def get_resource(folder: str, filename: str) -> numpy.ndarray:
    path = os.path.join(REPO_ROOT, "resources", folder, filename)
    return cv2.imread(path)


def get_score_resource(filename: str) -> numpy.ndarray:
    return get_resource("score_recognition", filename + ".png")


def get_map_resource(filename: str) -> numpy.ndarray:
    return get_resource("map_patching", filename + ".png")
