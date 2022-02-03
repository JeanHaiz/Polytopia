import os
import cv2
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent.absolute()


def get_resource(folder, filename):
    print(REPO_ROOT)
    path = os.path.join(REPO_ROOT, "resources", folder, filename)
    return cv2.imread(path)


def get_score_resource(filename):
    return get_resource("score_recognition", filename + ".png")
