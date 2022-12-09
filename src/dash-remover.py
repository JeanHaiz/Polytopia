import os
import re
import shutil


def get_dirs(path):
    return [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]


def remove_dashes():
    root = "/Users/jean/Documents/Coding/Polytopia/src/resources"
    dirs = get_dirs(root)
    for di in dirs:
        source = os.path.join(root, di)
        target = os.path.join(root, re.sub(r"[^a-zA-Z0-9]", "", di))
        
        if source != target:
            if len(os.listdir(source)) == 0:
                os.rmdir(source)
            else:
                print(re.sub(r"[^a-zA-Z0-9]", "", di), source, target)
                # os.rename(source, target)
                for file in os.listdir(source):
                    shutil.move(os.path.join(source, file), target)

remove_dashes()