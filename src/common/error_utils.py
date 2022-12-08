import os

from enum import Enum

admin_user_id = os.getenv("DISCORD_ADMIN_USER")


class MapPatchingErrors(Enum):
    SUCCESS = 0
    MISSING_MAP_SIZE = 1
    MISSING_MAP_INPUT = 5
    ATTACHMENT_NOT_LOADED = 2
    ATTACHMENT_NOT_SAVED = 3
    MAP_NOT_RECOGNIZED = 4
    NO_FILE_FOUND = 5
    ONLY_ONE_FILE = 6


MAP_PATCHING_ERROR_MESSAGES = {
    MapPatchingErrors.SUCCESS: "",
    MapPatchingErrors.MISSING_MAP_SIZE:
        "Missing map size. Please use the ´:size 196´ command.",
    MapPatchingErrors.MISSING_MAP_INPUT:
        "We couldn't find your image. <@%s> has been notified." % admin_user_id,
    MapPatchingErrors.ATTACHMENT_NOT_LOADED:
        "We couldn't find our map patching. <@%s> has been notified." % admin_user_id,
    MapPatchingErrors.ATTACHMENT_NOT_SAVED:
        "We couldn't save our map patching. <@%s> has been notified." % admin_user_id,
    MapPatchingErrors.MAP_NOT_RECOGNIZED:
        "The map couldn't be recognised. " +
        "Please try again with another screenshot. To to signal an error, react with ⁉️",
    MapPatchingErrors.NO_FILE_FOUND:
        "No image has been found for your patching. " +
        "Please use the add command from the message menu️ to the image to patch.",
    MapPatchingErrors.ONLY_ONE_FILE:
        "Thank you for saving your first picture. " +
        "When the second map screenshot will be posted, a patching will be generated."
}
