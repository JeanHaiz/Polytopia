from enum import Enum
from interactions import Channel
from interactions import CommandContext

from database.database_client import DatabaseClient


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
        "We couldn't find your image. <@338067113639936003> has been notified.",
    MapPatchingErrors.ATTACHMENT_NOT_LOADED:
        "We couldn't find our map patching. <@338067113639936003> has been notified.",
    MapPatchingErrors.ATTACHMENT_NOT_SAVED:
        "We couldn't save our map patching. <@338067113639936003> has been notified.",
    MapPatchingErrors.MAP_NOT_RECOGNIZED:
        "The map couldn't be recognised." +
        "Please try again with another screenshot. To to signal an error, react with ⁉️",
    MapPatchingErrors.NO_FILE_FOUND:
        "No image has been found for your patching. Please add 🖼️ to the image to patch.",
    MapPatchingErrors.ONLY_ONE_FILE:
        "Thank you for saving your first picture. " +
        "When the second map screenshot will be posted, a patching will be generated."
}


async def manage_slash_patching_errors(
        database_client: DatabaseClient,
        channel: Channel,
        ctx: CommandContext,
        patching_errors: list
) -> None:
    if patching_errors is not None and len(patching_errors) > 0:
        error_text = []
        for cause, error_filename in patching_errors:
            if error_filename is None:
                error_text.append(MAP_PATCHING_ERROR_MESSAGES[cause])
            else:
                channel_id, message_id = database_client.get_resource_message(error_filename)
                if channel_id is not None and message_id is not None:
                    message = await channel.get_message(message_id)
                    if message is None:
                        error_text.append(MAP_PATCHING_ERROR_MESSAGES[cause])
                    else:
                        await message.reply(MAP_PATCHING_ERROR_MESSAGES[cause])
        my_id = '<@338067113639936003>'  # Jean's id
        error_text.append('%s has been notified.' % my_id)
        await ctx.send("\n".join(error_text))
