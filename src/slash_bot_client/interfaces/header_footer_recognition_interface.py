from slash_bot_client.utils import bot_utils_callbacks
from database.database_client import get_database_client

database_client = get_database_client()


def __get_turn(image, channel_name: str) -> int:
    return 0


async def get_or_recognise_turn(patch_id, turn_requirement_uuid, message_id, resource_number):
    filename = database_client.set_filename_header(message_id, resource_number, 0)  # TODO actually call the routine
    bot_utils_callbacks.on_turn_recognition_complete(patch_id, turn_requirement_uuid)
