import os

from typing import Optional

from common.image_operation import ImageOp
from database.database_client import get_database_client

from slash_bot_client.queue_services import sender_service

DEBUG = os.getenv("POLYTOPIA_DEBUG")

database_client = get_database_client()


def send_map_patching_request(
        patching_id: str,
        number_of_images: Optional[int],
):
    patching_info = database_client.get_patching_process(patching_id)
    author_id = patching_info["process_author_discord_id"]
    player = database_client.get_player(author_id)
    channel_info = database_client.get_channel_info(patching_info["channel_discord_id"])
    
    resource_messages = database_client.get_channel_resource_messages(
        patching_info["channel_discord_id"],
        ImageOp.MAP_PROCESSED_IMAGE
    )
    files = [rm["filename"] for rm in resource_messages]
    
    sender_service.send_map_patch_request(
        patching_id,
        patching_info["channel_discord_id"],
        channel_info["channel_name"],
        author_id,
        player["discord_player_name"],
        channel_info["server_discord_id"],
        patching_info["interaction_id"],
        files,
        number_of_images
    )
