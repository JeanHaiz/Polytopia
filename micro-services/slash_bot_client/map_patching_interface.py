import os

from typing import Optional

from common.image_operation import ImageOp
from database.database_client import DatabaseClient
from slash_bot_client import command_context_store as cxs

from slash_bot_client import service_connector

DEBUG = os.getenv("POLYTOPIA_DEBUG")

database_client = DatabaseClient(
    user="discordBot",
    password="password123",
    port="5432",
    database="polytopiaHelper_dev",
    host="database"
)


def send_map_patching_request(
        patching_id: str,
        number_of_images: Optional[int],
):
    patching_info = database_client.get_patching_process(patching_id)
    author_id = patching_info["process_author_discord_id"]
    player = database_client.get_player(author_id)
    channel_info = database_client.get_channel_info(patching_info["channel_discord_id"])
    ctx = cxs.get(patching_id)
    
    resource_messages = database_client.get_channel_resource_messages(
        patching_info["channel_discord_id"],
        ImageOp.MAP_PROCESSED_IMAGE
    )
    files = [rm["filename"] for rm in resource_messages]
    
    service_connector.send_patch_request(
        patching_id,
        patching_info["channel_discord_id"],
        channel_info["channel_name"],
        author_id,
        player["discord_player_name"],
        channel_info["server_discord_id"],
        patching_info["interaction_id"],
        "0",  # TODO get actual turn
        files,
        number_of_images
    )
