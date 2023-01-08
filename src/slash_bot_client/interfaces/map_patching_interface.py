import os

from typing import Optional

from common.image_operation import ImageOp
from database.database_client import get_database_client

from slash_bot_client.queue_services.sender_service import SenderService

DEBUG = os.getenv("POLYTOPIA_DEBUG")

database_client = get_database_client()


class MapPatchingInterface:
    
    def __init__(self, sender_service: SenderService):
        self.sender_service = sender_service
    
    def send_map_patching_request(
            self,
            patch_uuid: str,
            number_of_images: Optional[int],
    ):
        patching_info = database_client.get_process(patch_uuid)
        author_id = patching_info["process_author_discord_id"]
        player = database_client.get_player(author_id)
        channel_info = database_client.get_channel_info(patching_info["channel_discord_id"])
        
        resource_messages = database_client.get_channel_resource_messages(
            patching_info["channel_discord_id"],
            ImageOp.MAP_PROCESSED_IMAGE
        )
        files = [rm["filename"] for rm in resource_messages]
        
        self.sender_service.send_map_patch_request(
            patch_uuid,
            patching_info["channel_discord_id"],
            channel_info["channel_name"],
            author_id,
            player["discord_player_name"],
            channel_info["server_discord_id"],
            patching_info["interaction_id"],
            files,
            number_of_images
        )
