import os

from common.image_operation import ImageOp
from database.database_client import get_database_client
from slash_bot_client.utils.bot_utils_callbacks import BotUtilsCallbacks
from slash_bot_client.queue_services.sender_service import SenderService

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

database_client = get_database_client()


class MapAnalysisInterface:
    
    def __init__(self, sender_service: SenderService, bot_utils_callbacks: BotUtilsCallbacks):
        self.sender_service = sender_service
        self.bot_utils_callbacks = bot_utils_callbacks
    
    async def get_or_analyse_map(
            self,
            patch_process_id: str,
            map_requirement_id: str,
            message_id: int,
            resource_number: int
    ):
        resource = database_client.get_resource(message_id, resource_number)
    
        operation = resource["operation"]
        channel_id = resource["source_channel_id"]
        channel_info = database_client.get_channel_info(channel_id)
    
        filename = str(resource["filename"])
        
        if DEBUG:
            print("get or analyse map", patch_process_id, map_requirement_id, message_id, flush=True)
            print("processed image", operation, ImageOp.MAP_PROCESSED_IMAGE.value,
                  operation == ImageOp.MAP_PROCESSED_IMAGE.value, flush=True)
            print("processed image", operation, ImageOp.MAP_INPUT.value, operation == ImageOp.MAP_INPUT.value,
                  flush=True)
        if operation == ImageOp.MAP_PROCESSED_IMAGE.value:
            self.bot_utils_callbacks.on_map_analysis_complete(
                patch_process_id,
                map_requirement_id
            )
        elif operation == ImageOp.MAP_INPUT.value:
            self.sender_service.send_map_analysis_request(
                patch_process_id,
                map_requirement_id,
                channel_id,
                channel_info["channel_name"],
                message_id,
                resource_number,
                filename
            )
        else:
            print("operation not recognised", operation, ImageOp.MAP_INPUT,
                  operation == ImageOp.MAP_PROCESSED_IMAGE.value, operation == ImageOp.MAP_INPUT.value, flush=True)
