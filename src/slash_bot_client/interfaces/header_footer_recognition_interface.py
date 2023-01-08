from database.database_client import get_database_client

from slash_bot_client.queue_services.sender_service import SenderService

database_client = get_database_client()


class HeaderFooterRecognitionInterface:
    
    def __init__(self, sender_service: SenderService, bot_utils_callbacks):
        self.sender_service = sender_service
        self.bot_utils_callbacks = bot_utils_callbacks
    
    async def get_or_recognise_turn(
            self,
            patch_uuid: str,
            turn_requirement_id: str,
            message_id: int,
            resource_number: int
    ):
        resource = database_client.get_resource(message_id, resource_number)
        header = database_client.get_filename_header(resource["filename"])
        
        if header is None or header["turn_value"] is None:
            self.bot_utils_callbacks.on_turn_recognition_complete(
                patch_uuid,
                turn_requirement_id
            )
        else:
            channel_id = resource["source_channel_id"]
            channel_info = database_client.get_channel_info(channel_id)
            
            self.sender_service.send_turn_recognition_request(
                patch_uuid,
                turn_requirement_id,
                channel_id,
                channel_info["channel_name"],
                message_id,
                resource_number,
                str(resource["filename"])
            )
