from database.database_client import get_database_client

from slash_bot_client.queue_services.sender_service import SenderService

database_client = get_database_client()


class HeaderFooterRecognitionInterface:
    
    def __init__(self, sender_service: SenderService, bot_utils_callbacks):
        self.sender_service = sender_service
        self.bot_utils_callbacks = bot_utils_callbacks
        
    @staticmethod
    def __get_turn(channel_name: str) -> int:
        return 0
    
    async def get_or_recognise_turn(
            self,
            patch_id: str,
            turn_requirement_uuid: str,
            message_id: int,
            resource_number: int
    ):
        filename = database_client.set_filename_header(message_id, resource_number, 0)  # TODO actually call the routine
        self.bot_utils_callbacks.on_turn_recognition_complete(patch_id, turn_requirement_uuid)
