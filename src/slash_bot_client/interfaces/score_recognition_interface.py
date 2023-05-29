from database.database_client import get_database_client

from slash_bot_client.queue_services.sender_service import SenderService

database_client = get_database_client()


class ScoreRecognitionInterface:
    
    def __init__(self, sender_service: SenderService):
        self.sender_service = sender_service
    
    async def get_or_recognise_score(
            self,
            process_uuid: str,
            score_requirement_id: str,
            author_name: str,
            message_id: int,
            resource_number: int
    ):
        
        resource = database_client.get_resource(message_id, resource_number)
        
        channel_id = resource["source_channel_id"]
        channel_info = database_client.get_channel_info(channel_id)
        
        self.sender_service.send_score_recognition_request(
            process_uuid,
            score_requirement_id,
            channel_id,
            channel_info["channel_name"],
            author_name,
            str(resource["filename"])
        )
