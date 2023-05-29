from database.database_client import get_database_client

from slash_bot_client.queue_services.sender_service import SenderService

database_client = get_database_client()


class ScoreVisualisationInterface:
    
    def __init__(self, sender_service: SenderService):
        self.sender_service = sender_service
    
    def get_or_visualise_scores(
            self,
            process_uuid: str,
            author_id: int,
            channel_id: int,
            channel_name: str
    ):
        
        self.sender_service.send_score_visualisation_request(
            process_uuid,
            channel_id,
            channel_name,
            author_id,
        )
