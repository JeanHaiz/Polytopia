import json
import pika

from pika import exceptions

from slash_bot_client.queue_services.queue_service import QueueService


class SenderService:
    
    def __init__(self, queue_service: QueueService):
        self.queue_service = queue_service
        self.__init_queues()
    
    def __init_queues(self):
        queues = [
            "worker"
            # "map_patching",
            # "map_analysis",
            # "score_recognition",
            # "score_visualisation",
            # "header_footer_recognition"
        ]
        for queue_name in queues:
            self.queue_service.declare_queue(queue_name)
    
    def send_map_analysis_request(
            self,
            process_uuid: str,
            map_requirement_id: str,
            channel_id: int,
            channel_name: str,
            message_id: int,
            resource_number: int,
            filename: str
    ):
        body = json.dumps({
            "action": "MAP_ANALYSIS",
            "process_uuid": process_uuid,
            "map_requirement_id": map_requirement_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "message_id": message_id,
            "resource_number": resource_number,
            "filename": filename
        })
        try:
            if self.queue_service.is_open():
                self.queue_service.send_message("worker", body=body)
            else:
                print("CONNECTION STATUS", self.queue_service.is_open())
                self.queue_service.reset_queues()
                self.send_map_analysis_request(
                    process_uuid,
                    map_requirement_id,
                    channel_id,
                    channel_name,
                    message_id,
                    resource_number,
                    filename
                )
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_map_analysis_request(
                process_uuid,
                map_requirement_id,
                channel_id,
                channel_name,
                message_id,
                resource_number,
                filename
            )
    
    def send_map_patch_request(
            self,
            process_uuid,
            channel_id,
            channel_name,
            author_id,
            author_name,
            server_id,
            interaction_id,
            files,
            number_of_images
    ):
        body = json.dumps({
            "action": "MAP_PATCHING",
            "process_uuid": process_uuid,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "author_id": author_id,
            "author_name": author_name,
            "guild_id": server_id,
            "interaction_id": interaction_id,
            "files": files,
            "n_images": number_of_images
        })
        try:
            self.queue_service.send_message(queue_name="worker", body=body)
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_map_patch_request(
                process_uuid,
                channel_id,
                channel_name,
                author_id,
                author_name,
                server_id,
                interaction_id,
                files,
                number_of_images
            )
    
    def send_turn_recognition_request(
            self,
            process_uuid: str,
            turn_requirement_id: str,
            channel_id: int,
            channel_name: str,
            message_id: int,
            resource_number: int,
            filename: str
    ):
        body = json.dumps({
            "action": "TURN_RECOGNITION",
            "process_uuid": process_uuid,
            "turn_requirement_id": turn_requirement_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "message_id": message_id,
            "resource_number": resource_number,
            "filename": filename
        })
        try:
            if self.queue_service.is_open():
                self.queue_service.send_message("worker", body=body)
            else:
                print("CONNECTION STATUS", self.queue_service.is_open())
                self.queue_service.reset_queues()
                self.send_map_analysis_request(
                    process_uuid,
                    turn_requirement_id,
                    channel_id,
                    channel_name,
                    message_id,
                    resource_number,
                    filename
                )
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_map_analysis_request(
                process_uuid,
                turn_requirement_id,
                channel_id,
                channel_name,
                message_id,
                resource_number,
                filename
            )
    
    def send_score_recognition_request(
            self,
            process_uuid: str,
            score_requirement_id: str,
            channel_id: int,
            channel_name: str,
            author_name: str,
            filename: str,
    ):
        body = json.dumps({
            "action": "SCORE_RECOGNITION",
            "process_uuid": process_uuid,
            "score_requirement_id": score_requirement_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "author_name": author_name,
            "filename": filename
        })
        try:
            self.queue_service.send_message(queue_name="worker", body=body)
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_score_recognition_request(
                process_uuid,
                score_requirement_id,
                channel_id,
                channel_name,
                author_name,
                filename,
            )

    def send_score_visualisation_request(
            self,
            process_uuid: str,
            channel_id: int,
            channel_name: str,
            author_id: int,
    ):
        body = json.dumps({
            "action": "SCORE_VISUALISATION",
            "process_uuid": process_uuid,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "author_id": author_id,
        })
        try:
            self.queue_service.send_message(queue_name="worker", body=body)
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_score_visualisation_request(
                process_uuid,
                channel_id,
                channel_name,
                author_id,
            )
