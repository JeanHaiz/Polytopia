import json
import pika

from pika import exceptions

from slash_bot_client.queue_services.queue_service import QueueService

url = 'amqp://slash_bot:slash_bot123@rabbitmq:5672/vhost'

params = pika.URLParameters(url)
params.socket_timeout = 5


class SenderService:
    
    def __init__(self, queue_service: QueueService):
        self.queue_service = queue_service
        self.__init_queues()

    def __init_queues(self):
        queues = [
            "map_patching",
            "map_analysis",
            "score_recognition",
            "score_visualisation",
            "header_footer_recognition"
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
                self.queue_service.send_message("map_analysis", body)
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
            self.queue_service.send_message(
                queue_name="map_patching",
                body=body
            )
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
                self.queue_service.send_message("header_footer_recognition", body)
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
        # TODO complete
    ):
        body = json.dumps({
            # TODO complete
        })
        try:
            self.queue_service.send_message(
                queue_name="score_recognition",
                body=body
            )
        except pika.exceptions.StreamLostError:
            self.queue_service.reset_queues()
            self.send_score_recognition_request(
                # TODO complete
            )
