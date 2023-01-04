import os
import json
import pika

from pika import exceptions

from common import queue_utils

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


class Sender:
    
    def __init__(
            self,
            queue_name: str,
            username: str,
            password: str,
    ):
        url = f"""amqp://{username}:{password}@rabbitmq/vhost"""
        self.queue_name = queue_name
        
        self.params = pika.URLParameters(url)
        self.params.socket_timeout = 5
        
        self.queue_channel = queue_utils.get_blocking_channel(self.params)
        self.queue_channel.queue_declare(queue=self.queue_name)

    def send_message(
            self,
            params: dict[str]
    ) -> None:
        body = json.dumps(params)
        try:
            self.queue_channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=body
            )
        except pika.exceptions.StreamLostError:
            self.queue_channel = queue_utils.get_blocking_channel(self.params)
            self.send_message(params)
