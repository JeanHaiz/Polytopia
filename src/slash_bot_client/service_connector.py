import logging
import json
import pika

from pika import exceptions

from common import queue_utils

logging.basicConfig()

url = 'amqp://guest:guest@rabbitmq:5672/'
queue_name = "map_patching"

params = pika.URLParameters(url)
params.socket_timeout = 5


queue_channel = queue_utils.get_blocking_channel(params)

queues = ["map_patching", "map_analysis"]
for queue_name in queues:
    queue_channel.queue_declare(queue=queue_name)  # Declare a queue


def send_analysis_request(
        patch_process_id: str,
        map_requirement_id: str,
        channel_id: int,
        channel_name: str,
        message_id: int,
        resource_number: int,
        filename: str
):
    global queue_channel
    body = json.dumps({
        "patch_process_id": patch_process_id,
        "map_requirement_id": map_requirement_id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "message_id": message_id,
        "resource_number": resource_number,
        "filename": filename
    })
    try:
        queue_channel.basic_publish(
            exchange='',
            routing_key="map_analysis",
            body=body
        )
    except pika.exceptions.StreamLostError:
        queue_channel = queue_utils.get_blocking_channel(params)
        send_analysis_request(
            patch_process_id,
            map_requirement_id,
            channel_id,
            channel_name,
            message_id,
            resource_number,
            filename
        )


def send_patch_request(
        patching_id,
        channel_id,
        channel_name,
        author_id,
        author_name,
        server_id,
        interaction_id,
        files,
        number_of_images
):
    global queue_channel
    body = json.dumps({
        "patching_process_id": patching_id,
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
        queue_channel.basic_publish(
            exchange='',
            routing_key="map_patching",
            body=body
        )
    except pika.exceptions.StreamLostError:
        queue_channel = queue_utils.get_blocking_channel(params)
        send_patch_request(
            patching_id,
            channel_id,
            channel_name,
            author_id,
            author_name,
            server_id,
            interaction_id,
            files,
            number_of_images
        )
