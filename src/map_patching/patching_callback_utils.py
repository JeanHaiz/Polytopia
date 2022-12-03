import json
import pika


from pika import exceptions

from common import queue_utils

url = "amqp://guest:guest@rabbitmq/"
queue_name = "bot_client"

params = pika.URLParameters(url)
params.socket_timeout = 5

queue_channel = queue_utils.get_blocking_channel(params)
connection = queue_channel.connection

queue_channel.queue_declare(queue=queue_name)


def send_patching_completion(
        patch_id: str,
        channel_id,
        filename
):
    global queue_channel
    body = json.dumps({
        "action": "MAP_PATCHING_COMPLETE",
        "patch_uuid": patch_id,
        "channel_id": channel_id,
        "filename": filename
    })
    try:
        queue_channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body
        )
    except pika.exceptions.StreamLostError:
        queue_channel = queue_utils.get_blocking_channel(params)
        send_patching_completion(
            patch_id,
            channel_id,
            filename
        )
