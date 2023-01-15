import os
import json
import pika

from pika import exceptions

from common import queue_utils

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

url = "amqp://guest:guest@rabbitmq/"
queue_name = "bot_client"

params = pika.URLParameters(url)
params.socket_timeout = 5

queue_channel = queue_utils.get_blocking_channel(params)
connection = queue_channel.connection

queue_channel.queue_declare(queue=queue_name)  # Declare a queue


def send_recognition_completion(
        process_uuid: str,
        map_requirement_id: str
):
    global queue_channel
    body = json.dumps({
        "action": "MAP_ANALYSIS_COMPLETE",
        "process_uuid": process_uuid,
        "map_requirement_id": map_requirement_id
    })
    try:
        queue_channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body
        )
    except pika.exceptions.StreamLostError:
        queue_channel = queue_utils.get_blocking_channel(params)
        send_recognition_completion(
            process_uuid,
            map_requirement_id
        )


def send_error(
        process_uuid,
        score_requirement_id,
        error: str
):
    global queue_channel
    body = json.dumps({
        "action": "SCORE_RECOGNITION_ERROR",
        "process_uuid": process_uuid,
        "map_requirement_id": score_requirement_id,
        "error": error
    })
    if DEBUG:
        print("Sending error:", body, flush=True)
    try:
        queue_channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body
        )
    except pika.exceptions.StreamLostError:
        queue_channel = queue_utils.get_blocking_channel(params)
        send_error(
            process_uuid,
            score_requirement_id,
            error
        )
