import os
import time
import json
import pika
import sys

from common import queue_utils
from map_patching import map_patching


def callback(ch, method, properties, body):
    print("received patching callback", flush=True)
    params = json.loads(body)
    map_patching.generate_patched_map_bis(**params)
    

def main():
    time.sleep(30)
    
    url = "amqp://guest:guest@rabbitmq/"
    queue_name = "map_patching"
    
    params = pika.URLParameters(url)
    params.socket_timeout = 5
    
    connection, channel = queue_utils.get_channel(params)

    channel.queue_declare(queue=queue_name)

    channel.basic_consume(queue='', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
