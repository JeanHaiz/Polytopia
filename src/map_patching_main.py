import os
import functools
import logging
import threading
import json
import pika
import traceback

from map_patching import map_patching, patching_callback_utils
from map_patching.map_patching_error import PatchingException

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

HOST = "rabbitmq"
QUEUE_NAME = "map_patching"


def ack_message(ch, delivery_tag):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def send_patching_request(body):
    action_params = json.loads(body)
    
    try:
        map_patching.generate_patched_map_bis(**action_params)
    except PatchingException as e:
        patch_process_id = action_params["patch_process_id"] if "patch_process_id" in action_params else None
        if DEBUG:
            print("Analysis error details:\npatch_process_id: %s\nmessage: %s" %
                  (patch_process_id, e.message))
        patching_callback_utils.send_error(
            patch_process_id,
            e.message
        )
    except BaseException:
        patch_process_id = action_params["patch_process_id"] if "patch_process_id" in action_params else None
        if DEBUG:
            print("Analysis error details:\npatch_process_id: %s\nmessage: %s" %
                  (patch_process_id, traceback.format_exc()))
        patching_callback_utils.send_error(
            patch_process_id,
            traceback.format_exc()
        )


def do_work(conn, ch, delivery_tag, body):
    thread_id = threading.get_ident()
    print('Thread id: %s Delivery tag: %s Message body: %s' % (thread_id, delivery_tag, body))
    
    send_patching_request(body)
    
    cb = functools.partial(ack_message, ch, delivery_tag)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


url = 'amqp://guest:guest@rabbitmq:5672/'
params = pika.URLParameters(url)
params.socket_timeout = 5
connection = pika.BlockingConnection(params)

channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)
channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume(QUEUE_NAME, on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()
