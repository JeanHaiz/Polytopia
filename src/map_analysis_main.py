import time
import json
import pika
import sys
import functools
import threading

from common import queue_utils
from map_analysis import map_analysis


def send_analysis_request(body):
    print("received analysis callback", flush=True)
    params = json.loads(body)
    map_analysis.map_analysis_request(**params)


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


def do_work(conn, ch, delivery_tag, body):
    thread_id = threading.get_ident()
    print('Thread id: %s Delivery tag: %s Message body: %s' % (thread_id, delivery_tag, body))
    
    send_analysis_request(body)
    
    cb = functools.partial(ack_message, ch, delivery_tag)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


def main():
    
    url = "amqp://guest:guest@rabbitmq:5672/"
    queue_name = "map_analysis"
    
    params = pika.URLParameters(url)
    params.socket_timeout = 5

    connection = pika.BlockingConnection(params)

    channel = connection.channel()
    channel.queue_declare(queue=queue_name)

    threads = []
    on_message_callback = functools.partial(on_message, args=(connection, threads))
    channel.basic_consume(queue=queue_name, on_message_callback=on_message_callback)

    print(' [*] Waiting for messages. To exit press CTRL+C', flush=True)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    connection.close()


try:
    main()
except KeyboardInterrupt:
    print('Interrupted')
    sys.exit(0)
