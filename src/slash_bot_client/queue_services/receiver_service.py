import json
from asyncio import Task

import pika
import asyncio
import threading
import nest_asyncio

from slash_bot_client.utils import bot_utils_callbacks

nest_asyncio.apply()

HOST = "rabbitmq"
QUEUE_NAME = "bot_client"


class RabbitmqReceive(threading.Thread):
    def __init__(self, queue_name, callback):
        threading.Thread.__init__(self)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="rabbitmq")
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue_name)
        self.consumer = self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True
        )
        self.channel.add_on_return_callback(
            lambda ch, mt, pr, bd: print("Channel returned", ch, mt, pr, bd, flush=True)
        )
        self.channel.add_on_cancel_callback(
            lambda mf: print("Channel cancelled", mf, flush=True)
        )
    
    def run(self):
        self.channel.start_consuming()


def _to_task(future, as_task, loop):
    if not as_task or isinstance(future, Task):
        return future
    return loop.create_task(future)


async def get_async_connection(queue, client, loop: asyncio.AbstractEventLoop):
    def action_reaction_request(channel, method, properties, body):

        def run_async(fct, **xargs):
            loop.call_soon_threadsafe(
                lambda: loop.run_until_complete(fct(**xargs)))
        
        print("action received", body)
        action_params: dict = json.loads(body)
        action = action_params.pop("action", "")
        if action == "MAP_ANALYSIS_COMPLETE":
            bot_utils_callbacks.on_map_analysis_complete(**action_params)
        elif action == "MAP_PATCHING_COMPLETE":
            run_async(bot_utils_callbacks.on_map_patching_complete, client=client, **action_params)
        elif action == "HEADER_RECOGNITION_COMPLETE":
            bot_utils_callbacks.on_turn_recognition_complete(**action_params)
        elif action == "MAP_ANALYSIS_ERROR":
            run_async(bot_utils_callbacks.on_analysis_error, client=client, **action_params)
        elif action == "MAP_PATCHING_ERROR":
            run_async(bot_utils_callbacks.on_patching_error, client=client, **action_params)
    
    url = "amqp://guest:guest@rabbitmq:5672/"
    
    params = pika.URLParameters(url)
    params.socket_timeout = 5
    
    rabbit_receive = RabbitmqReceive(queue, action_reaction_request)
    rabbit_receive.start()
    print("Slash bot message queue listener is running", flush=True)


"""
def ack_message(ch, delivery_tag):
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def do_work(conn, ch, delivery_tag, body):
    thread_id = threading.get_ident()
    print('Thread id: %s Delivery tag: %s Message body: %s', thread_id,
                delivery_tag, body)
    action_reaction_request(body)
    cb = functools.partial(ack_message, ch, delivery_tag)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


credentials = pika.PlainCredentials('guest', 'guest')
# Note: sending a short heartbeat to prove that heartbeats are still
# sent even though the worker simulates long-running work
parameters = pika.ConnectionParameters(
    host=HOST, credentials=credentials, heartbeat=5)

queue_channel.queue_declare(queue=QUEUE_NAME)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
queue_channel.basic_consume(QUEUE_NAME, on_message_callback)

try:
    queue_channel.start_consuming()
except KeyboardInterrupt:
    queue_channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

"""
