import os
import json
import pika
import functools
import threading
import traceback

from typing import Callable
from typing import Union
from typing import List
from typing import Type

from common.logger_utils import logger
DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


class Receiver:
    
    def __init__(
            self,
            queue_name: str,
            action_fct: Callable,
            error_fct: Union[Callable[[str, str], None], Callable[[str, str, str], None]],
            expected_exception: Type,
            param_keys: List[str]
    ):
        self.queue_name = queue_name
        self.error_fct = error_fct
        self.action_fct = action_fct
        self.expected_exception = expected_exception
        self.param_keys = param_keys
    
    def perform_request(self, body):
        # print("received analysis callback", flush=True)
        params = json.loads(body)
        try:
            self.action_fct(**params)
        except self.expected_exception as e:
            if DEBUG:
                print(e.__class__.__name__ + ":", body)
            extracted_params = {
                key: params[key] if key in params else None for key in self.param_keys
            }
            extracted_params["error"] = e.message
            if DEBUG:
                trace = e.__class__.__name__ + " details:\n" + "\n".join(
                    [k + ": " + v for k, v in extracted_params.items()]
                )
                print(trace, flush=True)
                logger.warning(trace)
            self.error_fct(
                **extracted_params
            )
        except BaseException as e:
            extracted_params = {
                key: params[key] if key in params else None for key in self.param_keys
            }
            extracted_params["error"] = traceback.format_exc()
            trace = e.__class__.__name__ + " details:\n" + "\n".join(
                [k + ": " + v for k, v in extracted_params.items()]
            )
            print(trace, flush=True)
            logger.warning(trace)
            self.error_fct(
                **extracted_params
            )
    
    @staticmethod
    def __ack_message(ch, delivery_tag):
        """Note that `ch` must be the same pika channel instance via which
        the message being ACKed was retrieved (AMQP protocol constraint).
        """
        if ch.is_open:
            ch.basic_ack(delivery_tag)
        else:
            # Channel is already closed, so we can't ACK this message;
            # log and/or do something that makes sense for your app in this case.
            pass
    
    def do_work(self, conn, ch, delivery_tag, body):
        thread_id = threading.get_ident()
        print('Thread id: %s Delivery tag: %s Message body: %s' % (thread_id, delivery_tag, body))
        
        self.perform_request(body)
        
        cb = functools.partial(self.__ack_message, ch, delivery_tag)
        conn.add_callback_threadsafe(cb)
    
    def on_message(self, ch, method_frame, _header_frame, body, args):
        (conn, thrds) = args
        delivery_tag = method_frame.delivery_tag
        t = threading.Thread(target=self.do_work, args=(conn, ch, delivery_tag, body))
        t.start()
        thrds.append(t)
    
    def run(self):
        url = "amqp://guest:guest@rabbitmq:5672/"
        
        params = pika.URLParameters(url)
        params.socket_timeout = 5
        
        connection = pika.BlockingConnection(params)
        
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name)
        channel.basic_qos(prefetch_count=1)
        
        threads = []
        on_message_callback = functools.partial(self.on_message, args=(connection, threads))
        channel.basic_consume(queue=self.queue_name, on_message_callback=on_message_callback)
        
        print(' [*] Waiting for messages. To exit press CTRL+C', flush=True)
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        connection.close()
