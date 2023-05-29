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
from typing import Dict

from dataclasses import dataclass

from common.logger_utils import logger

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


@dataclass
class ReceiverParams:
    action_fct: Callable
    error_fct: Union[Callable[[str, str], None], Callable[[str, str, str], None]]
    callback_function: Union[
        Callable[[str, str], None],
        Callable[[str, str, str], None],
        Callable[[str, int, str], None]
    ]
    expected_exception: Type
    param_keys: List[str]
    

class Receiver:
    
    def __init__(
            self,
            queue_name: str,
            username: str,
            password: str,
            receiver_params_dict: Dict[str, ReceiverParams]
    ):
        self.queue_name = queue_name
        self.username = username
        self.password = password
        self.receiver_params_dict = receiver_params_dict
    
    def perform_request(self, body, action):
        receiver_param = self.receiver_params_dict[action]
        # print("received analysis callback", flush=True)
        params = json.loads(body)
        params.pop("action", "")
        try:
            receiver_param.action_fct(**params, callback=receiver_param.callback_function)
        except receiver_param.expected_exception as e:
            if DEBUG:
                print("Catching ERROR:", e.__class__.__name__ + ":", body)
                logger.warning("Catching ERROR:", e.__class__.__name__ + ":", body)
            extracted_params = {
                key: params[key] if key in params else None for key in receiver_param.param_keys
            }
            extracted_params["error"] = e.message
            if DEBUG:
                trace = e.__class__.__name__ + " details:\n" + "\n".join(
                    [k + ": " + v for k, v in extracted_params.items()]
                )
                print("Detailed ERROR:", trace, flush=True)
                logger.warning(trace)
            receiver_param.error_fct(
                **extracted_params
            )
        except BaseException as e:
            extracted_params = {
                key: params[key] if key in params else None for key in receiver_param.param_keys
            }
            extracted_params["error"] = traceback.format_exc()
            trace = "Catching base exception ERROR:" + e.__class__.__name__ + " details:\n" + "\n".join(
                [k + ": " + v for k, v in extracted_params.items()]
            )
            print(trace, flush=True)
            logger.warning(trace)
            receiver_param.error_fct(
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
        print('Thread id: %s Delivery tag: %s Message body: %s' % (thread_id, delivery_tag, body), flush=True)
        
        params = json.loads(body)
        
        self.perform_request(body, params["action"])
        
        cb = functools.partial(self.__ack_message, ch, delivery_tag)
        conn.add_callback_threadsafe(cb)
    
    def on_message(self, ch, method_frame, _header_frame, body, args):
        params = json.loads(body)
        (conn, thrds) = args
        delivery_tag = method_frame.delivery_tag
        if "ping" in params:
            cb = functools.partial(self.__ack_message, ch, delivery_tag)
            conn.add_callback_threadsafe(cb)
        else:
            t = threading.Thread(target=self.do_work, args=(conn, ch, delivery_tag, body))
            t.start()
            thrds.append(t)
    
    def run(self):
        # url = f"""amqp://{self.username}:{self.password}@rabbitmq:5672/vhost"""
        # params = pika.URLParameters(url)
        # params.socket_timeout = 5
        
        params = pika.ConnectionParameters(
            host='rabbitmq',
            virtual_host="vhost",
            credentials=pika.credentials.PlainCredentials(self.username, self.password),
            heartbeat=0,
            port=5672,
            socket_timeout=5
        )
        
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
