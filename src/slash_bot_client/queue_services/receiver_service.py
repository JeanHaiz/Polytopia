import os
import threading
import nest_asyncio

from typing import Callable
from typing import Any

from slash_bot_client.queue_services.queue_service import QueueService

nest_asyncio.apply()

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))


class RabbitmqReceive(threading.Thread):
    def __init__(
            self,
            queue_name: str,
            callback: Callable[[Any, Any, Any, Any], None]
    ):
        threading.Thread.__init__(self)
        self.queue_service = QueueService()
        self.queue_service.declare_queue(queue_name)
        self.queue_service.consume(queue_name, on_message_callback=callback, auto_ack=True)
    
    def run(self):
        self.queue_service.start_consuming()
