import pika
import threading


class QueueService(threading.Thread):
    def __init__(self, channel_number=None):
        self.declared_queues = []
        
        threading.Thread.__init__(self)
        self.connection = None
        self.channel = None
        self.reset_queues(channel_number)

    def reset_queues(self, channel_number=None):
        self.connection = pika.BlockingConnection(
            pika.URLParameters(url="amqp://slash_bot:slash_bot123@rabbitmq:5672/vhost")
        )
        self.channel = self.connection.channel(channel_number=channel_number)
        self.channel.add_on_return_callback(
            lambda ch, mt, pr, bd: print("Channel returned", ch, mt, pr, bd, flush=True)
        )
        self.channel.add_on_cancel_callback(
            lambda mf: print("Channel cancelled", mf, flush=True)
        )
        
    def get_channel(self):
        return self.channel

    def declare_queue(self, queue_name):
        if queue_name not in self.declared_queues:
            self.declared_queues.append(queue_name)
            return self.channel.queue_declare(queue=queue_name)
        
    def consume(self, queue_name: str, **kwargs):
        return self.channel.basic_consume(queue=queue_name, **kwargs)
        
    def start_consuming(self):
        return self.channel.start_consuming()

    def process_data_events(self):
        return self.connection.process_data_events()

    def is_open(self):
        return self.channel.is_open and self.connection.is_open

    def send_message(self, queue_name, body):
        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body
        )
