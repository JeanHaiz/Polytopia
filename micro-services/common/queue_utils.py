import pika
import time

from pika import exceptions
from pika.adapters.asyncio_connection import AsyncioConnection


def get_blocking_channel(params):
    while True:
        try:
            connection = pika.BlockingConnection(params)
        except pika.exceptions.ProbableAuthenticationError:
            # self.logger.error('AMQP authentication failed!')
            print('AMQP authentication failed!')
            time.sleep(1)
        except pika.exceptions.ProbableAccessDeniedError:
            # self.logger.error('AMQP authentication for virtual host failed!')
            print('AMQP authentication for virtual host failed!')
            time.sleep(1)
        except pika.exceptions.AMQPConnectionError:
            print('AMQP connection failed!')
            time.sleep(1)
        else:
            # self.logger.info('AMQP connection successful.')
            channel = connection.channel()
            print('AMQP connection successful.')  # , connection, channel)
            # connection.close()
            return channel
