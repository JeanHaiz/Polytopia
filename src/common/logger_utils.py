import logging


def init_logging():
    new_logger = logging.getLogger('discord')
    new_logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(filename='./discord.log', encoding='utf-8', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))

    new_logger.addHandler(handler)
    return new_logger


logger = init_logging()
