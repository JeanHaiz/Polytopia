import logging


def init_logging():
    logger = logging.getLogger('discord')
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(filename='./discord.log', encoding='utf-8', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))

    logger.addHandler(handler)
    return logger


logger = init_logging()
