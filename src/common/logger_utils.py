import re
import logging

from logging.handlers import TimedRotatingFileHandler


def init_logging():
    new_logger = logging.getLogger('discord')
    new_logger.setLevel(logging.DEBUG)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_level = 10
    handler = TimedRotatingFileHandler("discord.log", when="midnight", interval=1)
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # add a suffix which you want
    handler.suffix = "%Y-%m-%d"

    # need to change the extMatch variable to match the suffix for it
    handler.extMatch = re.compile(r"^\d{8}$")

    # finally add handler to logger
    new_logger.addHandler(handler)

    # handler = logging.FileHandler(filename='./discord.log', encoding='utf-8', mode='w')
    # handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))

    # new_logger.addHandler(handler)
    return new_logger


logger = init_logging()
