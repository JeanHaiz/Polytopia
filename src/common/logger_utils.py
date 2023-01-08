import re
import logging

from logging.handlers import TimedRotatingFileHandler


def init_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handler = TimedRotatingFileHandler("discord_log/discord.log", when="midnight", interval=1)
    # handler.setLevel(logging.DEBUG if os.getenv("POLYTOPIA_DEBUG", 0) == "1" else logging.INFO)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # add a suffix which you want
    handler.suffix = "%Y-%m-%d"

    # need to change the extMatch variable to match the suffix for it
    handler.extMatch = re.compile(r"^\d{8}$")

    # finally add handler to logger
    bot_logger = logging.getLogger('discord')
    bot_logger.addHandler(handler)

    bot_client_logger = logging.getLogger('client')
    bot_client_logger.addHandler(handler)

    http_client_logger = logging.getLogger('htto')
    http_client_logger.addHandler(handler)

    # handler = logging.FileHandler(filename='./discord.log', encoding='utf-8', mode='w')
    # handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))

    # new_logger.addHandler(handler)
    return bot_logger


logger = init_logging()
