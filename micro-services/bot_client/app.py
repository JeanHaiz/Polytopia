import os
import asyncio

from bot_client import bot_client
from common.logger_utils import logger

DEBUG = os.getenv("POLYTOPIA_DEBUG")
token = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")

logger.debug("token: %s" % token)

loop = asyncio.get_event_loop()

bot_client_task = loop.create_task(bot_client.start(token=token))

loop.run_until_complete(bot_client_task)
