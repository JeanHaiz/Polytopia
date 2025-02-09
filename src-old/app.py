import os
import asyncio

from discord_bot.bot_client import bot_client, slash_bot_client
from common.logger_utils import logger

DEBUG = os.getenv("POLYTOPIA_DEBUG")
token = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")

logger.debug("TOKEN: %s" % token)

loop = asyncio.get_event_loop()


async def start_slash_bot_client():
    slash_bot_client.start()

task2 = loop.create_task(bot_client.start(token=token))
task1 = loop.create_task(start_slash_bot_client())

gathered = asyncio.gather(task1, task2)
loop.run_until_complete(gathered)
