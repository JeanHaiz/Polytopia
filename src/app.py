import os
import asyncio

from discord_bot.bot_client import bot_client
from common.logger_utils import logger

token = os.getenv("DISCORD_TOKEN")
debug = os.getenv("POLYTOPIA_DEBUG")

logger.debug("token: %s" % token)

asyncio.run(bot_client.start(token), debug=debug == "1")
