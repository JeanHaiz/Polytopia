import os
import asyncio
import interactions
import nest_asyncio

from slash_bot_client.bot_extension import SlashBotExtension
from slash_bot_client import receiver_service
from common.logger_utils import logger

"""
Application start
- starts the slash bot
- registers logger
"""

nest_asyncio.apply()

DEBUG = os.getenv("POLYTOPIA_DEBUG")
token = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")

print(DEBUG, token, flush=True)

logger.debug("TOKEN: %s" % token)


async def create_client():
    async def start_bot():
        slash_bot_client.start()

    loop = asyncio.get_event_loop()
    nest_asyncio.apply(loop)
    
    slash_bot_client = interactions.Client(
        token=token,
        intents=interactions.Intents.DEFAULT | interactions.Intents.GUILD_MESSAGE_CONTENT,
        prefix=":"
    )
    
    extensions = SlashBotExtension(slash_bot_client)
    
    task2 = loop.create_task(receiver_service.get_async_connection(
        "bot_client",
        slash_bot_client,
        loop
    ))
    task1 = loop.create_task(start_bot())
    gathered = asyncio.gather(task2, task1)
    loop.run_until_complete(gathered)

asyncio.run(create_client())
