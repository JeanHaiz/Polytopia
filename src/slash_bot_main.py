import os
import asyncio
import time
import datetime
import traceback
import signal
import aiohttp

import interactions
import nest_asyncio

from slash_bot_client.extensions.bot_extension import SlashBotExtension
from slash_bot_client.extensions.map_extension import MapExtension
from slash_bot_client.extensions.score_extension import ScoreExtension
from slash_bot_client.extensions.user_extension import UserExtension

from slash_bot_client.utils.bot_utils import BotUtils
from slash_bot_client.queue_services.queue_service import QueueService

from common.logger_utils import logger

"""
Application start
- starts the slash bot
- registers logger
"""

# TODO:
# - web-socket management: https://stackoverflow.com/questions/63033275/closing-shutting-down-a-python-websocket

nest_asyncio.apply()

DEBUG = os.getenv("POLYTOPIA_DEBUG")
token = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")

print(DEBUG, token, flush=True)

logger.debug("TOKEN: %s" % token)

exceptions = (
    aiohttp.client_exceptions.ClientConnectorError,
    interactions.api.error.LibraryException,
    ConnectionRefusedError
)


async def create_client():
    print("STARTING BOT at %s" % datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    loop = asyncio.get_event_loop()
    
    async def check_health():
        await slash_bot_client.wait_until_ready()
        alive = True
        
        async def inner() -> bool:
            inner_alive = True
            if DEBUG:
                print("HEALTH CHECK START", flush=True)
            try:
                await slash_bot_client._websocket.wait_until_ready()
                client_info = await slash_bot_client._http.get_current_bot_information()
                if client_info is None or \
                        not slash_bot_client._websocket.ready.is_set() or \
                        slash_bot_client._websocket._client is None or \
                        slash_bot_client._websocket._client.closed:
                    inner_alive = False
            
            except exceptions as e:
                print("RECOGNISED EXCEPTION", e, flush=True)
                logger.warning("RECOGNISED EXCEPTION" + str(e))
                inner_alive = False
            except BaseException as e:
                print("UN-RECOGNISED EXCEPTION:", e, flush=True)
                logger.warning("UN-RECOGNISED EXCEPTION" + str(e))
                inner_alive = False
            except object:
                print("BARE EXCEPTION:", traceback.format_exc(), flush=True)
                inner_alive = False
            if DEBUG:
                print("HEALTH CHECK END", alive, flush=True)
            return inner_alive
        
        while alive:
            alive = await inner()
            if alive:
                await asyncio.sleep(30)
            else:
                raise KeyboardInterrupt()
    
    async def start_bot():
        print("STARTING BOT ACTUALLY", flush=True)
        slash_bot_client.start()
    
    # nest_asyncio.apply(loop)
    
    slash_bot_client = interactions.Client(
        token=token,
        intents=interactions.Intents.DEFAULT | interactions.Intents.GUILD_MESSAGE_CONTENT,
        prefix=":"
    )
    
    queue_service = QueueService()
    bot_utils = BotUtils(queue_service)
    
    bot_extensions = SlashBotExtension(slash_bot_client, bot_utils)
    map_extensions = MapExtension(slash_bot_client, bot_utils)
    score_extensions = ScoreExtension(slash_bot_client, bot_utils)
    user_extensions = UserExtension(slash_bot_client, bot_utils)
    
    all_tasks = asyncio.all_tasks(loop)
    print("All tasks at run bot:", len(all_tasks), flush=True)
    
    task2 = loop.create_task(bot_utils.get_async_connection(
        queue_service,
        "bot_client",
        slash_bot_client,
        loop
    ))
    task1 = loop.create_task(start_bot())
    task3 = loop.create_task(check_health())
    
    tasks = [task2, task1, task3]
    
    def stop():
        try:
            loop.stop()
        except BaseException as e:
            print("Exception here", e, flush=True)
            pass
    
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, stop)
    
    all_tasks = asyncio.all_tasks(loop)
    print("Tasks created", len(all_tasks), flush=True)
    try:
        results = await asyncio.gather(*tasks)
        all_tasks = asyncio.all_tasks(loop)
        print("Tasks gathered", len(all_tasks), flush=True)
        print(results)
    
    finally:
        # if there are unfinished tasks, that is because one of them
        # raised - cancel the rest
        for t in tasks:
            if not t.done():
                try:
                    t.cancel()
                except (asyncio.exceptions.CancelledError, RuntimeError):
                    pass


while True:
    try:
        result = asyncio.run(create_client())
        print("Result", result, flush=True)
        time.sleep(10)
    except BaseException as e:
        print("Exception elsewhere: " + str(e), flush=True)
        if DEBUG:
            print(traceback.format_exc())
        logger.warning("Exception elsewhere" + str(e))
        time.sleep(10)
        print("Restarting bot", flush=True)
