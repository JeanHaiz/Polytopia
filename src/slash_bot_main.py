import os
import asyncio
import socket
import time
import datetime
import traceback

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
    socket.gaierror,
    interactions.api.error.LibraryException
)


async def create_client():
    print("STARTING BOT at %s" % datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    loop = asyncio.get_event_loop()
    
    async def __quit_coroutine(message, sleep_time=None):
        logger.warning("QUIT COROUTINE\n" + message)
        print("QUIT COROUTINE\n", message, flush=True)
        if sleep_time is not None:
            time.sleep(sleep_time)
        await slash_bot_client._websocket.close()
        
        all_tasks = asyncio.all_tasks(loop)
        print("All tasks:", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None)
        
        # loop.create_task(run_bot())
        # loop.run_forever()
        loop.run_until_complete(run_bot())
    
    async def check_health():
        await slash_bot_client.wait_until_ready()
        alive = True
        
        async def inner() -> bool:
            inner_alive = True
            # print("HEALTH CHECK START", flush=True)
            try:
                # print("Readiness before:", slash_bot_client._websocket.ready.is_set(), flush=True)
                await slash_bot_client._websocket.wait_until_ready()
                # print("Bot ready, getting info", flush=True)
                client_info = await slash_bot_client._http.get_current_bot_information()
                # print("client info", client_info is not None)
                """client_commands = await slash_bot_client._http.get_application_commands(
                    slash_bot_client.me.id  # , os.getenv("DISCORD_TEST_SERVER", None)
                )"""
                # slash bot app id 1036220176577863680
                # print("Client commands", slash_bot_client.me.id, len(client_commands), flush=True)
                
                # or client_info[0] is not None or isinstance(client_info[0], BaseException):
                if client_info is None or \
                        not slash_bot_client._websocket.ready.is_set() or \
                        slash_bot_client._websocket._client is None or \
                        slash_bot_client._websocket._client.closed:
                    inner_alive = False
                
                """print("HTTP Client info", client_info, flush=True)
                print(
                    "Websocket ready",
                    slash_bot_client._websocket.ready.is_set(),
                    not slash_bot_client._websocket._client.closed, flush=True)"""
                
                # if not slash_bot_client._websocket.ready.is_set():  # or slash_bot_client._websocket._client.closed:
                #     inner_alive = False
                # print("Ping", await slash_bot_client._websocket._client.ping(b"hello"), flush=True)
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
            # print("HEALTH CHECK END", alive, flush=True)
            return inner_alive
        
        # print("starting threads", flush=True)
        while alive:
            alive = await inner()
            if alive:
                await asyncio.sleep(30)
            else:
                await __quit_coroutine("BOT DEAD", 30)
    
    async def start_bot():
        try:
            print("STARTING BOT ACTUALLY", flush=True)
            slash_bot_client.start()
        except exceptions as e:
            await __quit_coroutine("BOT CONNECTION ERROR:\n" + str(e), 30)
        except BaseException as e:
            await __quit_coroutine("BOT FATAL FAILURE:\n" + str(e), 30)
        except object as e:
            print("HERE YOU ARE", flush=True)
            raise e
    
    nest_asyncio.apply(loop)
    
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
    
    async def run_bot():
        
        all_tasks = asyncio.all_tasks(loop)
        print("All tasks at run bot:", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None)
        
        task2 = loop.create_task(bot_utils.get_async_connection(
            queue_service,
            "bot_client",
            slash_bot_client,
            loop
        ))
        task1 = loop.create_task(start_bot())
        task3 = loop.create_task(check_health())
        
        tasks = [task2, task1, task3]
        
        all_tasks = asyncio.all_tasks(loop)
        print("Tasks created", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None, flush=True)
        try:
            results = await asyncio.gather(*tasks)
            all_tasks = asyncio.all_tasks(loop)
            print("Tasks gathered", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None, flush=True)
            print(results)
        
        finally:
            # if there are unfinished tasks, that is because one of them
            # raised - cancel the rest
            for t in tasks:
                if not t.done():
                    t.cancel()
    
    await run_bot()


asyncio.run(create_client())
