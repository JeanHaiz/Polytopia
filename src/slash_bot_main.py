import os
import asyncio
import socket
import time
import datetime
import aiohttp

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
    
    print("STARTING BOT at %s" % datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    loop = asyncio.get_event_loop()
    
    def __quit_coroutine(message, sleep_time=None):
        logger.warning(message)
        print(message, flush=True)
        if sleep_time is not None:
            time.sleep(sleep_time)

        loop.run_until_complete(run_bot())
    
    async def check_health():
        await slash_bot_client.wait_until_ready()
        
        async def inner():
            alive = True
            print("HEALTH CHECK START", flush=True)
            try:
                client_info = await slash_bot_client._http.get_current_bot_information()
                client_commands = await slash_bot_client._http.get_application_commands(
                    slash_bot_client.me.id  # , os.getenv("DISCORD_TEST_SERVER", None)
                )
                print("Client commands", slash_bot_client.me.id, len(client_commands), flush=True) # 1036220176577863680
                
                if client_info is None:  # or client_info[0] is not None or isinstance(client_info[0], BaseException):
                    alive = False

                """print("HTTP Client info", client_info, flush=True)
                print(
                    "Websocket ready",
                    slash_bot_client._websocket.ready.is_set(),
                    not slash_bot_client._websocket._client.closed, flush=True)"""
                
                if not slash_bot_client._websocket.ready.is_set():  # or slash_bot_client._websocket._client.closed:
                    alive = False
                # print("Ping", await slash_bot_client._websocket._client.ping(b"hello"), flush=True)
            except (aiohttp.client_exceptions.ClientConnectorError, socket.gaierror):
                alive = False
            except BaseException as e:
                print("Unrecognised error:", e)
                alive = False
            print("HEALTH CHECK END", alive, flush=True)
            if alive:
                await asyncio.sleep(30)
                await inner()
            else:
                __quit_coroutine("BOT DEAD", 30)

        print("starting threads", flush=True)
        await inner()

    async def start_bot():
        try:
            slash_bot_client.start()
        except (aiohttp.client_exceptions.ClientConnectorError, socket.gaierror):
            __quit_coroutine("BOT CONNECTION ERROR", 30)
        except BaseException as e:
            print("Exception:\n", e)
            __quit_coroutine("BOT FATAL FAILURE", 30)
    
    nest_asyncio.apply(loop)
    
    slash_bot_client = interactions.Client(
        token=token,
        intents=interactions.Intents.DEFAULT | interactions.Intents.GUILD_MESSAGE_CONTENT,
        prefix=":"
    )
    
    extensions = SlashBotExtension(slash_bot_client)
    
    async def run_bot():
        task2 = loop.create_task(receiver_service.get_async_connection(
            "bot_client",
            slash_bot_client,
            loop
        ))
        task1 = loop.create_task(start_bot())
        task3 = loop.create_task(check_health())
        
        tasks = [task2, task1, task3]
    
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # if there are unfinished tasks, that is because one of them
            # raised - cancel the rest
            for t in tasks:
                if not t.done():
                    t.cancel()
    await run_bot()

asyncio.run(create_client())
