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

exceptions = (
    aiohttp.client_exceptions.ClientConnectorError,
    socket.gaierror,
    interactions.api.error.LibraryException
)


async def create_client():
    print("STARTING BOT at %s" % datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    loop = asyncio.get_event_loop()
    
    async def __quit_coroutine(message, sleep_time=None):
        logger.warning(message)
        print(message, flush=True)
        if sleep_time is not None:
            time.sleep(sleep_time)
        await slash_bot_client._websocket.close()
        
        all_tasks = asyncio.all_tasks(loop)
        print("All tasks:", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None)
        
        loop.run_until_complete(run_bot())
    
    async def check_health():
        await slash_bot_client.wait_until_ready()
        
        async def inner():
            alive = True
            # print("HEALTH CHECK START", flush=True)
            try:
                # print("Readiness before:", slash_bot_client._websocket.ready.is_set(), flush=True)
                await slash_bot_client._websocket.wait_until_ready()
                # print("Bot ready, getting info", flush=True)
                client_info = await slash_bot_client._http.get_current_bot_information()
                # print("client info", client_info is not None)
                client_commands = await slash_bot_client._http.get_application_commands(
                    slash_bot_client.me.id  # , os.getenv("DISCORD_TEST_SERVER", None)
                )
                # slash bot app id 1036220176577863680
                # print("Client commands", slash_bot_client.me.id, len(client_commands), flush=True)

                # or client_info[0] is not None or isinstance(client_info[0], BaseException):
                if client_info is None or not slash_bot_client._websocket.ready.is_set():
                    alive = False
                
                """print("HTTP Client info", client_info, flush=True)
                print(
                    "Websocket ready",
                    slash_bot_client._websocket.ready.is_set(),
                    not slash_bot_client._websocket._client.closed, flush=True)"""
                
                if not slash_bot_client._websocket.ready.is_set():  # or slash_bot_client._websocket._client.closed:
                    alive = False
                # print("Ping", await slash_bot_client._websocket._client.ping(b"hello"), flush=True)
            except exceptions as e:
                print("RECOGNISED EXCEPTION", e, flush=True)
                logger.warning("RECOGNISED EXCEPTION" + str(e))
                alive = False
            except BaseException as e:
                print("UN-RECOGNISED EXCEPTION:", e, flush=True)
                logger.warning("UN-RECOGNISED EXCEPTION" + str(e))
                alive = False
            # print("HEALTH CHECK END", alive, flush=True)
            if alive:
                await asyncio.sleep(30)
                await inner()
            else:
                await __quit_coroutine("BOT DEAD", 30)
        
        # print("starting threads", flush=True)
        await inner()
    
    async def start_bot():
        try:
            slash_bot_client.start()
        except exceptions as e:
            await __quit_coroutine("BOT CONNECTION ERROR:\n" + str(e), 30)
        except BaseException as e:
            await __quit_coroutine("BOT FATAL FAILURE:\n" + str(e), 30)
    
    nest_asyncio.apply(loop)
    
    slash_bot_client = interactions.Client(
        token=token,
        intents=interactions.Intents.DEFAULT | interactions.Intents.GUILD_MESSAGE_CONTENT,
        prefix=":"
    )
    
    extensions = SlashBotExtension(slash_bot_client)
    
    async def run_bot():
        
        all_tasks = asyncio.all_tasks(loop)
        print("All tasks at run bot:", len(all_tasks), list(all_tasks)[0] if len(all_tasks) > 0 else None)
        
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
