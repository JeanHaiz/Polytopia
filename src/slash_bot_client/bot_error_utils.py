import os
import sys
import asyncio
import traceback

from typing import Callable
from typing import Coroutine

from interactions import Client
from interactions import Snowflake
from interactions import get

from common.logger_utils import logger

from interactions import Channel
from interactions import CommandContext

from database.database_client import DatabaseClient
from common import error_utils


async def manage_slash_patching_errors(
        database_client: DatabaseClient,
        channel: Channel,
        ctx: CommandContext,
        patching_errors: list
) -> None:
    if patching_errors is not None and len(patching_errors) > 0:
        error_text = []
        for cause, error_filename in patching_errors:
            if error_filename is None:
                error_text.append(error_utils.MAP_PATCHING_ERROR_MESSAGES[cause])
            else:
                channel_id, message_id = database_client.get_resource_message(error_filename)
                if channel_id is not None and message_id is not None:
                    message = await channel.get_message(message_id)
                    if message is None:
                        error_text.append(error_utils.MAP_PATCHING_ERROR_MESSAGES[cause])
                    else:
                        await message.reply(error_utils.MAP_PATCHING_ERROR_MESSAGES[cause])
        error_text.append('<@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
        await ctx.send("\n".join(error_text))


async def wrap_slash_errors(
        ctx: CommandContext,
        client: Client,
        guild_id: Snowflake,
        fct: Callable[[], Coroutine]) -> None:
    try:
        is_test_server = str(guild_id) == os.getenv("DISCORD_TEST_SERVER")
        is_dev_env = os.getenv("POLYTOPIA_ENVIRONMENT", "") == "DEVELOPMENT"
        if (is_dev_env and is_test_server) or (not is_test_server and not is_dev_env):
            await asyncio.create_task(fct())
        else:
            logger.warning("CALLABLE NOT CALLED - ENVIRONMENT NOT OK")
    except:
        error = sys.exc_info()[0]
        logger.warning("##### ERROR #####")
        logger.warning(error)
        logger.warning(traceback.format_exc())
        print("##### ERROR #####")
        print(error)
        traceback.print_exc()
        
        # Polytopia Helper Testing server, Error channel
        error_channel = await get(client, Channel, object_id=int(os.getenv("DISCORD_ERROR_CHANNEL")))
        
        channel = await ctx.get_channel()
        guild = await ctx.get_guild()
        
        await error_channel.send(
            f"""Hey <@{os.getenv("DISCORD_ADMIN_USER")}>,\nError in channel {channel.name} on server {guild.name}:\n{traceback.format_exc()}\n""")
        
        await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
