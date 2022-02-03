import os
from discord_bot.bot_client import bot_client
from common.logger_utils import logger

token = os.getenv("DISCORD_TOKEN")

logger.debug("token: %s" % token)


@bot_client.command()
async def reload(ctx, extension):
    print("reloading")
    bot_client.reload_extension(f"discord_bot.{extension}")
    print("reloaded extention: %s" % extension)

bot_client.run(token)
