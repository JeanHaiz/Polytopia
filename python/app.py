import os
from discord_bot.bot_client import bot_client

token = os.getenv("DISCORD_TOKEN")


@bot_client.command()
async def reload(ctx, extension):
    print("reloading")
    bot_client.reload_extension(f"discord_bot.{extension}")
    print("reloaded extention: %s" % extension)

bot_client.run(token)
