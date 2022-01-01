
import discord
import logging

from discord.ext import commands

from discord_bot import bot_utils
from database_interaction.database_client import DatabaseClient
from common import image_utils
from common.logger_utils import logger

# TODO: refactor with https://nik.re/posts/2021-09-25/object_oriented_discord_bot

bot_client = commands.Bot(":")

database_client = DatabaseClient(
    user="discordBot", password="password123", port=5432, database="polytopiaHelper_dev",
    host="database")


@bot_client.event
async def on_ready():
    print('We have logged in as {0}'.format(bot_client.user))


@bot_client.event
async def on_message(message):
    print('Message from {0.author}: {0.content}'.format(message))
    logger.log(logging.DEBUG, "received message: %s" % str(message))

    if message.author == bot_client.user:
        return

    if message.author.bot:
        return

    if message.attachments is not None:
        is_active = database_client.is_channel_active(message.channel.id)
        if is_active:
            for i, attachment in enumerate(message.attachments):
                filename = database_client.add_resource(message, message.author, image_utils.ImageOperation.INPUT, i)
                await image_utils.save_attachment(attachment, message, image_utils.ImageOperation.INPUT, filename)
                print("attachment saved", filename)
                if message.reactions is not None and len(message.reactions) > 0:
                    await bot_utils.reaction_message_routine(message, filename)
    await bot_client.process_commands(message)


@bot_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    is_active = database_client.is_channel_active(payload.channel_id)
    if is_active:
        await bot_utils.reaction_added_routine(payload, bot_client, database_client)


@bot_client.command()
async def reload_extention(ctx, extension):
    print("reloading")
    bot_client.reload_extension(f"{extension}")
    embed = discord.Embed(title='Reload', description=f'{extension} successfully reloaded', color=0xff00c8)
    await bot_client.send(embed=embed)


@bot_client.command()
async def activate(ctx):
    logger.log(logging.DEBUG, "activate channel %s" % ctx.channel)
    database_client.activate_channel(ctx.channel, ctx.guild)
    await ctx.send("channel activated")


@bot_client.command()
async def deactivate(ctx):
    logger.log(logging.DEBUG, "deactivate channel %s" % ctx.channel)
    database_client.deactivate_channel(ctx.channel)
    await ctx.send("channel deactivated")


@bot_client.command()
async def list_active_channels(ctx):
    logger.log(logging.DEBUG, "list active channels")
    active_channels = database_client.list_active_channels(ctx.guild)
    if len(active_channels) > 0:
        message = "active channels:\n- %s" % "\n- ".join([a[0] for a in active_channels if a[0] != ""])
    else:
        message = "no active channel"
    await ctx.send(message)
