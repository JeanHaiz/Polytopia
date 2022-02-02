
import discord

from discord.ext import commands

from discord_bot import bot_utils
from database_interaction.database_client import DatabaseClient
from common import image_utils
from common.logger_utils import logger
from common.image_utils import ImageOp

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
    logger.debug("received message: %s" % str(message))

    if message.author == bot_client.user:
        return

    if message.author.bot:
        return

    if message.attachments is not None and len(message.attachments) != 0:
        is_active = database_client.is_channel_active(message.channel.id)
        if is_active:
            for i, attachment in enumerate(message.attachments):
                filename = database_client.add_resource(message, message.author, ImageOp.INPUT, i)
                await image_utils.save_attachment(attachment, message.channel.name, ImageOp.INPUT, filename)
                print("attachment saved", filename)
                if message.reactions is not None and len(message.reactions) > 0:
                    await bot_utils.wrap_errors(
                        message.channel, message.guild.id, bot_utils.reaction_message_routine, True,
                        *(database_client, message, filename))
    await bot_client.process_commands(message)


@bot_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    is_active = database_client.is_channel_active(payload.channel_id)
    if is_active:
        channel = bot_client.get_channel(payload.channel_id)
        await bot_utils.wrap_errors(
            channel, channel.guild.id, bot_utils.reaction_added_routine, True, *(payload, bot_client, database_client))


@bot_client.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent):
    is_active = database_client.is_channel_active(payload.channel_id)
    if is_active:
        channel = bot_client.get_channel(payload.channel_id)
        await bot_utils.wrap_errors(
            channel, channel.guild.id, bot_utils.reaction_removed_routine, True,
            *(payload, bot_client, database_client))


@bot_client.command()
async def reload_extention(ctx, extension):
    print("reloading")
    bot_client.reload_extension(f"{extension}")
    embed = discord.Embed(title='Reload', description=f'{extension} successfully reloaded', color=0xff00c8)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, bot_client.send, embed=embed)


@bot_client.command()
async def activate(ctx):
    logger.debug("activate channel %s" % ctx.channel)
    database_client.activate_channel(ctx.channel, ctx.guild)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "channel activated :)")


@bot_client.command()
async def deactivate(ctx):
    logger.debug("deactivate channel %s" % ctx.channel)
    database_client.deactivate_channel(ctx.channel.id)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "channel deactivated")


@bot_client.command()
async def list_active_channels(ctx):
    logger.debug("list active channels")
    active_channels = database_client.list_active_channels(ctx.guild.id)
    if len(active_channels) > 0:
        message = "active channels:\n- %s" % "\n- ".join([a[0] for a in active_channels if a[0] != ""])
    else:
        message = "no active channel"
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, message)


@bot_client.command(name="setname")
async def set_player_discord_name(ctx, discord_id, discord_name, polytopia_name):
    logger.debug("set player name")
    database_client.set_player_discord_name(discord_id, discord_name, polytopia_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "Hi %s!" % discord_name)


@bot_client.command(name="opponent")
async def add_game_opponent(ctx, discord_name, polytopia_name):
    logger.debug("set player name")
    database_client.set_player_discord_name(None, discord_name, polytopia_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "Hi %s!" % discord_name)


@bot_client.command(name="setmyname")
async def set_self_discord_name(ctx, polytopia_name):
    logger.debug("set self player name")
    database_client.set_player_discord_name(ctx.author.id, ctx.author.name, polytopia_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "Hi %s!" % ctx.author.name)


@bot_client.command(name="scores")
async def get_channel_scores(ctx):
    await bot_utils.wrap_errors(ctx, ctx.guild.id, bot_utils.get_scores, True, *(database_client, ctx))


@bot_client.command(name="turn")
async def set_turn(ctx, turn):
    database_client.add_player_n_game(ctx.message, ctx.author)
    database_client.set_new_last_turn(ctx.channel.id, turn)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "current turn is now %s" % str(turn))


@bot_client.command(name="size")
async def set_map_size(ctx, size):
    database_client.add_player_n_game(ctx.message, ctx.author)
    if size.isnumeric() and int(size) in [121, 196, 256, 324, 400, 900]:
        database_client.set_game_map_size(ctx.channel.id, int(size))
        await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "current map size now is %s" % size)
    else:
        await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "map size not recognised")


@bot_client.command(name="drop")
async def drop_score(ctx, turn):
    await bot_utils.wrap_errors(ctx, ctx.guild.id, database_client.drop_score, False, *(ctx.channel.id, turn))


@bot_client.command(name="patch")
async def patch_map(ctx):
    print("in")
    turn = database_client.get_last_turn(ctx.channel.id)
    turn, patch = await bot_utils.generate_patched_map(
        database_client, ctx.channel.id, ctx.channel.name, ctx.message, turn)
    if patch is not None:
        return await ctx.channel.send(file=patch, content="map patched for turn %s" % turn)
    else:
        return await ctx.channel.send("patch failed")
    # await bot_utils.process_map_patching(ctx.message, ctx.channel, database_client)
    # await bot_utils.wrap_errors(ctx.message, ctx.guild.id, bot_utils.process_map_patching, True, ctx.message,
    # ctx.channel, database_client)


@bot_client.command(name="hello")
async def say_hello(ctx):
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "hello here")
