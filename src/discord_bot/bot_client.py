
import discord
import nest_asyncio
import pandas as pd

from discord.ext import commands
from discord.ext.commands import Context

from discord_bot import bot_utils
from database_interaction.database_client import DatabaseClient
from common import image_utils
from common.logger_utils import logger
from common.image_operation import ImageOp

nest_asyncio.apply()
# TODO: refactor with https://nik.re/posts/2021-09-25/object_oriented_discord_bot

bot_client = commands.Bot(":")

database_client = DatabaseClient(
    user="discordBot", password="password123", port="5432", database="polytopiaHelper_dev",
    host="database")


@bot_client.event
async def on_ready() -> None:
    print('We have logged in as {0}'.format(bot_client.user))


@bot_client.event
async def on_message(message: discord.Message) -> None:
    async def inner() -> None:
        print('Message from {0.author}: {0.content}'.format(message))
        logger.debug("received message: %s" % str(message))

        if message.author == bot_client.user:
            return

        if message.author.bot:
            return

        if message.attachments is not None and len(message.attachments) == 1:
            is_active = database_client.is_channel_active(message.channel.id)
            if is_active:
                for i, attachment in enumerate(message.attachments):
                    filename = database_client.add_resource(
                        message.guild.id, message.channel.id, message.id, message.author.id, message.author.name,
                        ImageOp.INPUT, i)
                    await image_utils.save_attachment(attachment, message.channel.name, ImageOp.INPUT, filename)
                    print("attachment saved", filename)
                    if message.reactions is not None and len(message.reactions) > 0:
                        await bot_utils.reaction_message_routine(bot_client, database_client, message, filename)
        await bot_client.process_commands(message)
    await bot_utils.wrap_errors(message, message.guild.id, inner, True)


@bot_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    async def inner() -> None:
        if payload.user_id == bot_client.user.id:
            return

        is_active = database_client.is_channel_active(payload.channel_id)
        if is_active:
            await bot_utils.reaction_added_routine(payload, bot_client, database_client)

    message = await bot_utils.get_message(bot_client, payload.channel_id, payload.message_id)
    if message is not None:
        await bot_utils.wrap_errors(message, message.channel.guild.id, inner, True)


@bot_client.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent) -> None:
    async def inner() -> None:
        is_active = database_client.is_channel_active(payload.channel_id)
        if is_active:
            await bot_utils.reaction_removed_routine(payload, bot_client, database_client)

    message = await bot_utils.get_message(bot_client, payload.channel_id, payload.message_id)
    if message is not None:
        await bot_utils.wrap_errors(message, message.channel.guild.id, inner, True)


@bot_client.command()
async def activate(ctx: Context) -> None:
    async def inner() -> None:
        logger.debug("activate channel %s" % ctx.channel)
        database_client.activate_channel(ctx.channel.id, ctx.channel.name, ctx.guild.id, ctx.guild.name)
        await ctx.send("channel activated")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command()
async def deactivate(ctx: Context) -> None:
    async def inner() -> None:
        logger.debug("deactivate channel %s" % ctx.channel)
        database_client.deactivate_channel(ctx.channel.id)
        await ctx.send("channel deactivated")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command()
async def list_active_channels(ctx: Context) -> None:
    async def inner() -> None:
        logger.debug("list active channels")
        active_channels = database_client.list_active_channels(ctx.guild.id)
        if len(active_channels) > 0:
            message = "active channels:\n- %s" % "\n- ".join([a[0] for a in active_channels if a[0] != ""])
        else:
            message = "no active channel"
        await ctx.send(message)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="setname")
async def set_player_discord_name(ctx: Context, discord_id: int, discord_name: str, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set player name")
        database_client.set_player_discord_name(discord_id, discord_name, polytopia_name)
        await ctx.send("Hi %s!" % discord_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="opponent")
async def add_game_opponent(ctx: Context, discord_name: str, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set player name")
        database_client.set_player_discord_name(None, discord_name, polytopia_name)
        await ctx.send("Hi %s!" % discord_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="setmyname")
async def set_self_discord_name(ctx: Context, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set self player name")
        database_client.set_player_discord_name(ctx.author.id, ctx.author.name, polytopia_name)
        await ctx.send("Hi %s!" % ctx.author.name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="scores")
async def get_channel_player_scores(ctx: Context, player: str = None) -> None:
    if player is None:
        await bot_utils.wrap_errors(ctx, ctx.guild.id, bot_utils.get_scores, True, database_client, ctx)
    else:
        await bot_utils.wrap_errors(ctx, ctx.guild.id, bot_utils.get_player_scores, True, database_client, ctx, player)


@bot_client.command(name="turn")
async def set_turn(ctx: Context, turn: int) -> None:
    async def inner() -> None:
        database_client.add_player_n_game(ctx.channel.id, ctx.guild.id, ctx.author.id, ctx.author.name)
        database_client.set_new_last_turn(ctx.channel.id, turn)
        await ctx.send("current turn is now %s" % str(turn))
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="size")
async def set_map_size(ctx: Context, size: str = None) -> None:
    async def inner() -> None:
        game_player_uuid = database_client.add_player_n_game(
            ctx.channel.id, ctx.guild.id, ctx.author.id, ctx.author.name)
        if game_player_uuid is not None:
            if size.isnumeric() and int(size) in [121, 196, 256, 324, 400, 900]:
                answer = database_client.set_game_map_size(ctx.channel.id, int(size))
                if answer.rowcount == 1:
                    await bot_utils.add_success_reaction(ctx.message)
                else:
                    await bot_utils.add_error_reaction(ctx.message)
                    myid = '<@338067113639936003>'  # Jean's id
                    await ctx.reply('There was an error. %s has been notified.' % myid, mention_author=False)
            else:
                await bot_utils.add_error_reaction(ctx.message)
                await ctx.send(
                    f"""Map size {str(size)} not recognised.\n"""
                    """Valid map sizes are 121, 196, 256, 324, 400 and 900.\n"""
                    """To to signal an error, react with ⁉️""")
        else:
            await bot_utils.add_error_reaction(ctx.message)
            myid = '<@338067113639936003>'  # Jean's id
            await ctx.reply('There was an error. %s has been notified.' % myid, mention_author=False)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="drop")
async def drop_score(ctx: Context, turn: str) -> None:
    async def inner() -> None:
        answer = database_client.drop_score(ctx.channel.id, turn)
        row_count = answer.rowcount
        if row_count != 0:
            await bot_utils.add_success_reaction(ctx.message)
            if row_count == 1:
                await ctx.reply(
                    "1 score entry was updated. \nTo to signal an error, react with ⁉️", mention_author=False)
            else:
                await ctx.reply(
                    "%d score entries were updated. \nTo to signal an error, react with ⁉️" % row_count,
                    mention_author=False)
        elif row_count == 0:
            await bot_utils.add_error_reaction(ctx.message)
            await ctx.reply("No score entry updated. \nTo to signal an error, react with ⁉️", mention_author=False)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="patch")
async def patch_map(ctx: Context, n_images: int = None, action_debug: bool = False) -> None:
    async def inner() -> None:
        await bot_utils.add_received_reaction(ctx.message)
        turn = database_client.get_last_turn(ctx.channel.id)
        turn, patch, patching_errors = await bot_utils.generate_patched_map_bis(
            database_client, ctx.channel.id, ctx.channel.name, ctx.message, turn, bot_client.loop, n_images, action_debug)
        await bot_utils.manage_patching_errors(ctx.channel, ctx.message, database_client, patching_errors)
        if patch is not None:
            return await ctx.channel.send(file=patch, content="map patched for turn %s" % turn)
        else:
            return await ctx.channel.send("patch failed")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="hello")
async def say_hello(ctx: Context) -> None:
    await bot_utils.wrap_errors(ctx, ctx.guild.id, ctx.send, True, "Welcome to my botifull world!")


@bot_client.command(name="players")
async def get_channel_players(ctx: Context) -> None:
    async def inner() -> None:
        game_players = database_client.get_game_players(ctx.channel.id)
        player_frame = pd.DataFrame(game_players)
        await ctx.send(player_frame.to_string())
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="trace")
async def get_map_trace(ctx: Context) -> None:
    async def inner() -> None:
        messages = database_client.get_channel_resource_messages(ctx.channel.id, ImageOp.MAP_INPUT)
        print("messages", messages)
        for i, m in enumerate(messages):
            try:
                message: discord.Message = await ctx.fetch_message(m['source_message_id'])
                sent_message = await message.reply("%s %d" % (ImageOp(m['operation']).name, i), mention_author=False)
            except discord.errors.NotFound:
                sent_message = await ctx.send("Message not found: %d" % m['source_message_id'])
            await bot_utils.add_delete_reaction(sent_message)
        if len(messages) == 0:
            await ctx.send("Trace empty")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="clear_maps")
async def clear_map_reactions(ctx: Context) -> None:
    async def inner() -> None:
        await bot_utils.clear_channel_map_reactions(database_client, ctx.channel)
        await bot_utils.add_success_reaction(ctx.message)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)


@bot_client.command(name="setscore")
async def set_player_score(ctx: Context, player_name: str, turn: int, score: int) -> None:
    async def inner() -> None:
        players = database_client.get_game_players(ctx.channel.id)
        matching_players = [p for p in players if p["polytopia_player_name"] == player_name]
        if len(matching_players) > 0:
            player_id = matching_players[0]["game_player_uuid"]
        else:
            player_id = database_client.add_missing_player(player_name, ctx.channel.id)
        answer = database_client.set_player_score(player_id, turn, score)
        row_count = answer.rowcount
        if row_count == 1:
            await bot_utils.add_success_reaction(ctx.message)
        elif row_count == 0:
            await bot_utils.add_error_reaction(ctx.message)
            await ctx.reply("No score entry was updated. \nTo to signal an error, react with ⁉️", mention_author=False)
        else:
            myid = '<@338067113639936003>'  # Jean's id
            await ctx.reply('There was an error. %s has been notified.' % myid, mention_author=False)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner, True)
