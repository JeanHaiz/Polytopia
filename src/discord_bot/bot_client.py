
import os
import discord
import interactions
import nest_asyncio
import pandas as pd

from interactions import CommandContext
from interactions import File
from discord.ext import commands
from discord.ext.commands import Context

from discord_bot import bot_utils
from database_interaction.database_client import DatabaseClient
from map_patching.map_patching_errors import MapPatchingErrors
from common import image_utils
from common.logger_utils import logger
from common.image_operation import ImageOp

VERSION = "0.1.5"

nest_asyncio.apply()
# TODO: refactor with https://nik.re/posts/2021-09-25/object_oriented_discord_bot

DEBUG = os.getenv("POLYTOPIA_DEBUG")
token = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")
print("token", token)
slash_bot_client = interactions.Client(
    token=token,
    intents=interactions.Intents.DEFAULT | interactions.Intents.GUILD_MESSAGE_CONTENT
)

intents = discord.Intents.default()
intents.message_content = True
bot_client = commands.Bot(":", intents=intents, application_id=918189457893621760, bot=True)

# guild_ids = [918195469245628446]

database_client = DatabaseClient(
    user="discordBot", password="password123", port="5432", database="polytopiaHelper_dev",
    host="database")


@bot_client.event
async def on_ready() -> None:
    print('We have logged in as {0}'.format(bot_client.user))
    logger.info('We have logged in as {0}'.format(bot_client.user))


@bot_client.event
async def on_error(event, *args, **kwargs) -> None:
    print("Error: \n%s\n%s\n%s\n" % (event, args, kwargs))


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
    await bot_utils.wrap_errors(message, message.guild.id, inner)


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
        await bot_utils.wrap_errors(message, message.channel.guild.id, inner)


@bot_client.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent) -> None:
    async def inner() -> None:
        is_active = database_client.is_channel_active(payload.channel_id)
        if is_active:
            await bot_utils.reaction_removed_routine(payload, bot_client, database_client)

    message = await bot_utils.get_message(bot_client, payload.channel_id, payload.message_id)
    if message is not None:
        await bot_utils.wrap_errors(message, message.channel.guild.id, inner)


@slash_bot_client.command(
    name="activate",
    description="Activates the bot in the channel. Will now respond to map patching and score recognition requests.",
    options=[
        interactions.Option(
            name="size",
            description="Map size",
            type=interactions.OptionType.INTEGER,
            required=True,
            choices=[
                interactions.Choice(name="11 x 11", value=121),
                interactions.Choice(name="14 x 14", value=196),
                interactions.Choice(name="16 x 16", value=256),
                interactions.Choice(name="18 x 18", value=324),
                interactions.Choice(name="20 x 20", value=400),
                interactions.Choice(name="30 x 30", value=900),
            ],
        ),
    ],
)
async def slash_activate(ctx: CommandContext, size: int) -> None:
    async def inner() -> None:
        logger.debug("activate channel %s" % ctx.channel_id)
        channel = await ctx.get_channel()
        guild = await ctx.get_guild()
        activation_result = database_client.activate_channel(ctx.channel_id, channel.name, ctx.guild_id, guild.name)
        database_client.set_game_map_size(ctx.channel.id, int(size))
        if activation_result.rowcount == 1:
            await ctx.send("channel activated")
        else:
            myid = '<@338067113639936003>'  # Jean's id
            await ctx.send('There was an error. %s has been notified.' % myid)
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="deactivate",
    description="Deactivates the channel. Reactions and image uploads will not be tracked anymore."
)
async def slash_deactivate(ctx: CommandContext) -> None:
    async def inner() -> None:
        logger.debug("deactivate channel %s" % ctx.channel_id)
        deactivation_result = database_client.deactivate_channel(ctx.channel_id)
        if deactivation_result.rowcount == 1:
            await ctx.send("channel deactivated")
        else:
            myid = '<@338067113639936003>'  # Jean's id
            await ctx.send('There was an error. %s has been notified.' % myid)
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="version",
    description="Current bot version."
)
async def version(ctx: CommandContext) -> None:
    await ctx.send("version %s" % VERSION)


@slash_bot_client.command(
    name="scores",
    description="Shows saved scores plot. If player is specified, shows the player score history.",
    options=[
        interactions.Option(
            name="player",
            description="Retrieves score for the player",
            type=interactions.OptionType.STRING,
            required=False,
        ),
    ]
)
async def slash_get_channel_player_scores(ctx: CommandContext, player: str = None) -> None:
    async def inner():
        if player is None:
            await bot_utils.get_scores(database_client, ctx)
        else:
            await bot_utils.get_player_scores(database_client, ctx, player)
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="patch",
    description="Patches saved maps together.",
    options=[
        interactions.Option(
            name="number_of_images",
            description="Maximum number of images to patch",
            type=interactions.OptionType.INTEGER,
            required=False
        )
    ]
)
async def slash_patch_map(ctx: CommandContext, number_of_images: int = None) -> None:
    async def inner() -> None:
        action_debug = ctx.author.id == 338067113639936003
        channel = await ctx.get_channel()
        await ctx.send("processing")
        turn = database_client.get_last_turn(ctx.channel_id)
        turn, output_tuple, patching_errors = await bot_utils.generate_patched_map_bis(
            database_client,
            channel.id,
            channel.name,
            ctx.author.id,
            ctx.author.name,
            ctx.guild_id,
            ctx.id,
            turn,
            bot_client.loop,
            number_of_images,
            action_debug)
        if output_tuple is not None:
            patch_path, patch_uuid, patch_filename = output_tuple
            database_client.update_patching_process_output_filename(patch_uuid, patch_filename)
            with open(patch_path, "rb") as fh:
                attachment = File(fp=fh, filename=patch_filename + ".png")
                if attachment is not None:
                    await channel.send(files=attachment, content="Map patched for turn %s" % turn)
                elif len(patching_errors) == 0 and attachment is None:
                    patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                    my_id = '<@338067113639936003>'  # Jean's id
                    await channel.send('There was an error. %s has been notified.' % my_id)
                fh.close()
        await bot_utils.manage_slash_patching_errors(channel, database_client, patching_errors)
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="set-score",
    description="Set score for a specific player and turn",
    options=[
        interactions.Option(
            name="player_name",
            description="player-name",
            type=interactions.OptionType.STRING,
            required=True
        ),
        interactions.Option(
            name="turn",
            description="turn",
            type=interactions.OptionType.INTEGER,
            required=True
        ),
        interactions.Option(
            name="score",
            description="score",
            type=interactions.OptionType.INTEGER,
            required=True
        )
    ]
)
async def slash_set_player_score(ctx: CommandContext, player_name: str, turn: int, score: int) -> None:
    async def inner() -> None:
        channel = await ctx.get_channel()
        players = database_client.get_game_players(channel.id)
        matching_players = [p for p in players if p["polytopia_player_name"] == player_name]
        if len(matching_players) > 0:
            player_id = matching_players[0]["game_player_uuid"]
        else:
            player_id = database_client.add_missing_player(player_name, channel.id)
        scores = database_client.get_channel_scores(channel.id)
        if scores is None or len(scores[(scores["turn"] == turn) & (scores["polytopia_player_name"] == player_name)]) == 0:
            answer = database_client.add_score(channel.id, player_id, score, turn)
        else:
            answer = database_client.set_player_score(player_id, turn, score)
        row_count = answer.rowcount
        if row_count == 1:
            await ctx.send("Score stored")
        elif row_count == 0:
            await ctx.send("No score entry was updated. \nTo to signal an error, react with ⁉️")
        else:
            myid = '<@338067113639936003>'  # Jean's id
            await ctx.send('There was an error. %s has been notified.' % myid)
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="clear_maps",
    description="Clear the stack of maps to patch"
)
async def slash_clear_map_reactions(ctx: CommandContext) -> None:
    async def inner() -> None:
        channel = await ctx.get_channel()
        await bot_utils.clear_channel_map_reactions(database_client, channel)
        await ctx.send("Done.")
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@slash_bot_client.command(
    name="channels",
    description="admin command — lists all active channels in the server"
)
async def slash_list_active_channels(ctx: CommandContext) -> None:
    async def inner() -> None:
        if ctx.author.id == 338067113639936003:
            logger.debug("list active channels")
            active_channels = database_client.list_active_channels(ctx.guild_id)
            if len(active_channels) > 0:
                message = "active channels:\n- %s" % "\n- ".join(
                    ["%s: <#%s>" % (a[1], a[0]) for a in active_channels if a[0] != ""])
            else:
                message = "no active channel"
            await ctx.send(message)
        else:
            await ctx.send("The command is reserved for admins.")
    await bot_utils.wrap_slash_errors(ctx, bot_client, ctx.guild_id, inner)


@bot_client.command(
    name="activate",
    description="command deprecated — use the activate slash command"
)  # exists as slash command
async def activate(ctx: Context) -> None:
    async def inner() -> None:
        await ctx.send("command deprecated — use the activate slash command")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="deactivate",
    description="command deprecated — use the deactivate slash command"
)  # exists as slash command
async def deactivate(ctx: Context) -> None:
    async def inner() -> None:
        await ctx.send("command deprecated — use the activate slash command")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="channels",
    description="admin command — lists all active channels in the server"
)  # exists as slash command
async def list_active_channels(ctx: Context) -> None:
    async def inner() -> None:
        if ctx.author.id == 338067113639936003:
            logger.debug("list active channels")
            active_channels = database_client.list_active_channels(ctx.guild.id)
            if len(active_channels) > 0:
                message = "active channels:\n- %s" % "\n- ".join(
                    ["%s: <#%s>" % (a[1], a[0]) for a in active_channels if a[0] != ""])
            else:
                message = "no active channel"
            await ctx.send(message)
        else:
            await ctx.send("The command is reserved for admins.")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="set_name",
    description="test command"
)
async def set_player_discord_name(ctx: Context, discord_id: int, discord_name: str, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set player name")
        database_client.set_player_discord_name(discord_id, discord_name, polytopia_name)
        await ctx.send("Hi %s!" % discord_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="opponent",
    description="test command"
)
async def add_game_opponent(ctx: Context, discord_name: str, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set player name")
        database_client.set_player_discord_name(None, discord_name, polytopia_name)
        await ctx.send("Hi %s!" % discord_name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="set_my_name",
    description="test command"
)
async def set_self_discord_name(ctx: Context, polytopia_name: str) -> None:
    async def inner() -> None:
        logger.debug("set self player name")
        database_client.set_player_discord_name(ctx.author.id, ctx.author.name, polytopia_name)
        await ctx.send("Hi %s!" % ctx.author.name)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="turn",
    description="test command"
)
async def set_turn(ctx: Context, turn: int) -> None:
    async def inner() -> None:
        database_client.add_player_n_game(ctx.channel.id, ctx.guild.id, ctx.author.id, ctx.author.name)
        database_client.set_new_last_turn(ctx.channel.id, turn)
        await ctx.send("current turn is now %s" % str(turn))
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="size",
    description="deprecated command"
)
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
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="drop",
    description="Drop scores for the specified turn"
)
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
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="patch",
    description="command deprecated — use the patch slash command"
)
async def patch_map(ctx: Context, n_images: int = None, action_debug: bool = False) -> None:
    async def inner() -> None:
        await bot_utils.add_received_reaction(ctx.message)
        turn = database_client.get_last_turn(ctx.channel.id)
        turn, output_tuple, patching_errors = await bot_utils.generate_patched_map_bis(
            database_client,
            ctx.channel.id,
            ctx.channel.name,
            ctx.message.author.id,
            ctx.message.author.name,
            ctx.guild.id,
            ctx.message.id,
            turn,
            bot_client.loop,
            n_images,
            action_debug
        )
        if output_tuple is not None:
            patch_path, patch_uuid, patch_filename = output_tuple
            database_client.update_patching_process_output_filename(patch_uuid, patch_filename)
            with open(patch_path, "rb") as fh:
                attachment = discord.File(fp=fh, filename=patch_filename + ".png")
                if attachment is not None:
                    await ctx.channel.send(file=attachment, content="Map patched for turn %s" % turn)
                elif len(patching_errors) == 0 and attachment is None:
                    patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                    my_id = '<@338067113639936003>'  # Jean's id
                    await ctx.message.reply(
                        'There was an error. %s has been notified.' % my_id, mention_author=False)
                fh.close()
        await bot_utils.manage_patching_errors(ctx.channel, ctx.message, database_client, patching_errors)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="hello",
    description="Health check, please report to @jeanh if no response within 10 seconds"
)
async def say_hello(ctx: Context) -> None:
    async def inner():
        await ctx.send("Welcome to my botifull world!")
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="players",
    description="List players in the game"
)
async def get_channel_players(ctx: Context) -> None:
    async def inner() -> None:
        game_players = database_client.get_game_players(ctx.channel.id)
        player_frame = pd.DataFrame(game_players)
        await ctx.send(player_frame.to_string())
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="trace",
    description="Lists all map pieces used as input for patchings"
)
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
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="clear_maps",
    description="Clear the stack of maps to patch"
)
async def clear_map_reactions(ctx: Context) -> None:
    async def inner() -> None:
        await bot_utils.clear_channel_map_reactions(database_client, ctx.channel)
        await bot_utils.add_success_reaction(ctx.message)
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)


@bot_client.command(
    name="setscore",
    description="Set score for a specific player and turn"
)
async def set_player_score(ctx: Context, player_name: str, turn: int, score: int) -> None:
    async def inner() -> None:
        players = database_client.get_game_players(ctx.channel.id)
        matching_players = [p for p in players if p["polytopia_player_name"] == player_name]
        if len(matching_players) > 0:
            player_id = matching_players[0]["game_player_uuid"]
        else:
            player_id = database_client.add_missing_player(player_name, ctx.channel.id)
        scores = database_client.get_channel_scores(ctx.channel.id)
        if scores is None or len(scores[(scores["turn"] == turn) & (scores["polytopia_player_name"] == player_name)]) == 0:
            answer = database_client.add_score(ctx.channel.id, player_id, score, turn)
        else:
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
    await bot_utils.wrap_errors(ctx, ctx.guild.id, inner)
