import os
import sys
import asyncio
import discord
import datetime
import functools
import traceback

import numpy as np
import pandas as pd
import concurrent.futures

from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import Coroutine
from difflib import SequenceMatcher
from discord import File
from discord import Embed
from discord.ext.commands import Bot
from discord.ext.commands import Context

from common import image_utils
from common.logger_utils import logger
from common.image_operation import ImageOp
from map_patching import map_patching_utils
from map_patching import header_recognition
from map_patching.map_patching_analysis import DEBUG
from map_patching.map_patching_analysis import analyse_map
from map_patching.map_patching_patching import patch_processed_images
from database_interaction.database_client import DatabaseClient
from map_patching.map_patching_errors import MapPatchingErrors
from map_patching.map_patching_errors import MAP_PATCHING_ERROR_MESSAGES
from score_recognition import score_recognition_utils
from score_recognition import score_visualisation


def find_matching(
        game_players: List[dict],
        scores: List[Tuple[str, int]],
        author_name: str) -> Tuple[list, list]:

    def get_game_player_name(game_player: Dict) -> str:
        return game_player["polytopia_player_name"] or game_player["discord_player_name"] or ""

    score_players = [
        (None, s[1]) if s[0] == "Unknown ruler"
        else ((author_name, s[1]) if s[0] == "Ruled by you" else s)
        for s in scores]

    similarity = np.zeros((len(score_players), len(game_players)))

    for i in range(len(score_players)):
        for j in range(len(game_players)):
            print("pre-text matching", score_players[i], game_players[j])
            print("text matching", score_players[i][0], get_game_player_name(game_players[j]))
            if score_players[i][0] is None:
                sim_ij = 0
            else:
                sim_ij = SequenceMatcher(None, score_players[i][0], get_game_player_name(game_players[j])).ratio()
            similarity[i, j] = sim_ij

    output = []
    while len(similarity) > 0 and len(similarity[0]) > 0:
        index = np.where(similarity == np.amax(similarity))
        index_max = index[0][0], index[1][0]

        if similarity[index_max[0], index_max[1]] == 0:
            if len(game_players) != 1 or len(score_players) != 1:
                break

        output.append((
            game_players[index_max[1]]["game_player_uuid"],
            score_players[index_max[0]][0],
            score_players[index_max[0]][1]))
        game_players.pop(index_max[1])
        score_players.pop(index_max[0])

        similarity = np.delete(np.delete(similarity, index_max[0], 0), index_max[1], 1)

    # Handle the case where there are more than 1 Unknown ruler

    return output, score_players


async def score_recognition_routine(
        database_client: DatabaseClient,
        message: discord.Message,
        filename: str) -> Optional[str]:

    # TODO remove attachment from params
    channel_name = message.channel.name
    channel_id = message.channel.id

    image = await image_utils.load_or_fetch_image(database_client, channel_name, message, filename, ImageOp.SCORE_INPUT)
    if image is None:
        return None

    scores = score_recognition_utils.read_scores(image)
    turn = database_client.get_last_turn(channel_id) or 0

    if scores is None:
        return None

    game_players = database_client.get_game_players(channel_id)
    print("game players", game_players)

    matching, remaining_scores = find_matching(game_players, scores, message.author.name)

    for player_uuid, player_name, player_score in matching:
        database_client.set_player_game_name(player_uuid, player_name)
        database_client.add_score(channel_id, player_uuid, player_score, turn)

    for player_name, player_score in remaining_scores:
        if player_name is not None and player_name != '':
            player_uuid = database_client.add_missing_player(player_name, channel_id)
        else:
            player_uuid = None
        database_client.add_score(channel_id, player_uuid, player_score, turn)

    score_text = "Scores for turn %d:\n" % turn
    score_text += "\n".join([(s[0] or "Unknown ruler") + ": " + str(s[1]) for s in scores])
    return score_text


async def map_patching_routine(
        database_client: DatabaseClient,
        message: discord.Message,
        image: np.ndarray,
        loop: asyncio.AbstractEventLoop,
        action_debug: bool) -> Tuple[Optional[str], Optional[Tuple[str, str, str]], list]:

    channel_name = message.channel.name
    channel_id = message.channel.id
    turn = header_recognition.get_turn(image, channel_name=channel_name)
    last_turn = database_client.get_last_turn(channel_id)
    if turn is None:
        turn = last_turn
    elif last_turn is None or int(last_turn) < int(turn):
        database_client.set_new_last_turn(channel_id, turn)
    return await generate_patched_map_bis(
        database_client,
        channel_id,
        channel_name,
        message.author.id,
        message.author.name,
        message.guild.id,
        message.id,
        turn,
        loop,
        None,
        action_debug)


async def manage_patching_errors(
        channel: discord.channel,
        original_message: discord.message,
        database_client: DatabaseClient,
        patching_errors: list) -> None:

    for cause, error_filename in patching_errors:
        if error_filename is None:
            await original_message.reply(MAP_PATCHING_ERROR_MESSAGES[cause])
        else:
            channel_id, message_id = database_client.get_resource_message(error_filename)
            if channel_id is not None and message_id is not None:
                try:
                    message = await channel.fetch_message(message_id)
                except discord.errors.NotFound:
                    message = original_message
                await message.reply(MAP_PATCHING_ERROR_MESSAGES[cause], mention_author=False)


async def generate_patched_map_bis(
        database_client: DatabaseClient,
        channel_id: int,
        channel_name: str,
        author_id: int,
        author_name: str,
        guild_id: int,
        interaction_id: int,
        turn: Optional[str],
        loop: asyncio.AbstractEventLoop,
        n_images: Optional[int] = 4,
        action_debug: bool = False) -> Tuple[Optional[str], Optional[Tuple[str, str, str]], list]:
    patch_uuid = database_client.add_patching_process(channel_id, author_id)
    map_size = database_client.get_game_map_size(channel_id)

    if map_size is None:
        return turn, None, [(MapPatchingErrors.MISSING_MAP_SIZE, None)]

    files = database_client.get_map_patching_files(channel_id)
    if len(files) == 0:
        return turn, None, [(MapPatchingErrors.NO_FILE_FOUND, None)]
    elif len(files) == 1:
        return turn, None, [(MapPatchingErrors.ONLY_ONE_FILE, None)]

    if n_images is not None:
        n_images = max(n_images, 2)
        files = files[-n_images:]

    if DEBUG or action_debug:
        print("files_log %s" % str(files))
    logger.debug("files_log %s" % str(files))

    for i, filename_i in enumerate(files):
        database_client.add_patching_process_input(patch_uuid, filename_i, i)

    func = functools.partial(
        patch_processed_images, files, map_size, guild_id, channel_id, channel_name,
        interaction_id, author_id, author_name, action_debug)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        output_file_path, filename, patching_errors = await loop.run_in_executor(pool, func)

    if len(patching_errors) == 0:
        status = "DONE"
    else:
        status = "ERRORS - " + "; ".join(
            [str(str(error.name) + "(" + str(filename) + ")") for error, filename in patching_errors])
    print("status", status)
    database_client.update_patching_process_status(patch_uuid, status)
    return turn, (output_file_path, patch_uuid, filename), patching_errors


async def reaction_message_routine(
        bot_client: Bot,
        database_client: DatabaseClient,
        message: discord.Message,
        filename: str) -> None:
    for attachment in message.attachments:
        if attachment.content_type.startswith("image/"):

            if score_recognition_utils.is_score_recognition_request(message.reactions, attachment, filename):
                image_utils.move_input_image(message.channel.name, filename, ImageOp.SCORE_INPUT)
                score_text = await score_recognition_routine(database_client, message, filename)
                if score_text is not None:
                    await message.channel.send(score_text)
                else:
                    await message.channel.send("Score recognition failed")

            if map_patching_utils.is_map_patching_request(message, attachment, filename):
                image_utils.move_input_image(message.channel.name, filename, ImageOp.MAP_INPUT)
                image = await image_utils.load_or_fetch_image(
                    database_client, message.channel.name, message, filename, ImageOp.INPUT)
                analyse_map(
                    image, database_client, message.channel.name, message.channel.id, filename, action_debug=False)

                turn, output_tuple, patching_errors = await map_patching_routine(
                    database_client, message, image, asyncio.get_event_loop(), False)
                
                if output_tuple is not None:
                    patch_path, patch_uuid, patch_filename = output_tuple
                    database_client.update_patching_process_output_filename(patch_uuid, patch_filename)
                    with open(patch_path, "rb") as fh:
                        attachment = File(fp=fh, filename=patch_filename + ".png")
                        if attachment is not None:
                            await message.channel.send(file=attachment, content="Map patched for turn %s" % turn)
                        elif len(patching_errors) == 0 and attachment is None:
                            patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                            my_id = '<@338067113639936003>'  # Jean's id
                            await message.reply(
                                'There was an error. %s has been notified.' % my_id, mention_author=False)
                        fh.close()
                await manage_patching_errors(message.channel, message, database_client, patching_errors)


async def reaction_removed_routine(
        payload: discord.RawReactionActionEvent,
        bot_client: Bot,
        database_client: DatabaseClient) -> None:

    if payload.emoji == discord.PartialEmoji(name="📈") or payload.emoji == discord.PartialEmoji(name="🖼️"):
        if payload.emoji == discord.PartialEmoji(name="📈"):
            source_operation = ImageOp.SCORE_INPUT
        else:
            source_operation = ImageOp.MAP_INPUT
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        reset_resource(database_client, channel, message.id, source_operation)


def reset_resource(
        database_client: DatabaseClient,
        channel: discord.TextChannel,
        message_id: int,
        source_operation: ImageOp) -> None:

    filename = database_client.set_resource_operation(message_id, ImageOp.INPUT, 0)
    if filename is not None:
        image_utils.move_back_input_image(channel.name, filename, source_operation)


async def reaction_added_routine(
        payload: discord.RawReactionActionEvent,
        bot_client: Bot,
        database_client: DatabaseClient) -> None:

    if payload.emoji == discord.PartialEmoji(name="📈"):
        channel: discord.TextChannel = bot_client.get_channel(payload.channel_id)
        message: discord.Message = await channel.fetch_message(payload.message_id)
        await add_received_reaction(message)
        await process_score_recognition(database_client, channel, message)

    elif payload.emoji == discord.PartialEmoji(name="🖼️"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        await add_received_reaction(message)
        await process_map_patching(bot_client, message, channel, database_client, False)

    elif payload.emoji == discord.PartialEmoji(name="⁉️"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        await add_received_reaction(message)
        print("users", message.author, bot_client.user)
        if message.author == bot_client.user:
            await message.reply(
                "Was there a small issue? Tell me more about it. Also <@338067113639936003> has been notified.",
                mention_author=False)

    elif payload.emoji == discord.PartialEmoji(name="🗑"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        if message.author.id == bot_client.user.id:
            if message.reference is not None and message.reference.message_id is not None \
                    and message.content.startswith("MAP_INPUT"):
                reset_resource(database_client, message.channel, message.reference.message_id, ImageOp.MAP_INPUT)
                reset_message: discord.Message = await channel.fetch_message(message.reference.message_id)
                await clear_map_reaction(reset_message)

            elif message.content.startswith("Message not found: "):
                reset_message_id = int(message.content[len("Message not found: "):])
                reset_resource(database_client, message.channel, reset_message_id, ImageOp.MAP_INPUT)
            await message.delete()

    elif payload.emoji == discord.PartialEmoji(name="🔄"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        await add_received_reaction(message)
        await refresh_image_analysis(channel, message, database_client, bot_client)
        await message.remove_reaction(payload.emoji, payload.member)


async def refresh_image_analysis(
        channel: discord.TextChannel,
        message: discord.Message,
        database_client: DatabaseClient,
        bot_client: Bot) -> None:

    filename = database_client.get_resource_filename(channel.id, message.id, ImageOp.MAP_INPUT, 0)
    if filename is None:
        print("None filename for channel %s and message %s" % (str(channel.id), str(message.id)))
        return
    else:
        print("Filename found:", filename)

    image = await image_utils.load_or_fetch_image(
        database_client, message.channel.name, message, filename, ImageOp.INPUT)
    analyse_map(image, database_client, message.channel.name, message.channel.id, filename, action_debug=False)
    await process_map_patching(bot_client, message, channel, database_client, True)


async def process_score_recognition(
        database_client: DatabaseClient,
        channel: discord.TextChannel,
        message: discord.Message) -> None:

    if len(message.attachments) == 1:
        output = ""
        for i, attachment in enumerate(message.attachments):
            if score_recognition_utils.is_score_recognition_request(message.reactions, attachment, "filename"):
                filename = await prepare_attachment(
                    database_client, channel, message, attachment, i, ImageOp.SCORE_INPUT)
                score_text = await score_recognition_routine(database_client, message, filename)
                print("score text:", score_text)
                if score_text is not None:
                    if output != "":
                        output += "\n\n"
                    output += score_text
        if len(output) > 0:
            await channel.send(output)
        print("output", output)
    elif len(message.attachments) > 1:
        await message.reply("Only one image per message is currently supported.", mention_author=False)


async def process_map_patching(
        bot_client: Bot,
        message: discord.Message,
        channel: discord.TextChannel,
        database_client: DatabaseClient,
        action_debug: bool) -> None:

    if len(message.attachments) == 1:
        for i, attachment in enumerate(message.attachments):
            if map_patching_utils.is_map_patching_request(message, attachment, "filename"):
                filename = await prepare_attachment(database_client, channel, message, attachment, i, ImageOp.MAP_INPUT)
                image = await image_utils.load_or_fetch_image(
                    database_client, channel.name, message, filename, ImageOp.INPUT)

                turn, output_tuple, patching_errors = await map_patching_routine(
                    database_client, message, image, asyncio.get_event_loop(), False)

                if output_tuple is not None:
                    patch_path, patch_uuid, patch_filename = output_tuple
                    database_client.update_patching_process_output_filename(patch_uuid, patch_filename)
                    with open(patch_path, "rb") as fh:
                        attachment = discord.File(fp=fh, filename=patch_filename + ".png")
                        if attachment is not None:
                            await message.channel.send(file=attachment, content="Map patched for turn %s" % turn)
                        elif len(patching_errors) == 0 and attachment is None:
                            patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                            my_id = '<@338067113639936003>'  # Jean's id
                            await message.reply(
                                'There was an error. %s has been notified.' % my_id, mention_author=False)
                        fh.close()
                await manage_patching_errors(message.channel, message, database_client, patching_errors)
    elif len(message.attachments) > 1:
        await message.reply("Only one image per message is currently supported.", mention_author=False)


async def prepare_attachment(
        database_client: DatabaseClient,
        channel: discord.TextChannel,
        message: discord.Message,
        attachment: discord.Attachment,
        i: int,
        image_op: ImageOp) -> str:

    filename = database_client.set_resource_operation(message.id, image_op, i)
    if filename is None:
        filename = database_client.add_resource(
            message.guild.id, channel.id, message.id, message.author.id, message.author.name, image_op, i)
        await image_utils.save_attachment(attachment, channel.name, image_op, filename)
    else:
        image_utils.move_input_image(channel.name, filename, image_op)
    return filename


def now() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')


async def get_message(bot_client: Bot, channel_id: int, message_id: int) -> Optional[discord.Message]:
    try:
        channel = bot_client.get_channel(channel_id)
        return await channel.fetch_message(message_id)
    except discord.errors.NotFound:
        return None


async def wrap_errors(
        ctx: Union[Context, discord.Message],
        bot_client: Bot,
        guild_id: int,
        fct: Callable[[], Coroutine]) -> None:
    try:
        is_test_server = str(guild_id) == "918195469245628446"
        is_dev_env = os.getenv("POLYTOPIA_ENVIRONMENT", "") == "DEVELOPMENT"
        # print("environment", is_test_server, is_dev_env, os.getenv("POLYTOPIA_TEST_SERVER", "0"),
        #       os.getenv("POLYTOPIA_ENVIRONMENT", ""))
        if (is_dev_env and is_test_server) or (not is_test_server and not is_dev_env):
            await asyncio.create_task(fct())
    except discord.errors.Forbidden:
        await ctx.reply("Missing permission. <@338067113639936003> has been notified.", mention_author=False)
    except:
        error = sys.exc_info()[0]
        logger.error("##### ERROR #####")
        logger.error(error)
        logger.error(traceback.format_exc())
        print("##### ERROR #####")
        print(error)
        traceback.print_exc()
        error_channel = bot_client.get_channel(1035274340125659230)  # Polytopia Helper Testing server, Error channel
        channel = bot_client.get_channel(ctx.channel.id)
        guild = await ctx.get_guild()
        await error_channel.send(f"""Error in channel {channel.name}, {guild.name}:\n{traceback.format_exc()}\n""")
        await ctx.reply('There was an error. <@338067113639936003> has been notified.', mention_author=False)


async def get_scores(database_client: DatabaseClient, ctx: Context) -> None:
    scores: pd.DataFrame = database_client.get_channel_scores(ctx.channel.id)
    if scores is not None and len(scores[scores['turn'] != -1]) > 0:
        scores = scores[scores['turn'] != -1]
        filepath, filename = score_visualisation.plot_scores(
            database_client, scores, ctx.channel.id, ctx.channel.name, ctx.author.id)
        with open(filepath, "rb") as fh:
            attachment = File(fp=fh, filename=filename + ".png")
            image_utils.load_attachment(filepath, "Score visualisation")
            await ctx.channel.send(files=[attachment], content="score recognition")

        score_text = score_visualisation.print_scores(scores)
        # await ctx.send(score_text)
        embed = discord.Embed(title='Game scores', description=score_text)
        await ctx.send(embeds=[embed])
    else:
        await ctx.send("No score found")


async def get_player_scores(database_client: DatabaseClient, ctx: Context, player: str) -> None:
    scores = database_client.get_channel_scores(ctx.channel.id)
    if scores is not None and len(scores[scores['turn'] != -1]) > 0:
        if player is not None and player not in scores["polytopia_player_name"].unique():
            players = scores["polytopia_player_name"].unique()
            # players.pop(None)
            await ctx.send("Player %s not recognised. Available players: %s" % (str(player), str(players)))
        else:
            score_text = score_visualisation.print_player_scores(scores, player)
            embed = Embed(title='%s scores' % str(player), description=score_text)
            await ctx.send(embeds=[embed])
    else:
        await ctx.send("No score found for player %s" % str(player))


async def add_success_reaction(message: discord.Message) -> None:
    await message.add_reaction("✅")


async def add_received_reaction(message: discord.Message) -> None:
    await message.add_reaction("📩")


async def add_error_reaction(message: discord.Message) -> None:
    await message.add_reaction("🚫")


async def add_delete_reaction(message: discord.Message) -> None:
    await message.add_reaction("🗑")


async def clear_channel_map_reactions(
        database_client: DatabaseClient,
        channel: discord.TextChannel,
        fct: Callable[[], Coroutine[Any, Any, Any]]) -> None:
    messages_ids = database_client.get_channel_resource_messages(channel.id, ImageOp.MAP_INPUT)

    for m_id in messages_ids:
        message: discord.Message = await channel.fetch_message(m_id['source_message_id'])
        await clear_map_reaction(message)
        database_client.set_resource_operation(m_id['source_message_id'], ImageOp.INPUT, 0)
    await fct()


async def clear_map_reaction(message: discord.Message) -> None:
    await message.clear_reaction("🖼️")


async def renew_patching(dry_run=False):
    database_client = DatabaseClient(
        user="discordBot", password="password123", port="5432", database="polytopiaHelper_dev",
        host="database")
    incomplete_channel_list = database_client.get_incomplete_patching_run()

    for incomplete_run in incomplete_channel_list:
        print(incomplete_run)
        # channel_details = database_client.get_channel_info(channel_id)
