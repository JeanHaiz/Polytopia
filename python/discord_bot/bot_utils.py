import os
import sys
import datetime
import discord
import traceback

from difflib import SequenceMatcher

from common import image_utils
from common.logger_utils import logger
from common.image_utils import ImageOp
from map_patching import map_patching_utils
from database_interaction.database_client import DatabaseClient
from score_recognition import score_recognition_utils
from score_recognition import score_visualisation


async def score_recognition_routine(database_client: DatabaseClient, message, filename):
    # TODO remove attachment from params
    channel_name = message.channel.name
    channel_id = message.channel.id

    image = await image_utils.load_image(database_client, channel_name, message, filename, ImageOp.SCORE_INPUT)
    if image is None:
        return

    # path = "./attachments/" + now() + str(message.id) + ".png"
    # await attachment.save(path)
    scores = score_recognition_utils.read_scores(image)
    turn = database_client.get_last_turn(channel_id)
    if scores is not None:
        for player, player_score in scores:
            if player == "Unknown ruler":
                database_client.add_score(channel_id, None, player_score, turn)
            else:
                if player == "Ruled by you":
                    player = message.author.name

                game_players = database_client.get_game_players(channel_id)
                print(game_players)
                name_proximity = [(
                    player_entry["discord_player_id"],
                    SequenceMatcher(
                        None,
                        player_entry["polytopia_player_name"] or player_entry["discord_player_name"] or "",
                        player).ratio())
                    for player_entry in game_players]
                print("name proximity", name_proximity)

                discord_player_id = sorted(name_proximity, key=lambda x: -x[1])[0][0]
                if discord_player_id is not None:
                    database_client.add_score(channel_id, discord_player_id, player_score, turn)
        score_text = "Scores for turn %d:\n" % turn
        score_text += "\n".join([(s[0] or "Unknown ruler") + ": " + str(s[1]) for s in scores])
        return score_text


async def map_patching_routine(database_client: DatabaseClient, attachment, message, filename):
    channel_name = message.channel.name
    channel_id = message.channel.id
    files = database_client.get_map_patching_files(channel_id)
    print("files_log %s" % str(files))
    logger.debug("files_log %s" % str(files))
    image = await image_utils.load_image(database_client, channel_name, message, filename, ImageOp.INPUT)
    turn = map_patching_utils.get_turn(image, channel_name)
    last_turn = database_client.get_last_turn(channel_id)
    if turn is None:
        turn = last_turn
    elif last_turn is None or int(last_turn) < int(turn):
        database_client.set_new_last_turn(channel_id, turn)

    map_size = database_client.get_game_map_size(channel_id)
    output_file_path = await map_patching_utils.patch_partial_maps(
        channel_name, files, map_size, database_client, message)
    if output_file_path is not None:
        return turn, image_utils.load_attachment(output_file_path, filename)


async def reaction_message_routine(database_client, message, filename):
    for attachment in message.attachments:
        if attachment.content_type.startswith("image/"):

            if score_recognition_utils.is_score_reconition_request(message.reactions, attachment, filename):
                image_utils.move_input_image(message.channel.name, filename, ImageOp.SCORE_INPUT)
                score_text = await score_recognition_routine(database_client, message, filename)
                if score_text is not None:
                    await message.channel.send(score_text)
                else:
                    await message.channel.send("Score recognition failed")

            if map_patching_utils.is_map_patching_request(message, attachment, filename):
                image_utils.move_input_image(message.channel.name, filename, ImageOp.MAP_INPUT)
                turn, patch = await map_patching_routine(database_client, attachment, message, filename)
                if patch is not None:
                    return await message.channel.send(file=patch, content="map patched for turn %d" % turn)
                else:
                    return await message.channel.send("Map patching failed")


async def reaction_removed_routine(payload, bot_client, database_client: DatabaseClient):
    if payload.emoji == discord.PartialEmoji(name="📈") or payload.emoji == discord.PartialEmoji(name="🖼️"):
        if payload.emoji == discord.PartialEmoji(name="📈"):
            source_operation = ImageOp.SCORE_INPUT
        else:
            source_operation = ImageOp.MAP_INPUT
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        filename = database_client.get_resource_filename(message, source_operation, 0)
        if filename is not None:
            image_utils.move_back_input_image(message.channel, filename, source_operation)
        # database_client.remove_resource(payload.message_id)


async def reaction_added_routine(payload, bot_client, database_client: DatabaseClient):
    if payload.emoji == discord.PartialEmoji(name="📈"):
        output = ""
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        if len(message.attachments) > 0:
            for i, attachment in enumerate(message.attachments):
                if score_recognition_utils.is_score_reconition_request(message.reactions, attachment, "filename"):
                    filename = database_client.set_resource_operation(message.id, ImageOp.INPUT, i)
                    if filename is None:
                        filename = database_client.add_resource(message, message.author, ImageOp.SCORE_INPUT, i)
                        await image_utils.save_attachment(attachment, channel.name, ImageOp.SCORE_INPUT, filename)
                    else:
                        image_utils.move_input_image(channel.name, filename, ImageOp.SCORE_INPUT)
                    score_text = await score_recognition_routine(database_client, message, filename)
                    print("score text:", score_text)
                    if score_text is not None:
                        if output != "":
                            output += "\n\n"
                        output += score_text
        if len(output) > 0:
            await channel.send(output)
        print("output", output)

    elif payload.emoji == discord.PartialEmoji(name="🖼️"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        return await process_map_patching(message, channel, database_client)

    elif payload.emoji == discord.PartialEmoji(name="⁉️"):
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        print("users", message.author, bot_client.user)
        if message.author == bot_client.user:
            myid = '<@338067113639936003>'  # Jean's id
            await message.reply(
                "Was there a small issue? Tell me more about it. Also %s has been notified." % myid,
                mention_author=False)

    else:
        print("emoji not recognised:", payload.emoji, discord.PartialEmoji(name="🖼️"))


async def process_map_patching(message, channel, database_client):
    if len(message.attachments) > 0:
        for i, attachment in enumerate(message.attachments):
            if map_patching_utils.is_map_patching_request(message, attachment, "filename"):
                filename = database_client.set_resource_operation(message.id, ImageOp.MAP_INPUT, i)
                if filename is None:
                    filename = database_client.add_resource(message, message.author, ImageOp.MAP_INPUT, i)
                    await image_utils.save_attachment(attachment, channel.name, ImageOp.MAP_INPUT, filename)
                else:
                    image_utils.move_input_image(channel.name, filename, ImageOp.MAP_INPUT)
                turn, patch = await map_patching_routine(database_client, attachment, message, filename)
                if patch is not None:
                    return await channel.send(file=patch, content="map patched for turn %s" % turn)
                else:
                    return await channel.send("patch failed")


def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')


async def get_message(bot_client, channel_id, message_id):
    channel = bot_client.get_channel(channel_id)
    return await channel.fetch_message(message_id)


async def get_attachments(bot_client, channel_id, message_id):
    message = await get_message(bot_client, channel_id, message_id)
    return message.attachments


async def wrap_errors(ctx, guild_id, fct, is_async, *params):
    try:
        is_test_server = str(guild_id) == os.getenv("POLYTOPIA_TEST_SERVER", "0")
        is_dev_env = os.getenv("POLYTOPIA_ENVIRONMENT", "") == "DEVELOPMENT"
        if (is_dev_env and is_test_server) or (not is_test_server and not is_dev_env):
            if is_async:
                return await fct(*params)
            else:
                return fct(*params)
    except Exception:
        error = sys.exc_info()[0]
        logger.error("##### ERROR #####")
        logger.error(error)
        logger.error(traceback.format_exc())
        print("##### ERROR #####")
        print(error)
        traceback.print_exc()
        myid = '<@338067113639936003>'  # Jean's id
        await ctx.reply('There was an error. %s has been notified.' % myid, mention_author=False)


async def get_scores(database_client, ctx):
    scores = database_client.get_channel_scores(ctx.channel.id)
    if scores is not None:
        scores = scores[scores['turn'] != -1]
        score_plt = await score_visualisation.plotScores(scores, ctx.channel.name, str(ctx.message.id))
        await ctx.message.channel.send(file=score_plt, content="score recognition")
        await ctx.send(str(scores))
    else:
        await ctx.send("No score found")
