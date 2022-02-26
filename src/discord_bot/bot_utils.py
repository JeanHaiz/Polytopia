import os
import sys
import datetime
import discord
import traceback
import numpy as np
import pandas as pd

from difflib import SequenceMatcher

from common import image_utils
from common.logger_utils import logger
from common.image_utils import ImageOp
from map_patching import map_patching_utils
from database_interaction.database_client import DatabaseClient
from score_recognition import score_recognition_utils
from score_recognition import score_visualisation


def find_matching(game_players, scores, author_name):
    def get_game_player_name(game_player):
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


async def score_recognition_routine(database_client: DatabaseClient, message, filename):
    # TODO remove attachment from params
    channel_name = message.channel.name
    channel_id = message.channel.id

    image = await image_utils.load_image(database_client, channel_name, message, filename, ImageOp.SCORE_INPUT)
    if image is None:
        return

    scores = score_recognition_utils.read_scores(image)
    turn = database_client.get_last_turn(channel_id)

    if scores is not None:
        game_players = database_client.get_game_players(channel_id)
        # game_players = [p["polytopia_player_name"] or p["discord_player_name"] or "" for p in game_players_raw]
        print("game players", game_players)
        # score_players = [s[0] for s in scores]

        # similarity = np.zeros((n_players, n_players))
        # for i in range(n_players):
        #     for j in range(n_players):
        #         similarity[i][j] = SequenceMatcher(None, score_players[i], game_players_other[j]).ratio()

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


async def map_patching_routine(database_client: DatabaseClient, attachment, message, filename):
    channel_name = message.channel.name
    channel_id = message.channel.id
    image = await image_utils.load_image(database_client, channel_name, message, filename, ImageOp.INPUT)
    turn = map_patching_utils.get_turn(image, channel_name)
    last_turn = database_client.get_last_turn(channel_id)
    if turn is None:
        turn = last_turn
    elif last_turn is None or int(last_turn) < int(turn):
        database_client.set_new_last_turn(channel_id, turn)
    return await generate_patched_map(database_client, channel_id, channel_name, message, turn)


async def generate_patched_map(database_client: DatabaseClient, channel_id, channel_name, message, turn):
    files = database_client.get_map_patching_files(channel_id)
    print("files_log %s" % str(files))
    logger.debug("files_log %s" % str(files))
    map_size = database_client.get_game_map_size(channel_id)
    print("map size", map_size)
    if map_size is None:
        return turn, None
    output_file_path, filename = await map_patching_utils.patch_partial_maps(
        channel_name, files, map_size, database_client, message)
    print("output path", output_file_path)
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
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        return await process_score_recognition(database_client, channel, message)

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


async def process_score_recognition(database_client, channel, message):
    if len(message.attachments) > 0:
        output = ""
        for i, attachment in enumerate(message.attachments):
            if score_recognition_utils.is_score_reconition_request(message.reactions, attachment, "filename"):
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


async def process_map_patching(message, channel, database_client):
    if len(message.attachments) > 0:
        for i, attachment in enumerate(message.attachments):
            if map_patching_utils.is_map_patching_request(message, attachment, "filename"):
                filename = await prepare_attachment(database_client, channel, message, attachment, i, ImageOp.MAP_INPUT)
                turn, patch = await map_patching_routine(database_client, attachment, message, filename)
                if patch is not None:
                    return await channel.send(file=patch, content="map patched for turn %s" % turn)
                else:
                    return await channel.send("patch failed")


async def prepare_attachment(database_client, channel, message, attachment, i, imageOp):
    filename = database_client.set_resource_operation(message.id, imageOp, i)
    if filename is None:
        filename = database_client.add_resource(message, message.author, imageOp, i)
        await image_utils.save_attachment(attachment, channel.name, imageOp, filename)
    else:
        image_utils.move_input_image(channel.name, filename, imageOp)
    return filename


def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')


async def get_message(bot_client, channel_id, message_id):
    channel = bot_client.get_channel(channel_id)
    return await channel.fetch_message(message_id)


async def get_attachments(bot_client, channel_id, message_id):
    message = await get_message(bot_client, channel_id, message_id)
    return message.attachments


async def wrap_errors(ctx, guild_id, fct, is_async, *params, **kwparams):
    try:
        is_test_server = str(guild_id) == "918195469245628446"
        is_dev_env = os.getenv("POLYTOPIA_ENVIRONMENT", "") == "DEVELOPMENT"
        # print("environment", is_test_server, is_dev_env, os.getenv("POLYTOPIA_TEST_SERVER", "0"),
        #       os.getenv("POLYTOPIA_ENVIRONMENT", ""))
        if (is_dev_env and is_test_server) or ((not is_test_server) and (not is_dev_env)):
            if is_async:
                return await fct(*params)
            else:
                return fct(*params)
    except BaseException:
        error = sys.exc_info()[0]
        logger.error("##### ERROR #####")
        logger.error(error)
        logger.error(traceback.format_exc())
        print("##### ERROR #####")
        print(error)
        traceback.print_exc()
        myid = '<@338067113639936003>'  # Jean's id
        await ctx.reply('There was an error. %s has been notified.' % myid, mention_author=False)


async def get_scores(database_client: DatabaseClient, ctx):
    scores = database_client.get_channel_scores(ctx.channel.id)
    if scores is not None and len(scores[scores['turn'] != -1]) > 0:
        scores: pd.DataFrame = scores[scores['turn'] != -1]
        score_plt = score_visualisation.plotScores(scores, ctx.channel.name, str(ctx.message.id))
        await ctx.message.channel.send(file=score_plt, content="score recognition")
        score_text = score_visualisation.print_scores(scores)
        # await ctx.send(score_text)
        embed = discord.Embed(title='Game scores', description=score_text)
        await ctx.send(embed=embed)
    else:
        await ctx.send("No score found")


async def get_player_scores(database_client, ctx, player):
    scores = database_client.get_channel_scores(ctx.channel.id)
    if player is not None and player not in scores["polytopia_player_name"].unique():
        players = scores["polytopia_player_name"].unique()
        players.pop(None)
        await ctx.send("Player %s not recognised. Available players: %s" % (str(player), str(players)))
    else:
        score_text = score_visualisation.print_player_scores(scores, player)
        embed = discord.Embed(title='%s scores' % str(player), description=score_text)
        await ctx.send(embed=embed)
