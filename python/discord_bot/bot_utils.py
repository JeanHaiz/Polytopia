import datetime
import discord

from difflib import SequenceMatcher

from common import image_utils
from common.logger_utils import logger
from common.image_utils import ImageOperation
from map_patching import map_patching_utils
from database_interaction.database_client import DatabaseClient
from score_recognition import score_recognition_utils


async def score_recognition_routine(database_client: DatabaseClient, message, filename):
    # TODO remove attachment from params

    image = await image_utils.load_image(database_client, message, filename, ImageOperation.INPUT)
    # path = "./attachments/" + now() + str(message.id) + ".png"
    # await attachment.save(path)
    scores = score_recognition_utils.read_scores(image)
    turn = database_client.get_last_turn(message.channel.id)

    for player, player_score in scores:
        if player == "Unknown ruler":
            database_client.add_score(message.channel, None, player_score, turn)
        else:
            if player == "Ruled by you":
                player = message.author.name

            game_players = database_client.get_game_players(message.channel)
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
                database_client.add_score(message.channel, discord_player_id, player_score, turn)
    score_text = "Scores:\n"
    score_text += "\n".join([(s[0] or "Unknown ruler") + ": " + str(s[1]) for s in scores])
    return score_text


async def map_patching_routine(database_client: DatabaseClient, attachment, message, filename):

    files = database_client.get_map_patching_files(message.channel)
    print("files_log %s" % str(files))
    logger.debug("files_log %s" % str(files))
    image = await image_utils.load_image(database_client, message, filename, image_utils.ImageOperation.INPUT)
    turn = map_patching_utils.get_turn(image)
    last_turn = database_client.get_last_turn(message.channel.id)
    if turn is None:
        turn = last_turn
    elif last_turn is None or int(last_turn) < int(turn):
        database_client.set_new_last_turn(message.channel.id, turn)
    output_file_path = await map_patching_utils.patch_partial_maps(message, files, database_client)
    if output_file_path is not None:
        return turn, image_utils.load_attachment(output_file_path, filename)


async def reaction_message_routine(database_client, message, filename):
    for i, attachment in enumerate(message.attachments):
        if attachment.content_type.startswith("image/"):

            if score_recognition_utils.is_score_reconition_request(message, attachment, filename):
                score_text = await score_recognition_routine(database_client, message, filename)
                await message.channel.send(score_text)

            if map_patching_utils.is_map_patching_request(message, attachment, filename):
                turn, patch = await map_patching_routine(database_client, attachment, message, filename)
                if patch is not None:
                    return await message.channel.send(file=patch, content="map patched for turn %d" % turn)
                else:
                    return await message.channel.send("patch failed")


def reaction_removed_routine(payload, bot_client, database_client: DatabaseClient):
    if payload.emoji == discord.PartialEmoji(name="📈") or payload.emoji == discord.PartialEmoji(name="🖼️"):
        database_client.remove_resource(payload.message_id)


async def reaction_added_routine(payload, bot_client, database_client: DatabaseClient):
    if payload.emoji == discord.PartialEmoji(name="📈"):
        output = ""
        channel = bot_client.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        if len(message.attachments) > 0:
            for i, attachment in enumerate(message.attachments):
                if score_recognition_utils.is_score_reconition_request(message, attachment, "filename"):
                    filename = database_client.set_resource_operation(message, ImageOperation.INPUT, i)
                    if filename is None:
                        filename = database_client.add_resource(message, message.author,
                                                                image_utils.ImageOperation.SCORE_INPUT, i)
                        await image_utils.save_attachment(attachment, message, ImageOperation.SCORE_INPUT, filename)
                    else:
                        image_utils.move_input_image(message.channel, filename, ImageOperation.SCORE_INPUT)
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
        if len(message.attachments) > 0:
            for i, attachment in enumerate(message.attachments):
                if map_patching_utils.is_map_patching_request(message, attachment, "filename"):
                    filename = database_client.set_resource_operation(message, ImageOperation.MAP_INPUT, i)
                    if filename is None:
                        filename = database_client.add_resource(message, message.author,
                                                                image_utils.ImageOperation.MAP_INPUT, i)
                        await image_utils.save_attachment(attachment, message, ImageOperation.MAP_INPUT, filename)
                    else:
                        image_utils.move_input_image(message.channel, filename, ImageOperation.MAP_INPUT)
                    turn, patch = await map_patching_routine(database_client, attachment, message, filename)
                    if patch is not None:
                        return await channel.send(file=patch, content="map patched for turn %s" % turn)
                    else:
                        return await channel.send("patch failed")

    else:
        print("emoji not recognised:", payload.emoji, discord.PartialEmoji(name="🖼️"))


def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')


async def get_message(bot_client, channel_id, message_id):
    channel = bot_client.get_channel(channel_id)
    return await channel.fetch_message(message_id)


async def get_attachments(bot_client, channel_id, message_id):
    message = await get_message(bot_client, channel_id, message_id)
    return message.attachments
