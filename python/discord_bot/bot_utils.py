import datetime
import discord


from common import image_utils
from common.logger_utils import logger
from common.image_utils import ImageOperation
from map_patching import map_patching_utils
from database_interaction.database_client import DatabaseClient
from score_recognition import score_recognition_utils


async def score_recognition_routine(database_client: DatabaseClient, attachment, message, filename):
    # TODO remove attachment from params
    # TODO swap message for channel

    image = await image_utils.load_image(database_client, message, filename, ImageOperation.INPUT)
    # path = "./attachments/" + now() + str(message.id) + ".png"
    # await attachment.save(path)
    scores = score_recognition_utils.read_scores(image)
    for player, player_score in scores:
        discord_player = database_client.get_player(message.channel, player)
        if discord_player is not None:
            database_client.add_score(message.channel, discord_player, player_score)
    score_text = "Scores:\n"
    score_text += "\n".join([s[0] + ": " + str(s[1]) for s in scores])
    return score_text


async def map_patching_routine(database_client: DatabaseClient, attachment, message, filename):

    files = database_client.get_map_patching_files(message.channel)
    print("files_log %s" % str(files))
    logger.debug("files_log %s" % str(files))
    output_file_path = await map_patching_utils.patch_partial_maps(message, files, database_client)
    if output_file_path is not None:
        return image_utils.load_attachment(output_file_path, filename)


async def reaction_message_routine(database_client, message, filename):
    for i, attachment in enumerate(message.attachments):
        if attachment.content_type.startswith("image/"):

            if score_recognition_utils.is_score_reconition_request(message, attachment, filename):
                score_text = await score_recognition_routine(database_client, attachment, message, filename)
                await message.channel.send(score_text)

            if map_patching_utils.is_map_patching_request(message, attachment, filename):
                patch = await map_patching_routine(database_client, attachment, message, filename)
                if patch is not None:
                    return await message.channel.send(file=patch, content="map patched")
                else:
                    return await message.channel.send("patch failed")


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
                                                                image_utils.ImageOperation.INPUT, i)
                        image_utils.save_attachment(attachment, message, ImageOperation.INPUT, filename)
                    score_text = await score_recognition_routine(database_client, attachment, message, filename)
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
                    filename = database_client.set_resource_operation(message, ImageOperation.INPUT, i)
                    patch = await map_patching_routine(database_client, attachment, message, filename)
                    if patch is not None:
                        return await channel.send(file=patch, content="map patched")
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
