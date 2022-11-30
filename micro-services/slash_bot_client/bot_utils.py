import os
import sys
import asyncio
import datetime
import traceback

from io import BytesIO

from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
from typing import Coroutine

from interactions import Channel
from interactions import HTTPClient
from interactions import Optional
from interactions import Client
from interactions import CommandContext
from interactions import Message
from interactions import Snowflake
from interactions import get

from database.database_client import DatabaseClient
from slash_bot_client import command_context_store as cxs
from slash_bot_client import bot_input_utils
from slash_bot_client import map_analysis_interface
from slash_bot_client import score_service_interface
from slash_bot_client import header_footer_recognition_interface
from common import image_utils
from common.logger_utils import logger
from common.image_operation import ImageOp

"""
Supports generic bot activities
- manages image downloads and registration
- registers ongoing processes
- calls specific pipelines or services
- unwraps discord objects
- catches process errors
"""

DEBUG = os.getenv("POLYTOPIA_DEBUG")

database_client = DatabaseClient(
    user="discordBot",
    password="password123",
    port="5432",
    database="polytopiaHelper_dev",
    host="database"
)


async def __download_attachment(
        channel: Channel,
        message_id: int,
        resource_number: int,
        bot_client: HTTPClient
) -> BytesIO:
    message = await channel.get_message(message_id)
    attachment = message.attachments[resource_number]
    attachment._client = bot_client
    return await attachment.download()


async def add_map_and_patch(
        ctx: CommandContext,
        bot_http_client: HTTPClient
):
    
    # Fulfilling precondition: Game setup
    database_client.add_player_n_game(
        ctx.channel_id,
        ctx.guild_id,
        ctx.author.id,
        ctx.author.name
    )
    
    # Helper variables
    channel = await ctx.get_channel()
    message = ctx.target
    n_resources = len(message.attachments)
    
    # Registering message attachments
    for resource_number in range(n_resources):
        message.attachments[resource_number]._client = bot_http_client

        await bot_input_utils.get_or_register_input(
            database_client,
            lambda i: message.attachments[i].download(),
            channel,
            message.id,
            message,
            resource_number,
            ImageOp.MAP_INPUT
        )

    # Patching images registered for channel
    await patch_images(ctx, bot_http_client, channel)


async def patch_images(
        ctx: CommandContext,
        bot_http_client: HTTPClient,
        channel: Channel,
):
    patch_process_id = database_client.add_patching_process(
        ctx.channel_id,
        ctx.author.id,
        ctx.id
    )

    cxs.put(patch_process_id, ctx)

    requirements = []
    
    message_resources = database_client.get_channel_resource_messages(
        ctx.channel_id,
        [ImageOp.MAP_INPUT, ImageOp.MAP_PROCESSED_IMAGE]
    )
    
    for message_resource_i in message_resources:
        
        message_id = message_resource_i["source_message_id"]
        resource_number = message_resource_i["resource_number"]

        requirements_i = await register_process_inputs(
            lambda i: __download_attachment(channel, message_id, resource_number, bot_http_client),
            patch_process_id,
            channel,
            message_id,
            None,
            resource_number
        )
        
        requirements.extend(requirements_i)
        
    print("requirements", requirements, flush=True)
    
    for requirement_i in requirements:
        
        await analyse_map(
            patch_process_id,
            requirement_i[0],
            requirement_i[1],
            requirement_i[2],
            requirement_i[3]
        )


async def register_process_inputs(
        download_fct: Callable[[int], Coroutine[Any, Any, BytesIO]],
        patch_process_id: str,
        channel: Channel,
        message_id: int,
        message: Optional[Message],
        resource_number: int
) -> List[Tuple[int, int, str, str]]:
    
    resource = database_client.get_resource(message_id, resource_number)
    filename = str(resource["filename"])
    
    """filename = await bot_input_utils.get_or_register_input(
        database_client,
        download_fct,
        channel,
        message_id,
        message,
        resource_number,
        ImageOp.MAP_INPUT
    )"""
    
    if filename is None:
        return []

    requirements = []
    
    turn_requirement_id = database_client.add_patching_process_requirement(
        patch_process_id,
        filename,
        "TURN"
    )
    requirements.append((message_id, resource_number, turn_requirement_id, "TURN"))

    map_requirement_id = database_client.add_patching_process_requirement(
        patch_process_id,
        filename,
        "MAP"
    )
    requirements.append((message_id, resource_number, map_requirement_id, "MAP"))
    
    return requirements
    

async def analyse_map(
        patch_process_id: str,
        message_id: int,
        resource_number: int,
        requirement_id: str,
        requirement_type: str
):
    if requirement_type == "TURN":
        await header_footer_recognition_interface.get_or_recognise_turn(
            patch_process_id,
            requirement_id,
            message_id,
            resource_number
        )
    elif requirement_type == "MAP":
        await map_analysis_interface.get_or_analyse_map(
            patch_process_id,
            requirement_id,
            message_id,
            resource_number
        )


async def remove_map(ctx: CommandContext):
    message_resources = database_client.get_channel_message_resource_messages(
        ctx.channel_id,
        ctx.target.id,
        ImageOp.MAP_INPUT
    )
    
    channel = await ctx.get_channel()
    
    for message_resource_i in message_resources:
        
        resource_number = message_resource_i["resource_number"]
        operation = message_resource_i["operation"]
        
        filename = database_client.set_resource_operation(
            ctx.message_id,
            ImageOp.INPUT,
            resource_number
        )
        
        if filename is not None:
            image_utils.move_back_input_image(
                channel.name,
                filename,
                ImageOp(operation)
            )


async def force_analyse_map_and_patch(
        ctx: CommandContext,
        bot_http_client: HTTPClient
):
    # Fulfilling precondition: Game setup
    database_client.add_player_n_game(
        ctx.channel_id,
        ctx.guild_id,
        ctx.author.id,
        ctx.author.name
    )
    
    # Helper variables
    channel = await ctx.get_channel()
    message = ctx.target
    n_resources = len(message.attachments)
    
    # Registering message attachments
    for resource_number in range(n_resources):
        message.attachments[resource_number]._client = bot_http_client
        
        await bot_input_utils.get_or_register_input(
            database_client,
            lambda i: message.attachments[i].download(),
            channel,
            message.id,
            message,
            resource_number,
            ImageOp.MAP_INPUT
        )
    
    # Patching images registered for channel
    await patch_images(ctx, bot_http_client, channel)


async def list_active_channels(ctx: CommandContext) -> None:
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


async def set_player_score(ctx: CommandContext, player_name: str, turn: int, score: int) -> None:
    players = database_client.get_game_players(
        ctx.channel_id
    )
    
    matching_players = [p for p in players if p["polytopia_player_name"] == player_name]
    
    if len(matching_players) > 0:
        player_id = matching_players[0]["game_player_uuid"]
    else:
        player_id = database_client.add_missing_player(
            player_name,
            ctx.channel_id
        )
    
    scores = database_client.get_channel_scores(
        ctx.channel_id
    )
    
    if scores is None or \
            len(scores[(scores["turn"] == turn) & (scores["polytopia_player_name"] == player_name)]) == 0:
        answer = database_client.add_score(ctx.channel_id, player_id, score, turn)
    else:
        answer = database_client.set_player_score(player_id, turn, score)
    
    row_count = answer.rowcount
    
    if row_count == 1:
        await ctx.send("Score stored")
    elif row_count == 0:
        await ctx.send("No score entry was updated. \nTo to signal an error, react with ⁉️")
    else:
        jean_id = '<@338067113639936003>'  # Jean's discord id TODO refactor
        await ctx.send('There was an error. %s has been notified.' % jean_id)


async def patch_map(
        ctx: CommandContext,
        bot_http_client: HTTPClient,
        number_of_images: int
) -> None:
    
    channel = await ctx.get_channel()
    await patch_images(
        ctx,
        bot_http_client,
        channel
    )
    
    """
    patch_process_id = database_client.add_patching_process(
        ctx.channel_id,
        ctx.author.id,
        ctx.id
    )
    
    await ctx.send("Loading")
    
    cxs.put(patch_process_id, ctx)
    
    resources = database_client.get_channel_resource_messages(ctx.channel_id, ImageOp.MAP_INPUT)

    for resource_i in resources:
        message_id = resource_i["source_message_id"]
        message = await channel.get_message(message_id)
        resource_number = resource_i["resource_number"]
        
        await analyse_map(
            patch_process_id,
            message_id,
            resource_number
        )
    
    # Not needed to trigger the patching if an analysis is done
    # await map_patching_interface.send_map_patching_request(patch_process_id, number_of_images)
    """


async def activate(ctx: CommandContext, size: int) -> None:
    logger.debug("activate channel %s" % ctx.channel_id)
    channel = await ctx.get_channel()
    guild = await ctx.get_guild()
    activation_result = database_client.activate_channel(ctx.channel_id, channel.name, ctx.guild_id, guild.name)
    database_client.set_game_map_size(ctx.channel_id, int(size))
    if activation_result.rowcount == 1:
        await ctx.send("channel activated")
    else:
        my_id = '<@338067113639936003>'  # Jean's id
        await ctx.send('There was an error. %s has been notified.' % my_id)


async def deactivate(ctx: CommandContext) -> None:
    logger.debug("deactivate channel %s" % ctx.channel_id)
    deactivation_result = database_client.deactivate_channel(ctx.channel_id)
    if deactivation_result.rowcount == 1:
        await ctx.send("channel deactivated")
    else:
        my_id = '<@338067113639936003>'  # Jean's id
        await ctx.send('There was an error. %s has been notified.' % my_id)


async def get_channel_player_score(ctx: CommandContext, player: str):
    if player is None:
        await score_service_interface.get_scores(
            database_client,
            ctx
        )
    else:
        await score_service_interface.get_player_scores(
            database_client,
            ctx,
            player
        )


def now() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')


async def wrap_slash_errors(
        ctx: CommandContext,
        client: Client,
        guild_id: Snowflake,
        fct: Callable[[], Coroutine]) -> None:
    try:
        is_test_server = str(guild_id) == "918195469245628446"
        is_dev_env = os.getenv("POLYTOPIA_ENVIRONMENT", "") == "DEVELOPMENT"
        if (is_dev_env and is_test_server) or (not is_test_server and not is_dev_env):
            await asyncio.create_task(fct())
    # except interactions.errors.Forbidden:
    #    await ctx.send("Missing permission. <@338067113639936003> has been notified.")
    except:
        error = sys.exc_info()[0]
        logger.error("##### ERROR #####")
        logger.error(error)
        logger.error(traceback.format_exc())
        print("##### ERROR #####")
        print(error)
        traceback.print_exc()
        error_channel = await get(client, Channel, object_id=1035274340125659230)
        # error_channel = ctx.get_channel(1035274340125659230)  # Polytopia Helper Testing server, Error channel
        channel = await ctx.get_channel()
        guild = await ctx.get_guild()
        await error_channel.send(f"""Error in channel {channel.name}, {guild.name}:\n{traceback.format_exc()}\n""")
        await ctx.send('There was an error. <@338067113639936003> has been notified.')


async def add_success_reaction(message: Message) -> None:
    await message.create_reaction("✅")


async def add_received_reaction(message: Message) -> None:
    await message.create_reaction("📩")


async def add_error_reaction(message: Message) -> None:
    await message.create_reaction("🚫")


async def add_delete_reaction(message: Message) -> None:
    await message.create_reaction("🗑")


async def clear_channel_map_reactions(
        channel: Channel,
        fct: Callable[[], Coroutine[Any, Any, Any]]) -> None:
    messages_ids = database_client.get_channel_resource_messages(
        channel.id,
        [ImageOp.MAP_INPUT, ImageOp.MAP_PROCESSED_IMAGE])
    
    for m_id in messages_ids:
        # message = await channel.get_message(m_id['source_message_id'])
        # await clear_map_reaction(message)
        database_client.set_resource_operation(m_id['source_message_id'], ImageOp.INPUT, 0)
    await fct()


async def clear_map_reaction(message: Message) -> None:
    await message.remove_all_reactions_of("🖼️")


async def trace(ctx: CommandContext) -> None:
    answer_message = await ctx.send("Processing")
    messages = database_client.get_channel_resource_messages(
        ctx.channel_id,
        [ImageOp.MAP_INPUT, ImageOp.MAP_PROCESSED_IMAGE]
    )
    print("messages", messages)
    channel = await ctx.get_channel()
    for i, m in enumerate(messages):
        try:
            message = await channel.get_message(m['source_message_id'])
            sent_message = await message.reply("%s %d" % (ImageOp(m['operation']).name, i))
        except BaseException as e:
            print(e, flush=True)
            sent_message = await ctx.send("Message not found: %d" % m['source_message_id'])
        await add_delete_reaction(sent_message)
    if len(messages) == 0:
        await ctx.send("Trace empty")
    await answer_message.edit("Done")

"""
async def renew_patching(dry_run=False):
    database_client = DatabaseClient(
        user="discordBot", password="password123", port="5432", database="polytopiaHelper_dev",
        host="database")
    incomplete_channel_list = database_client.get_incomplete_patching_run()
    
    for incomplete_run in incomplete_channel_list:
        print(incomplete_run)
        # channel_details = database_client.get_channel_info(channel_id)
"""
