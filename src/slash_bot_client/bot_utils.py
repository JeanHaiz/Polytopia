import os
import datetime
import interactions

from io import BytesIO

from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
from typing import Coroutine

from interactions import Channel
from interactions import HTTPClient
from interactions import Optional
from interactions import CommandContext
from interactions import Message
from interactions import Client
from interactions import get

from database.database_client import get_database_client
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

database_client = get_database_client()


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
    size = database_client.get_game_map_size(ctx.channel_id)
    if size is None:
        await ctx.send("Please set the channel map size with the 'activate' slash command")
    else:
        # Fulfilling precondition: Game setup
        channel_info = database_client.get_channel_info(ctx.channel_id)
        channel = await ctx.get_channel()
        
        if channel_info is None:
            server = await ctx.get_guild()
            
            database_client.activate_channel(
                ctx.channel_id,
                channel.name,
                ctx.guild_id,
                server.name
            )
        
        database_client.add_player_n_game(
            ctx.channel_id,
            ctx.guild_id,
            ctx.author.id,
            ctx.author.name
        )
        
        # Helper variables
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
        await patch_images(ctx, bot_http_client, channel, 3)


async def patch_images(
        ctx: CommandContext,
        bot_http_client: HTTPClient,
        channel: Channel,
        n_images: Optional[int]
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
    
    count = 0
    
    for message_resource_i in message_resources:
        
        message_id = message_resource_i["source_message_id"]
        resource_number = message_resource_i["resource_number"]
        
        requirements_i = await register_process_inputs(
            lambda i: __download_attachment(channel, message_id, resource_number, bot_http_client),
            patch_process_id,
            channel,
            message_id,
            resource_number
        )
        
        requirements.extend(requirements_i)
        if n_images is not None and len(requirements_i) != 0:
            count += 1
            if count >= n_images:
                break
    
    if DEBUG:
        print("Patching requirements", requirements, flush=True)
    
    if len(requirements) == 0:
        database_client.update_patching_process_status(patch_process_id, "NO-IMAGE")
        await ctx.send("No map added. Please add maps to create a collage.")
    elif len(requirements) == 2:
        database_client.update_patching_process_status(patch_process_id, "ONE-IMAGE")
        await ctx.send("Found one map only. Please add a second map to create a collage.")
    else:
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
        resource_number: int
) -> List[Tuple[int, int, str, str]]:
    resource = database_client.get_resource(message_id, resource_number)
    filename = str(resource["filename"])
    
    check = await image_utils.get_or_fetch_image_check(
        database_client,
        download_fct,
        channel.name,
        message_id,
        filename,
        ImageOp(resource["operation"])
    )
    
    if filename is None or not check:
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
    size = database_client.get_game_map_size(ctx.channel_id)
    if size is None:
        await ctx.send("Please set the channel map size with the 'activate' slash command")
    else:
        message_resources = database_client.get_channel_message_resource_messages(
            ctx.channel_id,
            ctx.target.id,
            [ImageOp.MAP_INPUT, ImageOp.MAP_PROCESSED_IMAGE]
        )
        
        channel = await ctx.get_channel()
        if DEBUG:
            print("message resources", len(message_resources), flush=True)
        
        for message_resource_i in message_resources:
            
            if DEBUG:
                print("message_resource_i", message_resource_i, flush=True)
            
            resource_number = int(message_resource_i["resource_number"])
            operation = message_resource_i["operation"]
            
            filename = database_client.set_resource_operation(
                ctx.target.id,
                ImageOp.INPUT,
                resource_number
            )
            
            if filename is not None:
                image_utils.move_back_input_image(
                    channel.name,
                    filename,
                    ImageOp(operation)
                )
            else:
                if DEBUG:
                    print("Filename is none", flush=True)


async def force_analyse_map_and_patch(
        ctx: CommandContext,
        bot_http_client: HTTPClient
):
    size = database_client.get_game_map_size(ctx.channel_id)
    if size is None:
        await ctx.send("Please set the channel map size with the 'activate' slash command")
    else:
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
            
            resource_i = database_client.get_resource(message.id, resource_number)
            
            if resource_i is not None:
                database_client.delete_image_param(str(resource_i["filename"]))
                database_client.set_resource_operation(message.id, ImageOp.MAP_INPUT, resource_number)
            
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
        await patch_images(ctx, bot_http_client, channel, 3)


async def list_active_channels(ctx: CommandContext) -> None:
    if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
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


async def drop_channel(ctx: CommandContext) -> None:
    if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
        logger.debug("drop channel")
        active = database_client.is_channel_active(ctx.channel_id)
        if active:
            message = database_client.drop_channel(ctx.channel_id)
            await ctx.send(message)
        else:
            await ctx.send("channel not active")
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
        await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))


async def patch_map(
        ctx: CommandContext,
        bot_http_client: HTTPClient,
        number_of_images: Optional[int]
) -> None:
    size = database_client.get_game_map_size(ctx.channel_id)
    if size is None:
        await ctx.send("Please set the channel map size with the 'activate' slash command")
    else:
        channel = await ctx.get_channel()
        await patch_images(
            ctx,
            bot_http_client,
            channel,
            number_of_images
        )


async def activate(ctx: CommandContext, size: int) -> None:
    logger.debug("activate channel %s" % ctx.channel_id)
    channel = await ctx.get_channel()
    guild = await ctx.get_guild()
    activation_result = database_client.activate_channel(ctx.channel_id, channel.name, ctx.guild_id, guild.name)
    database_client.add_player_n_game(ctx.channel_id, ctx.guild_id, ctx.author.id, ctx.author.name)
    database_client.set_game_map_size(ctx.channel_id, int(size))
    if activation_result.rowcount == 1:
        await ctx.send("channel activated")
    else:
        await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))


async def deactivate(ctx: CommandContext) -> None:
    logger.debug("deactivate channel %s" % ctx.channel_id)
    deactivation_result = database_client.deactivate_channel(ctx.channel_id)
    if deactivation_result.rowcount == 1:
        await ctx.send("channel deactivated")
    else:
        await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))


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
    channel = await ctx.get_channel()
    for i, m in enumerate(messages):
        try:
            message = await channel.get_message(m['source_message_id'])
            sent_message = await message.reply("%s %d" % (ImageOp(m['operation']).name, i))
        except BaseException as e:
            print("ERROR", e, flush=True)
            sent_message = await ctx.send("Message not found: %d" % m['source_message_id'])
        await add_delete_reaction(sent_message)
    if len(messages) == 0:
        await ctx.send("Trace empty")
    await answer_message.edit("Done")


def is_fervent(role: str) -> bool:
    return role == "FERVENT"


def is_hustler(role: str) -> bool:
    return role == "HUSTLER"


async def has_access(client: Client, ctx: CommandContext):
    # general_channel = await get(client, Channel, object_id=int(os.getenv("DISCORD_ERROR_CHANNEL")))
    # permissions = await general_channel.get_permissions_for(ctx.author)
    target_guild_id = os.getenv("DISCORD_TEST_SERVER") if DEBUG else os.getenv("DISCORD_GENERAL_SERVER")
    general_guild = [g for g in client.guilds if g.id == target_guild_id][0]
    
    is_white_listed = database_client.is_white_listed(ctx.author.id)
    if is_white_listed:
        # print("action whitelisted")
        return True
    
    try:
        member = await general_guild.get_member(ctx.author.id)
        count = database_client.get_request_count(member.id)
        general_guild_member_roles = [(await general_guild.get_role(r)).name for r in member.roles]
        if os.getenv("DISCORD_PATREON_FERVENT") in general_guild_member_roles:  # fervent
            limit = 15
        elif os.getenv("DISCORD_PATREON_HUSTLER") in general_guild_member_roles:  # hustler
            limit = 50
        elif member is not None:
            limit = 3
            if count < limit:
                embed = interactions.Embed()
                embed.description = (
                        f"""You have {limit - count - 1} actions remaining this month.\n""" +
                        """For more actions, """ +
                        """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper).""")
                await ctx.send(embeds=embed)
                return True
        else:
            embed = interactions.Embed()
            embed.description = (
                    """Your profile was not found on the poly helper guild.\n""" +
                    """Please join the [poly helper discord server](https://discord.gg/6kk6nJnf)""")
            await ctx.send(embeds=embed)
            return False
        # await ctx.send(str(limit) + ", " + str(count) + ", " + str(general_guild_member_roles))
        
        if count >= limit:
            if (datetime.datetime.now().year == 2023 and datetime.datetime.now().month <= 1) or \
                    datetime.datetime.now().year == 2022:  # TODO add trial period
                time_embed = interactions.Embed()
                time_embed.description = (
                        """You have passed your action limit for this month. \n""" +
                        """Here is a free trial for the poly helper bot.\n""" +
                        """For now, we're granting you unlimited actions.\n To support, """ +
                        """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper)."""
                )
                await ctx.send(embeds=time_embed)
                return True
            else:
                embed = interactions.Embed()
                embed.description = (
                        """You have passed your action limit for this month.\n""" +
                        """Please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper).""")
                await ctx.send(embeds=embed)
                return False
        else:
            # count is below limit for known user, access granted
            return True
    
    except interactions.api.error as e:
        await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
        print("ROLE ERROR\n" + e)
        logger.warning("ROLE ERROR\n" + e)
        
        return False


async def white_list(ctx: CommandContext):
    is_white_listed = database_client.is_white_listed(ctx.author.id)
    if is_white_listed is not None and is_white_listed:
        embed = interactions.Embed()
        embed.description = (
                f"""Hi {ctx.author.name}, you are white-listed. You have unlimited access.\n""" +
                """To support the bot and its creators,""" +
                """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper).""")
        await ctx.send(embeds=embed)
    else:
        embed = interactions.Embed()
        embed.description = (
                f"""You are NOT white-listed.\n""" +
                """To support the bot and its creators and access more functionalities, """ +
                """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper).""")
        await ctx.send(embeds=embed)


async def white_list_user(ctx: CommandContext):
    if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
        target_user = ctx.target
        database_client.white_list_user(target_user.id)
        await ctx.send("User whitelisted")
    else:
        await ctx.send("The command is reserved for admins.")


async def de_white_list_user(ctx: CommandContext):
    if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
        target_user = ctx.target
        database_client.de_white_list_user(target_user.id)
        await ctx.send("User removed from white-list")
    else:
        await ctx.send("The command is reserved for admins.")


async def renew_patching(client: Client, ctx: CommandContext, dry_run):
    def print_channel(channel_id):
        channel_details = database_client.get_channel_info(channel_id)
        server_name = database_client.get_server_name(channel_details["server_discord_id"])
        return server_name + " " + channel_details["channel_name"]
        
    if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
        incomplete_channel_list = database_client.get_incomplete_patching_run()
        
        if dry_run:
            if len(incomplete_channel_list) > 0:
                message = "\n".join(
                    [
                        print_channel(icl["channel_discord_id"]) + " on %s at %s" % (
                            icl["max_started_on_started"].strftime('%Y.%m.%d'),
                            icl["max_started_on_started"].strftime('%H:%M:%S')
                        )
                        for icl in incomplete_channel_list
                    ]
                )
            else:
                message = "All patching runs are complete."
            await ctx.send(message)
        else:
            for incomplete_run in incomplete_channel_list:
                print(incomplete_run["patch_uuid"], incomplete_run)
                channel = await get(client, Channel, object_id=incomplete_run["channel_discord_id"])
                await patch_images(ctx, client._http, channel, 3)
    else:
        await ctx.send("The command is reserved for admins.")
