import os
import datetime
import interactions

from interactions import Client
from interactions import CommandContext

from common import image_utils
from common.image_operation import ImageOp

from slash_bot_client.utils.bot_utils import DEBUG
from slash_bot_client.utils.bot_utils import database_client

admin_user_id = os.getenv("DISCORD_ADMIN_USER")


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
    except interactions.api.error.LibraryException:
        print("Unknown member")
        embed = interactions.Embed()
        embed.description = (
                """Your profile was not found on the poly helper guild.\n""" +
                """To use the bot, please join the [poly helper discord server](https://discord.gg/4TzqfZzM3f)""")
        await ctx.send(embeds=embed)
        return False
    
    general_guild_member_roles = [(await general_guild.get_role(r)).name for r in member.roles]
    if os.getenv("DISCORD_PATREON_FERVENT") in general_guild_member_roles:  # fervent
        limit = 80
    elif os.getenv("DISCORD_PATREON_HUSTLER") in general_guild_member_roles:  # hustler
        limit = 1000
    else:
        limit = 30
        if False:  # count < limit:
            embed = interactions.Embed()
            embed.description = (
                    f"""You have {limit - count - 1} actions remaining this month.\n""" +
                    """For more actions, """ +
                    """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper).""")
            await ctx.send(embeds=embed)
            return True
    
    if count >= limit:
        time_embed = interactions.Embed()
        time_embed.description = (
                """You have passed your action limit for this month. \n""" +
                f"""Please send a message to <@{admin_user_id}> for more, he'll grant them :wink:""",
                # """For now, we're granting you unlimited actions.\n To support, """ +
                # """please visit the [Poly Helper Patreon](https://www.patreon.com/polytopiahelper)."""
        )
        await ctx.send(embeds=time_embed)
        return False

    else:
        # count is below limit for known user, access granted
        return True


def is_fervent(role: str) -> bool:
    return role == "FERVENT"


def is_hustler(role: str) -> bool:
    return role == "HUSTLER"


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
