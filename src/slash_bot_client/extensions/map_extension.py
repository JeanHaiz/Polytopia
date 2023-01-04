import os
import interactions

from interactions import ApplicationCommandType
from interactions import CommandContext

import slash_bot_client.utils.bot_user_utils
from slash_bot_client.utils.bot_utils import BotUtils
from slash_bot_client.utils import bot_error_utils
from common.logger_utils import logger


VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG")
TOKEN = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")


class MapExtension(interactions.Extension):
    
    def __init__(self, client: interactions.Client, bot_utils: BotUtils) -> None:
        self.client: interactions.Client = client
        self.bot_utils = bot_utils

    @interactions.extension_command(
        name="trace",
        description="Lists all map pieces used as input for patching"
    )
    async def get_map_trace(self, ctx: CommandContext) -> None:
        logger.info("TRACE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.trace(ctx))

    @interactions.extension_command(
        name="clear_maps",
        description="Clear the stack of maps to patch"
    )
    async def slash_clear_map_reactions(self, ctx: CommandContext) -> None:
        logger.info("CLEAR MAPS - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        async def inner() -> None:
            message = await ctx.send("Processing")
            channel = await ctx.get_channel()
            await self.bot_utils.clear_channel_map_reactions(channel, lambda: message.edit("Done"))
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, inner)

    @interactions.extension_command(
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
    async def slash_patch_map(self, ctx: CommandContext, number_of_images: int = None) -> None:
        logger.info("PATCH - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        async def inner():
            if await slash_bot_client.utils.bot_user_utils.has_access(self.client, ctx):
                await ctx.send("Processing")
                await self.bot_utils.patch_map(
                    ctx,
                    self.client._http,
                    number_of_images
                )
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())

    @interactions.extension_command(
        name="Add map",
        type=ApplicationCommandType.MESSAGE
    )
    async def add_patch_source(self, ctx: CommandContext):
        logger.info("ADD MAP - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        async def inner():
            if await slash_bot_client.utils.bot_user_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
                
                    await self.bot_utils.add_map_and_patch(
                        ctx,
                        self.client._http
                    )
                
                    await message.edit("Analysing")
            
                else:
                    await ctx.send("Please add a message with an image")
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())

    @interactions.extension_command(
        name="Renew action",
        type=ApplicationCommandType.MESSAGE
    )
    async def renew_map_patching(self, ctx: CommandContext):
        logger.info("RENEW ACTION - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        async def inner():
            if await slash_bot_client.utils.bot_user_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
                
                    await self.bot_utils.force_analyse_map_and_patch(
                        ctx,
                        self.client._http
                    )
                
                    await message.edit("Analysing")
            
                else:
                    await ctx.send("Please add a message with an image")
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
