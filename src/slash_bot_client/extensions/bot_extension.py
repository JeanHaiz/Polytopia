import os
import interactions

from interactions import ApplicationCommandType
from interactions import CommandContext

from slash_bot_client.utils import bot_error_utils
from slash_bot_client.utils import bot_user_utils
from common.logger_utils import logger

from slash_bot_client.utils.bot_utils import BotUtils

"""
Slash bot command registry
- manages bot events
- manages user commands
- sends requests to bot utils
"""

VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG", 0)
TOKEN = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")

"""
raw_socket_create
on_start
on_interaction
on_command
on_command_error
on_component
on_autocomplete
on_modal
"""


class SlashBotExtension(interactions.Extension):
    
    def __init__(self, client: interactions.Client, bot_utils: BotUtils) -> None:
        self.client = client
        self.bot_utils = bot_utils
    
    @interactions.extension_listener()
    async def listener(self, something):
        print("LISTENER", something, flush=True)
    
    @interactions.extension_command(
        name="activate",
        description="Activates the bot in the channel. Will respond to map patching and score recognition requests.",
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
                    # interactions.Choice(name="30 x 30", value=900),
                ],
            ),
        ],
    )
    async def slash_activate(self, ctx: CommandContext, size: int) -> None:
        logger.info("ACTIVATE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.activate(ctx, size))
    
    @interactions.extension_command(
        name="deactivate",
        description="Deactivates the channel. Reactions and image uploads will not be tracked anymore."
    )
    async def slash_deactivate(self, ctx: CommandContext) -> None:
        logger.info("DEACTIVATE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.deactivate(ctx))
    
    @interactions.extension_command(
        name="version",
        description="Current bot version."
    )
    async def version(self, ctx: CommandContext) -> None:
        logger.info("VERSION - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        await ctx.send("slash bot version %s" % VERSION)
    
    @interactions.extension_command(
        name="channels",
        description="admin command — lists all active channels in the server"
    )
    @interactions.autodefer(30)
    async def slash_list_active_channels(self, ctx: CommandContext) -> None:
        logger.info("CHANNELS - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.list_active_channels(ctx))
    
    @interactions.extension_command(
        name="drop",
        description="admin command — deletes the channel from our memory"
    )
    async def drop_channel(self, ctx: CommandContext) -> None:
        logger.info("DROP - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.drop_channel(ctx))
    
    @interactions.extension_command(
        name="Remove image",
        type=ApplicationCommandType.MESSAGE
    )
    async def remove_patch_source(self, ctx: CommandContext):
        logger.info("REMOVE IMAGE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        async def inner():
            if len(ctx.target.attachments) > 0:
                message = await ctx.send("Processing")
                
                await bot_user_utils.remove_map(ctx)
                await message.edit("Done")
            
            else:
                await ctx.send("Please remove a message with an image")
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())


def setup(client: interactions.Client, bot_utils: BotUtils):
    SlashBotExtension(client, bot_utils)
