import os
import interactions

from interactions import ApplicationCommandType, autodefer
from interactions import CommandContext

from slash_bot_client import bot_utils
from slash_bot_client import bot_error_utils
from common.logger_utils import logger

"""
Slash bot command registry
- manages bot events
- manages user commands
- sends requests to bot utils
"""

VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG")
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
    
    def __init__(self, client: interactions.Client) -> None:
        self.client: interactions.Client = client
    
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
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.activate(ctx, size))
    
    @interactions.extension_command(
        name="deactivate",
        description="Deactivates the channel. Reactions and image uploads will not be tracked anymore."
    )
    async def slash_deactivate(self, ctx: CommandContext) -> None:
        logger.info("DEACTIVATE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.deactivate(ctx))
    
    @interactions.extension_command(
        name="version",
        description="Current bot version."
    )
    async def version(self, ctx: CommandContext) -> None:
        logger.info("VERSION - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        await ctx.send("slash bot version %s" % VERSION)
    
    @interactions.extension_command(
        name="scores",
        description="Shows saved scores plot. If player is specified, shows the player score history.",
        options=[
            interactions.Option(
                name="player",
                description="Retrieves score for the player",
                type=interactions.OptionType.STRING,
                required=False,
            ),
        ]
    )
    async def slash_get_channel_player_scores(self, ctx: CommandContext, player: str = None) -> None:
        logger.info("PLAYER SCORE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client,
                                                lambda: bot_utils.get_channel_player_score(ctx, player))
    
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
            if await bot_utils.has_access(self.client, ctx):
                await bot_utils.patch_map(
                    ctx,
                    self.client._http,
                    number_of_images
                )
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
    
    @interactions.extension_command(
        name="set-score",
        description="Set score for a specific player and turn",
        options=[
            interactions.Option(
                name="player_name",
                description="player-name",
                type=interactions.OptionType.STRING,
                required=True
            ),
            interactions.Option(
                name="turn",
                description="turn",
                type=interactions.OptionType.INTEGER,
                required=True
            ),
            interactions.Option(
                name="score",
                description="score",
                type=interactions.OptionType.INTEGER,
                required=True
            )
        ]
    )
    async def slash_set_player_score(self, ctx: CommandContext, player_name: str, turn: int, score: int) -> None:
        logger.info("SET SCORE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.set_player_score(
            ctx,
            player_name,
            turn,
            score
        ))
    
    @interactions.extension_command(
        name="clear_maps",
        description="Clear the stack of maps to patch"
    )
    async def slash_clear_map_reactions(self, ctx: CommandContext) -> None:
        logger.info("CLEAR MAPS - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        async def inner() -> None:
            message = await ctx.send("Processing")
            channel = await ctx.get_channel()
            await bot_utils.clear_channel_map_reactions(channel, lambda: message.edit("Done"))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, inner)
    
    @interactions.extension_command(
        name="channels",
        description="admin command — lists all active channels in the server"
    )
    @interactions.autodefer(30)
    async def slash_list_active_channels(self, ctx: CommandContext) -> None:
        logger.info("CHANNELS - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.list_active_channels(ctx))
    
    @interactions.extension_command(
        name="drop",
        description="admin command — deletes the channel from our memory"
    )
    async def drop_channel(self, ctx: CommandContext) -> None:
        logger.info("DROP - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.drop_channel(ctx))
    
    @interactions.extension_command(
        name="Add map",
        type=ApplicationCommandType.MESSAGE
    )
    @autodefer()
    async def add_patch_source(self, ctx: CommandContext):
        logger.info("ADD MAP - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        async def inner():
            if await bot_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
                    
                    await bot_utils.add_map_and_patch(
                        ctx,
                        self.client._http
                    )
                    
                    await message.edit("Analysing")
                
                else:
                    await ctx.send("Please add a message with an image")
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
    
    @interactions.extension_command(
        name="Remove image",
        type=ApplicationCommandType.MESSAGE
    )
    async def remove_patch_source(self, ctx: CommandContext):
        logger.info("REMOVE IMAGE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        async def inner():
            if len(ctx.target.attachments) > 0:
                message = await ctx.send("Processing")
                
                await bot_utils.remove_map(ctx)
                await message.edit("Done")
            
            else:
                await ctx.send("Please remove a message with an image")
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
    
    @interactions.extension_command(
        name="Renew action",
        type=ApplicationCommandType.MESSAGE
    )
    async def renew_map_patching(self, ctx: CommandContext):
        logger.info("RENEW ACTION - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        async def inner():
            if await bot_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
                    
                    await bot_utils.force_analyse_map_and_patch(
                        ctx,
                        self.client._http
                    )
                    
                    await message.edit("Analysing")
                
                else:
                    await ctx.send("Please add a message with an image")
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
    
    @interactions.extension_command(
        name="trace",
        description="Lists all map pieces used as input for patching"
    )
    async def get_map_trace(self, ctx: CommandContext) -> None:
        logger.info("TRACE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.trace(ctx))
    
    @interactions.extension_command(
        name="whitelist",
        description="Tells you if you are on the user white list"
    )
    async def is_white_list(self, ctx: CommandContext) -> None:
        logger.info("IS WHITE LIST - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.white_list(ctx))
    
    @interactions.extension_command(
        name="roles",
        description="admin command — lists all active channels in the server"
    )
    async def get_roles(self, ctx: CommandContext) -> None:
        logger.info("ROLES - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        has_access = await bot_utils.has_access(self.client, ctx)
        if has_access:
            await ctx.send("Welcome to poly helper, please submit your action")
    
    @interactions.extension_command(
        name="White list user",
        type=ApplicationCommandType.USER
    )
    async def white_list_user(self, ctx: CommandContext):
        logger.info("PUT WHITE LIST USER - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.white_list_user(ctx))
    
    @interactions.extension_command(
        name="De white list user",
        type=ApplicationCommandType.USER
    )
    async def de_white_list_user(self, ctx: CommandContext):
        logger.info("POP WHITE LIST USER - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_utils.de_white_list_user(ctx))
    
    @interactions.extension_command(
        name="renew-incomplete-patching-runs",
        description="Tells you if you are on the user white list",
        options=[
            interactions.Option(
                name="dry-run",
                description="prints the list of missing patchings",
                converter="dry_run",
                type=interactions.OptionType.BOOLEAN,
                required=False
            )
        ]
    )
    async def renew_patching_runs(self, ctx: CommandContext, dry_run: bool = True):
        logger.info("RENEW PATCHINGS - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client,
                                                lambda: bot_utils.renew_patching(self.client, ctx, dry_run))
