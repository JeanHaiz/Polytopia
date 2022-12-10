import os
import interactions

from interactions import ApplicationCommandType, autodefer
from interactions import CommandContext

from slash_bot_client import bot_utils
from slash_bot_client import bot_error_utils

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
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
            lambda: bot_utils.activate(ctx, size)
        )
    
    @interactions.extension_command(
        name="deactivate",
        description="Deactivates the channel. Reactions and image uploads will not be tracked anymore."
    )
    async def slash_deactivate(self, ctx: CommandContext) -> None:
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
            lambda: bot_utils.deactivate(ctx)
        )
    
    @interactions.extension_command(
        name="version",
        description="Current bot version."
    )
    async def version(self, ctx: CommandContext) -> None:
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
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
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
        await ctx.send("Loading")
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
            lambda: bot_utils.patch_map(
                ctx,
                self.client._http,
                number_of_images
            )
        )
    
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
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
            lambda: bot_utils.set_player_score(
                ctx,
                player_name,
                turn,
                score
            )
        )
    
    @interactions.extension_command(
        name="clear_maps",
        description="Clear the stack of maps to patch"
    )
    async def slash_clear_map_reactions(self, ctx: CommandContext) -> None:
        async def inner() -> None:
            message = await ctx.send("Processing")
            channel = await ctx.get_channel()
            await bot_utils.clear_channel_map_reactions(channel, lambda: message.edit("Done"))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, ctx.guild_id, inner)
    
    @interactions.extension_command(
        name="channels",
        description="admin command — lists all active channels in the server"
    )
    async def slash_list_active_channels(self, ctx: CommandContext) -> None:
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild_id,
            lambda: bot_utils.list_active_channels(ctx)
        )
    
    @interactions.extension_command(
        name="Add map",
        type=ApplicationCommandType.MESSAGE
    )
    @autodefer()
    async def add_patch_source(self, ctx: CommandContext):
        async def inner():
            if len(ctx.target.attachments) > 0:
                message = await ctx.send("Loading")
                
                await bot_utils.add_map_and_patch(
                    ctx,
                    self.client._http
                )
                
                await message.edit("Analysing")
            
            else:
                await ctx.send("Please add a message with an image")
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild.id,
            lambda: inner()
        )
    
    @interactions.extension_command(
        name="Remove",
        type=ApplicationCommandType.MESSAGE
    )
    async def remove_patch_source(self, ctx: CommandContext):
        async def inner():
            if len(ctx.target.attachments) > 0:
                message = await ctx.send("Processing")
                
                await bot_utils.remove_map(ctx)
                await message.edit("Done")
            
            else:
                await ctx.send("Please remove a message with an image")
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild.id,
            lambda: inner()
        )
    
    @interactions.extension_command(
        name="Renew",
        type=ApplicationCommandType.MESSAGE
    )
    async def renew_map_patching(self, ctx: CommandContext):
        async def inner():
            if len(ctx.target.attachments) > 0:
                message = await ctx.send("Loading")
                
                await bot_utils.force_analyse_map_and_patch(
                    ctx,
                    self.client._http
                )
                
                await message.edit("Analysing")
            
            else:
                await ctx.send("Please add a message with an image")
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild.id,
            lambda: inner()
        )
    
    @interactions.extension_command(
        name="trace",
        description="Lists all map pieces used as input for patching"
    )
    async def get_map_trace(self, ctx: CommandContext) -> None:
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            ctx.guild.id,
            lambda: bot_utils.trace(ctx)
        )
