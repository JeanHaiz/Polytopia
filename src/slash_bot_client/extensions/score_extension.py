import os
import interactions

from interactions import CommandContext
from interactions import ApplicationCommandType

from slash_bot_client.utils import bot_utils, bot_error_utils
from common.logger_utils import logger

VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG")
TOKEN = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")


class ScoreExtension(interactions.Extension):
    
    def __init__(self, client: interactions.Client) -> None:
        self.client: interactions.Client = client

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
    
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            lambda: bot_utils.get_channel_player_score(ctx, player)
        )

    @interactions.extension_command(
        name="Add score",
        type=ApplicationCommandType.MESSAGE
    )
    async def add_score_image(self, ctx: CommandContext):
        logger.info("ADD SCORE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))

        async def inner():
            if await bot_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
            
                    await bot_utils.add_score_and_plot(
                        ctx,
                        self.client._http
                    )
            
                    await message.edit("Analysing")
        
                else:
                    await ctx.send("Please add a message with an image")

        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
