import os
import interactions

from interactions import CommandContext
from interactions import ApplicationCommandType

import slash_bot_client.utils.bot_user_utils
from slash_bot_client.utils import bot_error_utils
from common.logger_utils import logger

from slash_bot_client.utils.bot_utils import BotUtils

VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG", 0)
TOKEN = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")


class ScoreExtension(interactions.Extension):
    
    def __init__(self, client: interactions.Client, bot_utils: BotUtils) -> None:
        self.client: interactions.Client = client
        self.bot_utils = bot_utils

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
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: self.bot_utils.set_player_score(
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
            lambda: self.bot_utils.get_channel_player_score(ctx, player)
        )

    @interactions.extension_command(
        name="turn",
        description="Set game turn for the score or map analysis",
        options=[
            interactions.Option(
                name="turn",
                description="Turn to be saved",
                type=interactions.OptionType.INTEGER,
                required=True,
            ),
        ]
    )
    async def set_channel_turn(self, ctx: CommandContext, turn: int) -> None:
        logger.info("TURN - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            lambda: self.bot_utils.set_new_last_turn(ctx, turn)
        )

    @interactions.extension_command(
        name="drop-score",
        description="Drop score for a specific turn",
        options=[
            interactions.Option(
                name="turn",
                description="Turn for which score are drop",
                type=interactions.OptionType.INTEGER,
                required=True,
            ),
        ]
    )
    async def set_channel_turn(self, ctx: CommandContext, turn: int) -> None:
        logger.info("DROP SCORES - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            lambda: self.bot_utils.drop_scores(ctx, turn)
        )

    @interactions.extension_command(
        name="Add score",
        type=ApplicationCommandType.MESSAGE
    )
    async def add_score_image(self, ctx: CommandContext):
        logger.info("ADD SCORE - %d - %d" % (int(ctx.id), int(ctx.channel_id)))

        async def inner():
            if await slash_bot_client.utils.bot_user_utils.has_access(self.client, ctx):
                if len(ctx.target.attachments) > 0:
                    message = await ctx.send("Loading")
            
                    await self.bot_utils.add_score_and_plot(
                        ctx,
                        self.client._http
                    )
            
                    await message.edit("Analysing")
        
                else:
                    await ctx.send("Please add a message with an image")

        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: inner())
