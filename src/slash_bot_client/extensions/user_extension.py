import os
import interactions

from interactions import ApplicationCommandType
from interactions import CommandContext

from slash_bot_client.utils import bot_user_utils
from slash_bot_client.utils import bot_error_utils
from common.logger_utils import logger

from slash_bot_client.utils.bot_utils import BotUtils

VERSION = os.getenv("SLASH_BOT_VERSION")
DEBUG = os.getenv("POLYTOPIA_DEBUG")
TOKEN = os.getenv("DISCORD_TEST_TOKEN" if DEBUG else "DISCORD_TOKEN")


class UserExtension(interactions.Extension):
    
    def __init__(self, client: interactions.Client, bot_utils: BotUtils) -> None:
        self.client: interactions.Client = client
        self.bot_utils = bot_utils
    
    @interactions.extension_command(
        name="White list user",
        type=ApplicationCommandType.USER
    )
    async def white_list_user(self, ctx: CommandContext):
        logger.info("PUT WHITE LIST USER - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_user_utils.white_list_user(ctx))
    
    @interactions.extension_command(
        name="De white list user",
        type=ApplicationCommandType.USER
    )
    async def de_white_list_user(self, ctx: CommandContext):
        logger.info("POP WHITE LIST USER - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
        
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_user_utils.de_white_list_user(ctx))
    
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
        
        await bot_error_utils.wrap_slash_errors(
            ctx,
            self.client,
            lambda: self.bot_utils.renew_patching(self.client, ctx, dry_run)
        )

    @interactions.extension_command(
        name="whitelist",
        description="Tells you if you are on the user white list"
    )
    async def is_white_list(self, ctx: CommandContext) -> None:
        logger.info("IS WHITE LIST - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        await bot_error_utils.wrap_slash_errors(ctx, self.client, lambda: bot_user_utils.white_list(ctx))

    @interactions.extension_command(
        name="roles",
        description="admin command — lists all active channels in the server"
    )
    async def get_roles(self, ctx: CommandContext) -> None:
        logger.info("ROLES - %d - %d" % (int(ctx.id), int(ctx.channel_id)))
    
        has_access = await bot_user_utils.has_access(self.client, ctx)
        if has_access:
            await ctx.send("Welcome to poly helper, please submit your action")
