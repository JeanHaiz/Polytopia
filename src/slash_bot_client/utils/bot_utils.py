import os
import json
import aiohttp
import asyncio
import datetime
import traceback

from io import BytesIO

from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
from typing import Coroutine

from interactions import AllowedMentions
from interactions import Channel
from interactions import HTTPClient
from interactions import Optional
from interactions import CommandContext
from interactions import Message
from interactions import Client
from interactions import get

from common import image_utils
from common.logger_utils import logger
from common.image_operation import ImageOp
from common.process_type import ProcessType
from database.database_client import get_database_client
from slash_bot_client import command_context_store as cxs
from slash_bot_client.utils import bot_input_utils
from slash_bot_client.utils import bot_error_utils
from slash_bot_client.utils.bot_utils_callbacks import BotUtilsCallbacks
from slash_bot_client.interfaces.map_analysis_interface import MapAnalysisInterface
from slash_bot_client.interfaces.map_patching_interface import MapPatchingInterface
from slash_bot_client.interfaces.header_footer_recognition_interface import HeaderFooterRecognitionInterface
from slash_bot_client.interfaces.score_recognition_interface import ScoreRecognitionInterface
from slash_bot_client.interfaces.score_visualisation_interface import ScoreVisualisationInterface
from slash_bot_client.queue_services.queue_service import QueueService
from slash_bot_client.queue_services.sender_service import SenderService
from slash_bot_client.queue_services.receiver_service import RabbitmqReceive

"""
Supports generic bot activities
- manages image downloads and registration
- registers ongoing processes
- calls specific pipelines or services
- unwraps discord objects
- catches process errors
"""

DEBUG = os.getenv("POLYTOPIA_DEBUG", 0)

database_client = get_database_client()


class BotUtils:
    
    def __init__(self, queue_service):
        self.queue_service = queue_service
        self.sender_service = SenderService(self.queue_service)
        
        self.map_patching_interface = MapPatchingInterface(self.sender_service)
        
        self.score_visualisation_interface = ScoreVisualisationInterface(self.sender_service)
        
        self.bot_utils_callbacks = BotUtilsCallbacks(self.map_patching_interface, self.score_visualisation_interface)
        
        self.map_analysis_interface = MapAnalysisInterface(self.sender_service, self.bot_utils_callbacks)
        
        self.score_recognition_interface = ScoreRecognitionInterface(self.sender_service)
        
        self.header_footer_recognition_interface = HeaderFooterRecognitionInterface(
            self.sender_service,
            self.bot_utils_callbacks
        )
    
    @staticmethod
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
            self,
            ctx: CommandContext,
            bot_http_client: HTTPClient
    ):
        async def initial_condition(inner_ctx: CommandContext):
            map_size = database_client.get_game_map_size(inner_ctx.channel_id)
            print("MAP SIZE", map_size, type(map_size), flush=True)
            if map_size is None or map_size == "" or map_size not in [121, 196, 256, 324, 400]:
                await inner_ctx.send("Please set the channel map size with the 'activate' slash command")
                return False
            else:
                return True
        
        await self.__add_image_and_run(
            ctx,
            bot_http_client,
            initial_condition,
            ImageOp.MAP_INPUT,
            self.patch_images
        )
    
    @staticmethod
    async def __add_image_and_run(
            ctx: CommandContext,
            bot_http_client: HTTPClient,
            initial_condition: Callable[[CommandContext], Coroutine[Any, Any, bool]],
            operation: ImageOp,
            end_action: Callable[[CommandContext, HTTPClient, Channel, Any], Coroutine[Any, Any, None]]
    ):
        if await initial_condition(ctx):
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
                    operation
                )
            
            # Patching images registered for channel
            await end_action(ctx, bot_http_client, channel, 3)
    
    async def add_score_and_plot(
            self,
            ctx: CommandContext,
            bot_http_client: HTTPClient
    ):
        async def initial_condition(_: CommandContext):
            turn = database_client.get_last_turn(ctx.channel_id)
            print("TURN", turn, type(turn), flush=True)
            if turn is None or turn == -1:
                await ctx.send("Please set the channel current with the 'turn' slash command")
                return False
            else:
                return True
        
        await self.__add_image_and_run(
            ctx,
            bot_http_client,
            initial_condition,
            ImageOp.SCORE_INPUT,
            self.gather_scores
        )

    async def gather_scores(
            self,
            ctx: CommandContext,
            bot_http_client: HTTPClient,
            channel: Channel,
            n_images: Optional[int]
    ):
        process_uuid = database_client.add_operation_process(
            ctx.channel_id,
            ctx.author.id,
            ctx.id,
            ProcessType.SCORE_VISUALISATION
        )
        
        cxs.put(process_uuid, ctx)
        
        requirements = []
        
        message_resources = database_client.get_channel_resource_messages(
            ctx.channel_id,
            [ImageOp.SCORE_INPUT]
        )
        
        count = 0
        
        for message_resource_i in message_resources:
            
            message_id = message_resource_i["source_message_id"]
            resource_number = message_resource_i["resource_number"]
            
            requirements_i = await self.register_process_inputs(
                lambda i: self.__download_attachment(channel, message_id, resource_number, bot_http_client),
                process_uuid,
                channel,
                message_id,
                resource_number,
                "SCORE"
            )
            
            requirements.extend(requirements_i)
            if n_images is not None and len(requirements_i) != 0:
                count += 1
                if count >= n_images:
                    break
        
        if DEBUG:
            print("Score recognition requirements", requirements, flush=True)
        
        if len(requirements) == 0:
            database_client.update_process_status(process_uuid, "NO-IMAGE")
            await ctx.send("No map added. Please add maps to create a collage.")
        else:
            for requirement_i in requirements:
                await self.send_score_recognition_requirement(
                    process_uuid,
                    requirement_i[0],
                    requirement_i[1],
                    requirement_i[2],
                    requirement_i[3]
                )
    
    async def patch_images(
            self,
            ctx: CommandContext,
            bot_http_client: HTTPClient,
            channel: Channel,
            n_images: Optional[int]
    ):
        database_client.update_channel(channel.id, channel.name)
        
        process_uuid = database_client.add_operation_process(
            ctx.channel_id,
            ctx.author.id,
            ctx.id,
            ProcessType.MAP_PATCHING
        )
        
        cxs.put(process_uuid, ctx)
        
        requirements = []
        
        message_resources = database_client.get_channel_resource_messages(
            ctx.channel_id,
            [ImageOp.MAP_INPUT, ImageOp.MAP_PROCESSED_IMAGE]
        )
        
        count = 0
        
        for message_resource_i in message_resources:
            
            message_id = message_resource_i["source_message_id"]
            resource_number = message_resource_i["resource_number"]
            
            requirements_i = await self.register_process_inputs(
                lambda i: self.__download_attachment(channel, message_id, resource_number, bot_http_client),
                process_uuid,
                channel,
                message_id,
                resource_number,
                "MAP"
            )
            
            requirements.extend(requirements_i)
            if n_images is not None and len(requirements_i) != 0:
                count += 1
                if count >= n_images:
                    break
        
        if DEBUG:
            print("Patching requirements", requirements, flush=True)
        
        if len(requirements) == 0:
            database_client.update_process_status(process_uuid, "NO-IMAGE")
            await ctx.send("No map added. Please add maps to create a collage.")
        elif len(requirements) == 2:
            database_client.update_process_status(process_uuid, "ONE-IMAGE")
            await ctx.send("Found one map only. Please add a second map to create a collage.")
        else:
            for requirement_i in requirements:
                await self.send_map_analysis_requirement(
                    process_uuid,
                    requirement_i[0],
                    requirement_i[1],
                    requirement_i[2],
                    requirement_i[3]
                )
    
    @staticmethod
    async def register_process_inputs(
            download_fct: Callable[[int], Coroutine[Any, Any, BytesIO]],
            process_uuid: str,
            channel: Channel,
            message_id: int,
            resource_number: int,
            requirement_type: str
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
        
        if requirement_type == "MAP":
        
            turn_requirement_id = database_client.add_patching_process_requirement(
                process_uuid,
                filename,
                "TURN"
            )
            requirements.append((message_id, resource_number, turn_requirement_id, "TURN"))
            
            map_requirement_id = database_client.add_patching_process_requirement(
                process_uuid,
                filename,
                "MAP"
            )
            requirements.append((message_id, resource_number, map_requirement_id, "MAP"))
        
        elif requirement_type == "SCORE":
    
            map_requirement_id = database_client.add_patching_process_requirement(
                process_uuid,
                filename,
                "SCORE"
            )
            requirements.append((message_id, resource_number, map_requirement_id, "SCORE"))
        
        return requirements
    
    async def send_map_analysis_requirement(
            self,
            process_uuid: str,
            message_id: int,
            resource_number: int,
            requirement_id: str,
            requirement_type: str
    ):
        if requirement_type == "TURN":
            await self.header_footer_recognition_interface.get_or_recognise_turn(
                process_uuid,
                requirement_id,
                message_id,
                resource_number
            )
        elif requirement_type == "MAP":
            await self.map_analysis_interface.get_or_analyse_map(
                process_uuid,
                requirement_id,
                message_id,
                resource_number
            )
            
    async def send_score_recognition_requirement(
            self,
            process_uuid: str,
            message_id: int,
            resource_number: int,
            requirement_id: str,
            author_name: str
    ):
        await self.score_recognition_interface.get_or_recognise_score(
            process_uuid,
            requirement_id,
            author_name,
            message_id,
            resource_number
        )
    
    async def force_analyse_map_and_patch(
            self,
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
            await self.patch_images(ctx, bot_http_client, channel, 3)
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
            self,
            ctx: CommandContext,
            bot_http_client: HTTPClient,
            number_of_images: Optional[int]
    ) -> None:
        size = database_client.get_game_map_size(ctx.channel_id)
        if size is None:
            await ctx.send("Please set the channel map size with the 'activate' slash command")
        else:
            channel = await ctx.get_channel()
            await self.patch_images(
                ctx,
                bot_http_client,
                channel,
                number_of_images
            )
    
    @staticmethod
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
    
    @staticmethod
    async def deactivate(ctx: CommandContext) -> None:
        logger.debug("deactivate channel %s" % ctx.channel_id)
        deactivation_result = database_client.deactivate_channel(ctx.channel_id)
        if deactivation_result.rowcount == 1:
            await ctx.send("channel deactivated")
        else:
            await ctx.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
    
    async def get_channel_player_score(self, ctx: CommandContext, player: str):
        message = await ctx.send("Loading")

        channel = await ctx.get_channel()
        
        if player is None:
    
            process_uuid = database_client.add_operation_process(
                ctx.channel_id,
                ctx.author.id,
                ctx.id,
                ProcessType.SCORE_VISUALISATION
            )
            
            self.score_visualisation_interface.get_or_visualise_scores(
                process_uuid,
                int(ctx.author.id),
                int(channel.id),
                channel.name
            )

        else:
            await self.bot_utils_callbacks.on_player_score_request(
                channel,
                player
            )
    
    @staticmethod
    def now() -> str:
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    
    @staticmethod
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
    
    @staticmethod
    async def clear_map_reaction(message: Message) -> None:
        await message.remove_all_reactions_of("🖼️")
    
    @staticmethod
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
                
                await message.reply("%s %d" % (ImageOp(m['operation']).name, i),
                                    allowed_mentions=AllowedMentions(replied_user=False))
            except BaseException as e:
                print("ERROR", e, flush=True)
                await ctx.send("Message not found: %d" % m['source_message_id'])
        if len(messages) == 0:
            await ctx.send("Trace empty")
        await answer_message.edit("Done")
    
    async def renew_patching(  # TODO augment with score visualisation
            self,
            client: Client,
            ctx: CommandContext,
            dry_run: bool
    ):
        def print_channel(channel_id):
            channel_details = database_client.get_channel_info(channel_id)
            server_name = database_client.get_server_name(channel_details["server_discord_id"])
            return server_name + " " + channel_details["channel_name"]
        
        if ctx.author.id == os.getenv("DISCORD_ADMIN_USER"):
            incomplete_channel_list = database_client.get_incomplete_process_runs(ProcessType.MAP_PATCHING)
            
            if dry_run:
                if len(incomplete_channel_list) > 0:
                    entries = [
                            print_channel(icl["channel_discord_id"]) + " on %s at %s: %s" % (
                                icl["max_started_on_started"].strftime('%Y.%m.%d'),
                                icl["max_started_on_started"].strftime('%H:%M:%S'),
                                icl["agg_status"][:100]
                            )
                            for icl in sorted(incomplete_channel_list, key= lambda x: x["max_started_on_started"])
                            if icl["max_started_on_started"].strftime('%Y.%m.%d').startswith("2023")
                        ]
                    message = "\n".join(entries)
                    if len(message) == 0:
                        message = "All patching runs are complete."
                        await ctx.send(message)
                    elif len(message) < 2000:
                        await ctx.send(message)
                    else:
                        chunk_size = 10
                        for i in range(0, len(entries), chunk_size):
                            message = "\n".join(entries[i:i + chunk_size])
                            await ctx.send(message)
                    
                else:
                    message = "All patching runs are complete."
                    await ctx.send(message)
            else:
                for incomplete_run in incomplete_channel_list:
                    print(incomplete_run["process_uuid"], incomplete_run)
                    channel = await get(client, Channel, object_id=incomplete_run["channel_discord_id"])
                    await self.patch_images(ctx, client._http, channel, 3)
        else:
            await ctx.send("The command is reserved for admins.")
    
    async def get_async_connection(
            self,
            queue_service: QueueService,
            queue_name: str,
            client: Client,
            loop: asyncio.AbstractEventLoop
    ):
        def action_reaction_request(channel, method, properties, body):
            action_params: dict = json.loads(body)
            action = action_params.pop("action", "")
            print("action received", body)
            
            try:
                if action == "MAP_ANALYSIS_COMPLETE":
                    self.bot_utils_callbacks.on_map_analysis_complete(**action_params)
                elif action == "MAP_PATCHING_COMPLETE":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_map_patching_complete(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                elif action == "TURN_RECOGNITION_COMPLETE":
                    self.bot_utils_callbacks.on_turn_recognition_complete(**action_params)
                elif action == "SCORE_RECOGNITION_COMPLETE":
                    self.bot_utils_callbacks.on_score_recognition_complete(**action_params)
                elif action == "SCORE_VISUALISATION_COMPLETE":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_score_visualisation_complete(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                elif action == "MAP_ANALYSIS_ERROR":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_analysis_error(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                elif action == "TURN_RECOGNITION_ERROR":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_turn_recognition_error(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                elif action == "MAP_PATCHING_ERROR":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_patching_error(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                elif action == "SCORE_VISUALISATION_ERROR":
                    asyncio.run_coroutine_threadsafe(
                        self.bot_utils_callbacks.on_visualisation_error(
                            **action_params,
                            client=client
                        ),
                        loop
                    )
                else:
                    raise ValueError("Unsupported operation: %s" % action)
            except (
                    BaseException,
                    aiohttp.client_exceptions.ClientOSError,
                    aiohttp.ClientOSError,
                    json.decoder.JSONDecodeError,
            ) as e:
                trace = (
                        "Catching base exception ERROR:" + e.__class__.__name__ +
                        " details:\n" + str(e) + "\n\n" + traceback.format_exc()
                )
                print(trace, flush=True)
                logger.warning(trace)
                bot_error_utils.error_callback(
                    database_client,
                    action_params["process_uuid"],
                    client
                )
        
        await client.wait_until_ready()
        channel = await get(client, Channel, object_id=int(os.getenv("DISCORD_ERROR_CHANNEL")))
        await channel.send("Bot started")
        
        rabbit_receive = RabbitmqReceive(queue_name, action_reaction_request)
        rabbit_receive.start()
        print("Slash bot message queue listener is running", flush=True)

    @staticmethod
    async def set_new_last_turn(ctx: CommandContext, turn: int):
        
        database_client.add_player_n_game(ctx.channel_id, ctx.guild_id, ctx.author.id, ctx.author.name)
        
        database_client.set_new_last_turn(ctx.channel_id, turn)
        
        await ctx.send("New last turn set to %d" % turn)

    @staticmethod
    async def drop_scores(ctx, turn):
        database_client.drop_score(ctx.channel_id, turn)
        ctx.send("Done")
