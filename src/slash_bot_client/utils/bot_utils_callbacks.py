import os
import re

import numpy as np
import pandas as pd

from typing import List
from typing import Tuple
from typing import Optional

from interactions import Embed
from interactions import File
from interactions import Channel
from interactions import Client
from interactions.utils.get import get

from common import image_utils
from slash_bot_client.utils import bot_error_utils
from common.image_operation import ImageOp
from database.database_client import get_database_client
from slash_bot_client.interfaces.map_patching_interface import MapPatchingInterface
from slash_bot_client.interfaces.score_visualisation_interface import ScoreVisualisationInterface
from common.error_utils import MapPatchingErrors

DEBUG = os.getenv("POLYTOPIA_DEBUG", 0)
database_client = get_database_client()


class BotUtilsCallbacks:
    
    def __init__(
            self,
            map_patching_interface: MapPatchingInterface,
            score_visualisation_interface: ScoreVisualisationInterface
    ):
        self.map_patching_interface = map_patching_interface
        self.score_visualisation_interface = score_visualisation_interface
    
    async def on_analysis_error(
            self,
            process_uuid: str,
            map_requirement_id: str,
            client: Client,
            error: str
    ):
        await self.__on_error(process_uuid, map_requirement_id, client, error)
    
    async def on_patching_error(
            self,
            process_uuid: str,
            client: Client,
            error: str
    ):
        await self.__on_error(process_uuid, None, client, error)
    
    async def on_turn_recognition_error(
            self,
            process_uuid: str,
            turn_requirement_id: str,
            client: Client,
            error: str
    ):
        await self.__on_error(process_uuid, turn_requirement_id, client, error)
    
    async def on_visualisation_error(
            self,
            process_uuid: str,
            client: Client,
            error: str
    ):
        await self.__on_error(process_uuid, None, client, error)

    @staticmethod
    async def __on_error(
            process_uuid: str,
            requirement_id: Optional[str],
            client: Client,
            error: str
    ):
        if DEBUG:
            print("Error message received", process_uuid, requirement_id, client, error, flush=True)
        
        database_client.update_process_status(process_uuid, "ERROR - %s" % error)
        
        if requirement_id is not None:
            database_client.update_patching_process_requirement(process_uuid, requirement_id, "ERROR - %s" % error)
        
        patch_info = database_client.get_process(process_uuid)
        error_channel = await get(client, Channel, object_id=int(os.getenv("DISCORD_ERROR_CHANNEL")))
        
        if DEBUG:
            print("Error management", patch_info is not None, patch_info, error_channel, flush=True)
        
        if patch_info is not None:
            channel_id = patch_info["channel_discord_id"]
            channel = await get(client, Channel, object_id=channel_id)
            await channel.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
            
            channel_info = database_client.get_channel_info(channel_id)
            server_name = database_client.get_server_name(channel_info["server_discord_id"])
            await error_channel.send(
                f"""Hey <@{os.getenv("DISCORD_ADMIN_USER")}>,\n""" +
                f"""Error in channel {channel.name} on server {server_name}:\n{error}\n""")
        else:
            print(error, flush=True)
            await error_channel.send(
                f"""Hey <@{os.getenv("DISCORD_ADMIN_USER")}>,\n""" +
                f"""Error in unknown channel for\npatch {process_uuid}, \nrequirement {requirement_id}""")
    
    def on_map_analysis_complete(
            self,
            process_uuid: str,
            map_requirement_id: str
    ):
        database_client.complete_patching_process_requirement(
            map_requirement_id
        )
        
        if DEBUG:
            print("complete analysis", process_uuid, map_requirement_id, flush=True)
        
        if self.__check_process_requirements_complete(process_uuid):
            if DEBUG:
                print("sending patching request")
            
            self.map_patching_interface.send_map_patching_request(
                process_uuid,
                number_of_images=None
            )
    
    async def on_map_patching_complete(
            self,
            client: Client,
            process_uuid: str,
            channel_id: int,
            filename: str
    ) -> None:
        if DEBUG:
            print("Done patching, callback completed", process_uuid, flush=True)
        
        turn = database_client.get_last_turn(
            channel_id
        )
        channel = await get(client, Channel, object_id=channel_id)
        # await channel.send("Done patching")
        channel_info = database_client.get_channel_info(channel_id)
        
        patch_path = image_utils.get_file_path(
            channel_info["channel_name"],
            ImageOp.MAP_PATCHING_OUTPUT,
            filename
        )
        
        patching_errors = self.get_patching_errors(process_uuid)
        
        if DEBUG:
            print("patching errors", patching_errors, flush=True)
            print("channel", channel, flush=True)
        
        with open(patch_path, "rb") as fh:
            attachment = File(fp=fh, filename=filename + ".png")
            
            if attachment is not None:
                await channel.send(files=attachment, content="Map patched for turn %s" % turn)
                database_client.update_process_status(process_uuid, "DONE")
            else:
                patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
            fh.close()
        
        await bot_error_utils.manage_slash_patching_errors(database_client, channel, channel, patching_errors)

    def __print_scores(self, scores: pd.DataFrame):
        scores = self.__augment_scores(scores)
        return self.__pretty_print_scores(scores)
        # return augment_scores(scores).to_string(index=False)

    @staticmethod
    def __pretty_print_scores(scores_raw):
        # print(scores)
        scores = scores_raw.drop(columns="score_uuid")
        score_list = scores.to_numpy().tolist()
        player_width = np.max([len(d) if type(d) == str else 0 for data in score_list for d in data])
        print("player_width", player_width)
    
        def width(i):
            return player_width if i == 0 else 5
    
        def align(i, item):
            if i == 0:
                if item is None:
                    item = "  ?"
                return item.ljust(width(i), ' ')
            else:
                return str(item).rjust(width(i), ' ')
    
        scores.sort_values(by="turn")
    
        header = ['Player', 'Turn', 'Score', 'Delta']
        s = ['   '.join([str(item).ljust(width(i), ' ') for i, item in enumerate(header)])]
    
        for data in score_list:
            s.append('   '.join([align(i, item) for i, item in enumerate(data)]))
        d = '```' + '\n'.join(s) + '```'
        return d

    def __print_player_scores(self, scores: pd.DataFrame, player):
        scores = self.__augment_scores(scores)
        player_scores = scores[scores['polytopia_player_name'] == player]
        return self.__pretty_print_scores(player_scores)

    @staticmethod
    def __augment_scores(scores: pd.DataFrame):
        for player in scores['polytopia_player_name'].drop_duplicates():
            player_scores = scores[scores['polytopia_player_name'] == player]
            player_scores.sort_values(by="turn", ascending=False)
            scores.loc[scores['polytopia_player_name'] == player, "delta"] = player_scores["score"].diff()
        # scores.delta = pd.to_numeric(scores.delta, errors='coerce')
        return scores
    
    async def on_player_score_request(self, channel, player):
        scores = database_client.get_channel_scores(channel.id)
        if scores is not None and len(scores[scores['turn'] != -1]) > 0:
            if player is not None and player not in scores["polytopia_player_name"].unique():
                players = scores["polytopia_player_name"].unique()
                await channel.send("Player %s not recognised. Available players: %s" % (str(player), str(players)))
            else:
                score_text = self.__print_player_scores(scores, player)
                embed = Embed(title='%s scores' % str(player), description=score_text)
                await channel.send(embeds=embed)
        else:
            await channel.send("No score found for player %s" % str(player))

    async def on_score_visualisation_complete(
            self,
            client: Client,
            process_uuid: str,
            channel_id: int,
            filename: str
    ) -> None:
        if DEBUG:
            print("Done visualisation, callback completed", process_uuid, flush=True)

        scores: pd.DataFrame = database_client.get_channel_scores(channel_id)
        
        channel = await get(client, Channel, object_id=channel_id)
        channel_info = database_client.get_channel_info(channel_id)

        patch_path = image_utils.get_file_path(
            channel_info["channel_name"],
            ImageOp.SCORE_PLT,
            filename
        )

        # patching_errors = self.get_patching_errors(process_uuid)

        if DEBUG:
            # print("patching errors", patching_errors, flush=True)
            print("channel", channel, flush=True)
            
        with open(patch_path, "rb") as fh:
            attachment = File(fp=fh, filename=filename + ".png")
    
            if attachment is not None:
                await channel.send(files=attachment, content="Score visualisation")
                database_client.update_process_status(process_uuid, "DONE")
            else:
                # patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                print("attachment is none")
            fh.close()

        score_text = self.__print_scores(scores)
        
        print("score text", score_text)
        # await ctx.send(score_text)
        embed = Embed(title='Game scores', description=score_text)
        await channel.send(embeds=embed)
        
    @staticmethod
    def get_patching_errors(
            process_uuid: str
    ) -> List[Tuple[MapPatchingErrors, Optional[str]]]:
        patching_status = database_client.get_process_status(process_uuid)
        
        if DEBUG:
            print("patching status", patching_status, flush=True)
        
        if patching_status.startswith("ERRORS - "):
            return [
                (re.search(r"([A-Z_]+)\(", status).group(1), re.search(r"([a-z0-9-]{36}|None)", status).group(0))
                for status in patching_status[len("ERRORS - "):].split(";")]
    
    def on_turn_recognition_complete(
            self,
            process_uuid: str,
            turn_requirement_id: str
    ):
        database_client.complete_patching_process_requirement(
            turn_requirement_id
        )
        
        if self.__check_process_requirements_complete(process_uuid):
            self.map_patching_interface.send_map_patching_request(process_uuid, number_of_images=None)

    def on_score_recognition_complete(
            self,
            process_uuid: str,
            score_requirement_id: str,
            channel_id: int
    ):
        database_client.complete_patching_process_requirement(
            score_requirement_id
        )
    
        if self.__check_process_requirements_complete(process_uuid):
            process = database_client.get_process(process_uuid)
            channel = database_client.get_channel_info(channel_id)
        
            self.score_visualisation_interface.get_or_visualise_scores(
                process_uuid,
                process["process_author_discord_id"],
                channel_id,
                channel["channel_name"]
            )

    @staticmethod
    def __check_process_requirements_complete(process_uuid: str):
        requirements = database_client.get_patching_process_requirement(process_uuid)
        all_requirement_check = all([r["complete"] for r in requirements])
        
        if DEBUG:
            print("requirements", all_requirement_check, requirements, flush=True)
        
        return all_requirement_check
