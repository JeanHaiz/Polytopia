import re
import uuid

import sqlalchemy
import pandas as pd

from typing import List
from typing import Tuple
from typing import Optional
from sqlalchemy.engine.result import ResultProxy
from common.image_operation import ImageOp
from map_patching.image_param import ImageParam


class DatabaseClient:

    def __init__(self, user: str, password: str, port: str, database: str, host: str) -> None:
        self.database = database
        self.database_url = f"""postgresql://{user}:{password}@{host}:{port}/{database}"""
        self.engine = sqlalchemy.create_engine(self.database_url, echo=True)

    def execute(self, query: str) -> ResultProxy:
        return self.engine.execute(query)

    def is_channel_active(self, channel_id: int) -> bool:
        is_active = self.execute(
            f"""SELECT is_active FROM discord_channel
                WHERE channel_discord_id = {channel_id};""").fetchone()
        return is_active is not None and len(is_active) > 0 and is_active[0]

    def activate_channel(self, channel_id: int, channel_name: str, server_id: int, server_name: str) -> bool:
        channel_name = re.sub(r"[^a-zA-Z0-9]", "", channel_name)[:40]
        server_name = re.sub(r"[^a-zA-Z0-9]", "", server_name)[:40]
        result = self.execute(
            f"""INSERT INTO discord_server
                (server_discord_id, server_name)
                VALUES ({server_id}, '{server_name}')
                ON CONFLICT (server_discord_id) DO UPDATE
                SET server_name = '{server_name}';

                INSERT INTO discord_channel
                (server_discord_id, channel_discord_id, channel_name, is_active)
                VALUES ({server_id}, {channel_id}, '{channel_name}', true)
                ON CONFLICT (channel_discord_id) DO UPDATE
                SET is_active = true;""")
        return result.rowcount > 0

    def deactivate_channel(self, channel_id: int) -> bool:
        result = self.execute(
            f"""UPDATE discord_channel
                SET is_active = false
                WHERE channel_discord_id = {channel_id};""")
        return result.rowcount > 0

    # TODO: add create_if_missing=True param
    def get_game_players(self, channel_id: int) -> list:
        result_proxy = self.execute(
            f"""SELECT polytopia_player.game_player_uuid,
                       polytopia_player.polytopia_player_name,
                       polytopia_player.discord_player_name
                FROM game_players
                JOIN polytopia_player
                ON game_players.game_player_uuid = polytopia_player.game_player_uuid
                WHERE channel_discord_id = {channel_id};""").fetchall()
        return [dict(row) for row in result_proxy]

    def add_score(self, channel_id: int, player_id: int, score: int, turn: int) -> ResultProxy:
        player_str_id = ("'" + str(player_id) + "'") if player_id is not None else 'NULL'
        return self.execute(
            f"""INSERT INTO game_player_scores
                (channel_discord_id, game_player_uuid, turn, score, confirmed)
                VALUES ({channel_id}, {player_str_id}, {turn}, {score}, false);""")

    def get_channel_scores(self, channel_id: int) -> pd.DataFrame:
        scores = self.execute(
            f"""SELECT polytopia_player_name, turn, score
                FROM game_player_scores
                LEFT JOIN polytopia_player
                ON game_player_scores.game_player_uuid = polytopia_player.game_player_uuid
                WHERE channel_discord_id = {channel_id}
                ORDER BY turn ASC;""")
        score_entries = scores.fetchall()
        if score_entries is not None and len(score_entries) > 0:
            scores_df = pd.DataFrame(score_entries)
            scores_df.columns = scores.keys()
            return scores_df

    def get_channel_scores_gb(self, channel_id: int) -> list:
        return self.execute(
            f"""SELECT game_player_uuid, array_agg(turn), array_agg(score)
                FROM game_player_scores
                WHERE channel_discord_id = {channel_id}
                GROUP BY game_player_uuid;""").fetchall()

    def list_active_channels(self, server_id: int) -> list:
        return self.execute(
            f"""SELECT channel_name FROM discord_channel
                WHERE server_discord_id = {server_id}
                AND is_active = true;""").fetchall()

    def remove_resource(self, message_id: int) -> list:
        filenames = self.execute(
            f"""DELETE FROM message_resources
                WHERE source_message_id = {message_id}
                RETURNING filename::text;""")
        return [f[0] for f in filenames]

    def add_resource(
        self, guild_id: int, channel_id: int, message_id: int, author_id: int, author_name: str,
            operation: ImageOp, resource_number: int = 0) -> Optional[str]:
        self.add_player_n_game(channel_id, guild_id, author_id, author_name)
        filename = self.execute(
            f"""INSERT INTO message_resources
                (source_channel_id, source_message_id, resource_number, author_id, operation)
                VALUES ({channel_id}, {message_id}, {resource_number}, {author_id}, {operation.value})
                RETURNING filename::text;
                """).fetchone()
        return filename[0] if filename is not None and len(filename) > 0 else None

    def get_resource(self, message_id: int, resource_number: int = 0) -> Optional[dict]:
        resource = self.execute(
            f"""SELECT * FROM message_resources
                WHERE source_message_id = {message_id}
                AND resource_number = {resource_number};""").fetchall()
        if len(resource) > 0:
            return dict(resource[0])
        else:
            return None

    def set_resource_operation(self, message_id: int, operation: ImageOp, resource_number: int) -> Optional[str]:
        filename = self.execute(
            f"""UPDATE message_resources
                SET operation = {operation.value}
                WHERE source_message_id = {message_id}
                AND resource_number = {resource_number}
                RETURNING filename::text;""").fetchone()
        return filename[0] if filename is not None and len(filename) > 0 else None

    def get_resource_filename(
        self, channel_id: int, message_id: int, operation: ImageOp, resource_number: int
    ) -> Optional[str]:
        filename = self.execute(
            f"""SELECT filename::text
                FROM message_resources
                WHERE source_channel_id = {channel_id}
                AND source_message_id = {message_id}
                AND resource_number = {resource_number}
                AND operation = {operation.value};""").fetchone()
        return filename[0] if filename is not None and len(filename) > 0 else None

    def get_map_patching_files(self, channel_id: int) -> list:
        filenames = self.execute(
            f"""SELECT DISTINCT ON (source_message_id) source_message_id,
                filename::text, author_id
                FROM message_resources
                WHERE source_channel_id = {channel_id}
                AND operation = {ImageOp.MAP_INPUT.value};""").fetchall()
        return [dict(row)["filename"] for row in filenames]

    def get_resource_number(self, filename: str) -> str:
        result = self.execute(
            f"""SELECT resource_number FROM message_resources
                WHERE filename::text = '{filename}';""").fetchone()
        return result[0] if result is not None and len(result) > 0 else None

    def set_player_discord_name(
        self, discord_player_id: int, discord_player_name: str, polytopia_player_name: str
    ) -> ResultProxy:
        return self.execute(
            f"""INSERT INTO polytopia_player
                (discord_player_id, discord_player_name, polytopia_player_name)
                VALUES ({discord_player_id}, '{str(discord_player_name)}', '{str(polytopia_player_name)}')
                ON CONFLICT (discord_player_id) DO UPDATE
                SET discord_player_name = '{str(discord_player_name)}',
                polytopia_player_name = '{str(polytopia_player_name)}';""")

    def set_player_game_name(self, game_player_uuid: str, polytopia_player_name: str) -> ResultProxy:
        return self.execute(
            f"""UPDATE polytopia_player
                SET polytopia_player_name = '{polytopia_player_name}'
                WHERE game_player_uuid = '{game_player_uuid}';""")

    def get_last_turn(self, channel_id: int) -> Optional[str]:
        latest_turn = self.execute(
            f"""SELECT latest_turn FROM polytopia_game
                WHERE channel_discord_id = {channel_id};""").fetchone()
        return latest_turn[0] if latest_turn is not None and len(latest_turn) > 0 else None

    def set_new_last_turn(self, channel_id: int, turn: int) -> ResultProxy:
        return self.execute(
            f"""UPDATE polytopia_game
                SET latest_turn = {turn}
                WHERE channel_discord_id = {channel_id};""")

    def get_game_map_size(self, channel_id: int) -> Optional[str]:
        map_size = self.execute(
            f"""SELECT map_size FROM polytopia_game
                WHERE channel_discord_id = {channel_id};""").fetchone()
        return map_size[0] if map_size is not None and len(map_size) > 0 else None

    def set_game_map_size(self, channel_id: int, size: int) -> ResultProxy:
        return self.execute(
            f"""UPDATE polytopia_game
                SET map_size = {size}
                WHERE channel_discord_id = {channel_id};""")

    def add_player_n_game(self, channel_id: int, guild_id: int, author_id: int, author_name: str) -> Optional[str]:
        game_player_uuid = self.execute(
            f"""INSERT INTO polytopia_player
                (discord_player_id, discord_player_name)
                VALUES ({author_id}, '{str(author_name)}')
                ON CONFLICT (discord_player_id) DO UPDATE
                SET discord_player_name = '{author_name}'
                RETURNING game_player_uuid::text;""").fetchone()
        if game_player_uuid is not None and len(game_player_uuid) > 0:
            self.execute(
                f"""INSERT INTO polytopia_game
                    (server_discord_id, channel_discord_id)
                    VALUES ({guild_id}, {channel_id})
                    ON CONFLICT (channel_discord_id) DO NOTHING;

                    INSERT INTO game_players
                    (game_player_uuid, channel_discord_id, is_alive)
                    VALUES ('{game_player_uuid[0]}', {channel_id}, true)
                    ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING;""")
            return game_player_uuid[0]
        else:
            return None

    def add_missing_player(self, player_name: str, channel_id: int) -> Optional[str]:
        game_player_uuid = self.execute(
            f"""INSERT INTO polytopia_player
                (polytopia_player_name)
                VALUES ('{str(player_name)}')
                RETURNING game_player_uuid::text;""").fetchone()
        if game_player_uuid is not None and len(game_player_uuid) > 0:
            self.execute(
                f"""INSERT INTO game_players
                    (game_player_uuid, channel_discord_id, is_alive)
                    VALUES ('{game_player_uuid[0]}', {channel_id}, true)
                    ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING;""")
            return game_player_uuid[0]
        else:
            return None

    def drop_score(self, channel_id: int, turn: str) -> ResultProxy:
        return self.execute(
            f"""DELETE FROM game_player_scores
                WHERE channel_discord_id = {channel_id}
                AND turn = {turn};""")

    def get_channel_resource_messages(self, channel_id: int, operation: ImageOp) -> list:
        resources = self.execute(
            f"""SELECT source_message_id, operation
                FROM message_resources
                WHERE source_channel_id = {channel_id}
                AND operation = {operation.value};""").fetchall()
        return [dict(row) for row in resources]

    def add_player_to_game(self, game_player_uuid: str, channel_id: int) -> ResultProxy:
        return self.execute(
            f"""INSERT INTO game_players
                (game_player_uuid, channel_discord_id, is_alive)
                VALUES ('{game_player_uuid}', {channel_id}, true)
                ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING""")

    def set_player_score(self, game_player_uuid: str, turn: int, score: int) -> ResultProxy:
        return self.execute(
            f"""UPDATE game_player_scores
                SET game_player_uuid = '{game_player_uuid}'
                WHERE score_uuid = (
                    SELECT score_uuid
                    FROM game_player_scores
                    WHERE turn = {turn}
                    AND score = {score}
                    LIMIT 1);""")

    def get_resource_message(self, filename: str) -> Optional[Tuple]:
        resources = self.execute(
            f"""SELECT source_message_id, source_channel_id
                FROM message_resources
                WHERE filename::text = '{filename}';""").fetchone()
        if resources is not None and len(resources) > 0:
            return resources[1], resources[0]
        else:
            return None, None

    def add_patching_process(self, channel_id: str, author_id: str) -> ResultProxy:
        patch_uuid = self.execute(
            f"""INSERT INTO map_patching_process
                (channel_discord_id, process_author_discord_id, status)
                VALUES ('{channel_id}', '{author_id}', 'STARTED')
                RETURNING patch_uuid::text;""").fetchone()
        if patch_uuid is not None and len(patch_uuid) > 0:
            return patch_uuid[0]
        else:
            return None

    def add_patching_process_input(self, patch_uuid: str, input_filename: str, order: int) -> ResultProxy:
        return self.execute(
            f"""INSERT INTO map_patching_process_input
                (patch_uuid, input_filename, input_order)
                VALUES ('{uuid.UUID(patch_uuid)}', '{input_filename}', {order});""")

    def get_patching_processes(self, channel_id: str) -> list:
        processes = self.execute(
            f"""SELECT * FROM map_patching_process
                WHERE channel_discord_id = '{channel_id}'
                ORDER BY started_on DSC;""").fetchall()
        return [dict(row) for row in processes]

    def get_patching_process(self, patch_uuid: str) -> Optional[dict]:
        process = self.execute(
            f"""SELECT * FROM map_patching_process
                WHERE patch_uuid::text = '{patch_uuid}';""").fetchall()
        if process is not None and len(process) > 0:
            return dict(process[0])
        else:
            return None

    def get_patching_inputs(self, patch_uuid: str) -> list:
        process_inputs = self.execute(
            f"""SELECT * FROM map_patching_process_input
                WHERE patch_uuid::text = '{patch_uuid}';""").fetchall()
        return [dict(row) for row in process_inputs]

    def update_patching_process_status(self, patch_uuid: str, status: str) -> bool:
        patch_uuid = self.execute(
            f"""UPDATE map_patching_process
                SET status = '{status}'
                WHERE patch_uuid::text = '{patch_uuid}'
                RETURNING patch_uuid::text;""").fetchone()
        return patch_uuid is not None and len(patch_uuid) > 0

    def update_patching_process_output_filename(self, patch_uuid: str, filename: str) -> bool:
        patch_uuid = self.execute(
            f"""UPDATE map_patching_process
                SET output_filename = '{filename}', ended_on = now()
                WHERE patch_uuid::text = '{patch_uuid}'
                RETURNING patch_uuid::text;""").fetchone()
        return patch_uuid is not None and len(patch_uuid) > 0

    def update_patching_process_input_status(self, patch_uuid: str, filename: str, status: str) -> bool:
        patch_input_uuid = self.execute(
            f"""UPDATE map_patching_process_input
                SET status = '{status}'
                WHERE patch_uuid::text = '{patch_uuid}'
                AND input_filename = '{filename}'
                RETURNING patch_input_uuid::text;""").fetchone()
        return patch_input_uuid is not None and len(patch_input_uuid) > 0

    def set_image_params(self, image_params: ImageParam) -> bool:
        filename_output = self.execute(
            f"""INSERT INTO map_patching_input_param
                (filename, cloud_scale, corners)
                VALUES ('{image_params.filename}',
                {image_params.cloud_scale},
                ARRAY{self.__format_tuple_list(image_params.corners)})
                ON CONFLICT (filename) DO UPDATE
                SET
                    cloud_scale = {image_params.cloud_scale},
                    corners = ARRAY{self.__format_tuple_list(image_params.corners)}::integer[][]
                RETURNING filename::text;""").fetchone()
        return filename_output is not None and len(filename_output) > 0

    def set_background_image_params(self, map_size: str, image_params: ImageParam) -> bool:
        filename_output = self.execute(
            f"""INSERT INTO map_patching_background_param
                (map_size, cloud_scale, corners)
                VALUES ('{map_size}', {image_params.cloud_scale}, ARRAY{self.__format_tuple_list(image_params.corners)})
                ON CONFLICT (map_size) DO UPDATE
                SET
                    cloud_scale = {image_params.cloud_scale},
                    corners = ARRAY{self.__format_tuple_list(image_params.corners)}::integer[][]
                RETURNING map_size;""").fetchone()
        return filename_output is not None and len(filename_output) > 0

    def get_image_params(self, filename: str) -> Optional[ImageParam]:
        result_proxy = self.execute(
            f"""SELECT *
                FROM map_patching_input_param
                WHERE filename::text = {filename};""").fetchall()
        rows = [dict(row) for row in result_proxy]
        if len(rows) == 1:
            return ImageParam(str(filename), rows[0]["cloud_scale"], rows[0]["corners"])
        else:
            return None

    def get_bulk_image_params(self, filenames: List[str]) -> List[ImageParam]:
        filename_list = str(tuple(filenames))
        result_proxy = self.execute(
            f"""SELECT *
                FROM map_patching_input_param
                WHERE filename::text IN {filename_list};""").fetchall()
        return [
            ImageParam(
                str(dict(row)["filename"]),
                dict(row)["cloud_scale"],
                dict(row)["corners"]) for row in result_proxy]

    def get_background_image_params(self, map_size: int) -> Optional[ImageParam]:
        result_proxy = self.execute(
            f"""SELECT *
                FROM map_patching_background_param
                WHERE map_size = {map_size};""").fetchall()
        rows = [dict(row) for row in result_proxy]
        if len(rows) == 1:
            return ImageParam("processed_background_template_%s" % map_size, rows[0]["cloud_scale"], rows[0]["corners"])
        else:
            return None

    def __format_tuple_list(self, s: List) -> str:
        return str(list([list(s_i) for s_i in s]))
