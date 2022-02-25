import re

import sqlalchemy
import pandas as pd

from common.image_utils import ImageOp


class DatabaseClient:

    def __init__(self, user, password, port, database, host):
        # db_url = 'postgresql://discordBot:password123@polytopia_database_1:5432/polytopiaHelper_dev'
        # db_url = 'postgresql://discordBot:password123@172.25.0.2:5432/polytopiaHelper_dev'

        self.database = database
        self.database_url = f"""postgresql://{user}:{password}@{host}:{port}/{database}"""
        self.engine = sqlalchemy.create_engine(self.database_url, echo=True)

    def is_channel_active(self, channel_id: int):
        is_active = self.engine.execute(
            f"""SELECT is_active FROM discord_channel
                WHERE channel_discord_id = {channel_id};""").fetchone()
        return is_active is not None and len(is_active) > 0 and is_active[0]

    def activate_channel(self, channel, server):
        channel_name = re.sub(r"[^a-zA-Z0-9 ]", "", channel.name)[:40]
        return self.engine.execute(
            f"""INSERT INTO discord_server
                (server_discord_id, server_name)
                VALUES ({server.id}, '{server.name}')
                ON CONFLICT (server_discord_id) DO UPDATE
                SET server_name = '{server.name}';

                INSERT INTO discord_channel
                (server_discord_id, channel_discord_id, channel_name, is_active)
                VALUES ({server.id}, {channel.id}, '{channel_name}', true)
                ON CONFLICT (channel_discord_id) DO UPDATE
                SET is_active = true;""")

    def deactivate_channel(self, channel_id):
        return self.engine.execute(
            f"""UPDATE discord_channel
                SET is_active = false
                WHERE channel_discord_id = {channel_id};""")

    # TODO: add create_if_missing=True param
    # TODO: result should be modified: result = [dict(row) for row in resultproxy]
    def get_game_players(self, channel_id):
        result_proxy = self.engine.execute(
            f"""SELECT polytopia_player.game_player_uuid,
                       polytopia_player.polytopia_player_name,
                       polytopia_player.discord_player_name
                FROM game_players
                JOIN polytopia_player
                ON game_players.game_player_uuid = polytopia_player.game_player_uuid
                WHERE channel_discord_id = {channel_id};""").fetchall()
        return [dict(row) for row in result_proxy]

    def add_score(self, channel_id, player_id, score, turn):
        player_id = ("'" + str(player_id) + "'") if player_id is not None else 'NULL'
        return self.engine.execute(
            f"""INSERT INTO game_player_scores
                (channel_discord_id, game_player_uuid, turn, score, confirmed)
                VALUES ({channel_id}, {player_id}, {turn}, {score}, false);""")

    def get_channel_scores(self, channel_id) -> pd.DataFrame:
        scores = self.engine.execute(
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

    def get_channel_scores_gb(self, channel_id):
        return self.engine.execute(
            f"""SELECT game_player_uuid, array_agg(turn), array_agg(score)
                FROM game_player_scores
                WHERE channel_discord_id = {channel_id}
                GROUP BY game_player_uuid;""").fetchall()

    def list_active_channels(self, server_id):
        return self.engine.execute(
            f"""SELECT channel_name FROM discord_channel
                WHERE server_discord_id = {server_id}
                AND is_active = true;""").fetchall()

    def remove_resource(self, message_id):
        filenames = self.engine.execute(
            f"""DELETE FROM message_resources
                WHERE source_message_id = {message_id}
                RETURNING filename::text;""")
        return [f[0] for f in filenames]

    def add_resource(self, message, author, operation, resource_number=0):
        self.add_player_n_game(message, author)
        filename = self.engine.execute(
            f"""INSERT INTO message_resources
                (source_channel_id, source_message_id, resource_number, author_id, operation)
                VALUES ({message.channel.id}, {message.id}, {resource_number}, {author.id}, {operation.value})
                RETURNING filename::text;
                """).fetchone()
        if len(filename) > 0:
            return filename[0]

    def get_resource(self, message_id, resource_number=0):
        resource = self.engine.execute(
            f"""SELECT * FROM message_resources
                WHERE source_message_id = {message_id}
                AND resource_number = {resource_number};""").fetchall()
        if len(resource) > 0:
            return resource[0]

    def set_resource_operation(self, message_id, operation, resource_number):
        filename = self.engine.execute(
            f"""UPDATE message_resources
                SET operation = {operation.value}
                WHERE source_message_id = {message_id}
                AND resource_number = {resource_number}
                RETURNING filename::text;""").fetchone()
        if filename is not None and len(filename) > 0:
            return filename[0]

    def get_resource_filename(self, message, operation, resource_number):
        filename = self.engine.execute(
            f"""SELECT filename::text
                FROM message_resources
                WHERE source_channel_id = {message.channel.id}
                AND source_message_id = {message.id}
                AND resource_number = {resource_number}
                AND operation = {operation.value};""").fetchone()
        if filename is not None and len(filename) > 0:
            return filename[0]

    def get_map_patching_files(self, channel_id):
        filenames = self.engine.execute(
            f"""SELECT DISTINCT ON (source_message_id) source_message_id,
                filename::text, author_id
                FROM message_resources
                WHERE source_channel_id = {channel_id}
                AND operation = {ImageOp.MAP_INPUT.value};""").fetchall()
        return [dict(row)["filename"] for row in filenames]

    def get_resource_number(self, filename):
        return self.engine.execute(
            f"""SELECT resource_number FROM message_resources
                WHERE filename::text = '{filename}';""").fetchone()[0]

    def set_player_discord_name(self, discord_player_id, discord_player_name, polytopia_player_name):
        return self.engine.execute(
            f"""INSERT INTO polytopia_player
                (discord_player_id, discord_player_name, polytopia_player_name)
                VALUES ({discord_player_id}, '{str(discord_player_name)}', '{str(polytopia_player_name)}')
                ON CONFLICT (discord_player_id) DO UPDATE
                SET discord_player_name = '{str(discord_player_name)}',
                polytopia_player_name = '{str(polytopia_player_name)}';""")

    def set_player_game_name(self, game_player_uuid, polytopia_player_name):
        return self.engine.execute(
            f"""UPDATE polytopia_player
                SET polytopia_player_name = '{polytopia_player_name}'
                WHERE game_player_uuid = '{game_player_uuid}';""")

    def get_last_turn(self, channel_id):
        latest_turn = self.engine.execute(
            f"""SELECT latest_turn FROM polytopia_game
                WHERE channel_discord_id = {channel_id};""").fetchone()
        if latest_turn is not None and len(latest_turn) > 0:
            return latest_turn[0]

    def set_new_last_turn(self, channel_id, turn):
        return self.engine.execute(
            f"""UPDATE polytopia_game
                SET latest_turn = {turn}
                WHERE channel_discord_id = {channel_id};""")

    def get_game_map_size(self, channel_id):
        map_size = self.engine.execute(
            f"""SELECT map_size FROM polytopia_game
                WHERE channel_discord_id = {channel_id};""").fetchone()
        if len(map_size) > 0:
            return map_size[0]

    def set_game_map_size(self, channel_id, size):
        return self.engine.execute(
            f"""UPDATE polytopia_game
                SET map_size = {size}
                WHERE channel_discord_id = {channel_id};""")

    def add_player_n_game(self, message, author):
        game_player_uuid = self.engine.execute(
            f"""INSERT INTO polytopia_player
                (discord_player_id, discord_player_name)
                VALUES ({author.id}, '{str(author.name)}')
                ON CONFLICT (discord_player_id) DO UPDATE
                SET discord_player_name = '{author.name}'
                RETURNING game_player_uuid::text;""").fetchone()
        if game_player_uuid is not None and len(game_player_uuid) > 0:
            return self.engine.execute(
                f"""INSERT INTO polytopia_game
                    (server_discord_id, channel_discord_id)
                    VALUES ({message.guild.id}, {message.channel.id})
                    ON CONFLICT (channel_discord_id) DO NOTHING;

                    INSERT INTO game_players
                    (game_player_uuid, channel_discord_id, is_alive)
                    VALUES ('{game_player_uuid[0]}', {message.channel.id}, true)
                    ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING;""")

    def add_missing_player(self, player_name, channel_id):
        game_player_uuid = self.engine.execute(
            f"""INSERT INTO polytopia_player
                (polytopia_player_name)
                VALUES ('{str(player_name)}')
                RETURNING game_player_uuid::text;""").fetchone()
        if game_player_uuid is not None and len(game_player_uuid) > 0:
            self.engine.execute(
                f"""INSERT INTO game_players
                    (game_player_uuid, channel_discord_id, is_alive)
                    VALUES ('{game_player_uuid[0]}', {channel_id}, true)
                    ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING;""")
            return game_player_uuid[0]

    def drop_score(self, channel_id, turn):
        return self.engine.execute(
            f"""DELETE FROM game_player_scores
                WHERE channel_discord_id = {channel_id}
                AND turn = {turn};""")

    def get_channel_resource_messages(self, channel_id, operation):
        resources = self.engine.execute(
            f"""SELECT source_message_id
                FROM message_resources
                WHERE source_channel_id = {channel_id}
                AND operation = {operation.value};""").fetchall()
        return [dict(row) for row in resources]

    def add_player_to_game(self, game_player_uuid, channel_id):
        return self.engine.execute(
            f"""INSERT INTO game_players
                (game_player_uuid, channel_discord_id, is_alive)
                VALUES ('{game_player_uuid}', {channel_id}, true)
                ON CONFLICT (game_player_uuid, channel_discord_id) DO NOTHING""")

    def set_player_score(self, game_player_uuid, turn, score):
        return self.engine.execute(
            f"""UPDATE game_player_scores
                SET game_player_uuid = '{game_player_uuid}'
                WHERE score_uuid = (
                    SELECT score_uuid
                    FROM game_player_scores
                    WHERE turn = {turn}
                    AND score = {score}
                    LIMIT 1);""")
