import sqlalchemy


class DatabaseClient:

    def __init__(self, user, password, port, database, host):
        # db_url = 'postgresql://discordBot:password123@polytopia_database_1:5432/polytopiaHelper_dev'
        # db_url = 'postgresql://discordBot:password123@172.25.0.2:5432/polytopiaHelper_dev'

        self.database = database
        database_url = f"""postgresql://{user}:{password}@{host}:{port}/{database}"""
        self.engine = sqlalchemy.create_engine(database_url, echo=True)

    def is_channel_active(self, channel_id: int):
        is_active = self.engine.execute(
            f"""SELECT is_active FROM discord_channel
                WHERE channel_discord_id = {channel_id};""").fetchone()
        return is_active is not None and len(is_active) > 0 and is_active[0]

    def activate_channel(self, channel, server):
        return self.engine.execute(
            f"""INSERT INTO discord_server
                (server_discord_id, server_name)
                VALUES ({server.id}, '{server.name}')
                ON CONFLICT (server_discord_id) DO UPDATE
                SET server_name = '{server.name}';

                INSERT INTO discord_channel
                (server_discord_id, channel_discord_id, channel_name, is_active)
                VALUES ({server.id}, {channel.id}, '{channel.name}', true)
                ON CONFLICT (channel_discord_id) DO UPDATE
                SET is_active = true;""")

    def deactivate_channel(self, channel):
        return self.engine.execute(
            f"""UPDATE discord_channel
                SET is_active = false
                WHERE channel_discord_id = {channel.id};""")

    # TODO: add create_if_missing=True param
    # TODO: result should be modified: result = [dict(row) for row in resultproxy]
    def get_game_players(self, channel):
        result_proxy = self.engine.execute(
            f"""SELECT game_players.discord_player_id,
                       polytopia_player.polytopia_player_name,
                       polytopia_player.discord_player_name
                FROM game_players
                JOIN polytopia_player
                ON game_players.discord_player_id = polytopia_player.discord_player_id
                WHERE channel_discord_id = {channel.id};""").fetchall()
        return [dict(row) for row in result_proxy]

    def add_score(self, channel, player, score):
        return

    def get_channel_scores(self, channel_id):
        return

    def list_active_channels(self, server):
        return self.engine.execute(
            f"""SELECT channel_name FROM discord_channel
                WHERE server_discord_id = {server.id}
                AND is_active = true;""").fetchall()

    def add_resource(self, message, author, operation, resource_number=0):

        filename = self.engine.execute(
            f"""INSERT INTO polytopia_player
                (discord_player_id, discord_player_name)
                VALUES ({author.id}, '{str(author.name)}')
                ON CONFLICT (discord_player_id) DO UPDATE
                SET discord_player_name = '{author.name}';

                INSERT INTO polytopia_game
                (server_discord_id, channel_discord_id)
                VALUES ({message.guild.id}, {message.channel.id})
                ON CONFLICT (channel_discord_id) DO NOTHING;

                INSERT INTO game_players
                (discord_player_id, channel_discord_id, is_alive)
                VALUES ({author.id}, {message.channel.id}, true)
                ON CONFLICT (discord_player_id, channel_discord_id) DO NOTHING;

                INSERT INTO message_resources
                (source_channel_id, source_message_id, resource_number, author_id, operation)
                VALUES ({message.channel.id}, {message.id}, {resource_number}, {author.id}, {operation.value})
                RETURNING filename::text;
                """).fetchone()
        if len(filename) > 0:
            return filename[0]

    def get_resource(self, message, resource_number=0):
        resource = self.engine.execute(
            f"""SELECT * FROM message_resources
                WHERE source_channel_id = {message.channel.id}
                AND source_message_id = {message.id}
                AND resource_number = {resource_number};""").fetchall()
        if len(resource) > 0:
            return resource[0]

    def set_resource_operation(self, message, operation, resource_number):
        filename = self.engine.execute(
            f"""UPDATE message_resources
                SET operation = {operation.value}
                WHERE source_channel_id = {message.channel.id}
                AND source_message_id = {message.id}
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
        if len(filename) > 0:
            return filename[0]

    def get_map_patching_files(self, channel):
        filenames = self.engine.execute(
            f"""SELECT DISTINCT ON (source_message_id) source_message_id,
                filename::text, author_id
                FROM message_resources
                WHERE source_channel_id = {channel.id};""").fetchall()
        print("pre filenames", filenames)
        return [dict(row)["filename"] for row in filenames]

    def get_resource_number(self, filename):
        return self.engine.execute(
            f"""SELECT resource_number FROM message_resources
                WHERE filename::text = '{filename}';""").fetchone()[0]
