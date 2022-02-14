drop table game_player_scores;
drop table game_players;

CREATE TABLE IF NOT EXISTS game_players(
	game_player_uuid uuid NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	is_alive BOOLEAN,
	PRIMARY KEY (game_player_uuid, channel_discord_id),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES discord_channel(channel_discord_id),
	CONSTRAINT fk_player FOREIGN KEY (game_player_uuid) REFERENCES polytopia_player(game_player_uuid)
);
GRANT ALL PRIVILEGES ON TABLE game_players TO discordBot;

CREATE TABLE IF NOT EXISTS game_player_scores(
	score_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	game_player_uuid uuid,
	turn INT,
	score INT,
	confirmed BOOLEAN,
	PRIMARY KEY (score_uuid),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES polytopia_game(channel_discord_id),
	CONSTRAINT fk_player FOREIGN KEY (game_player_uuid) REFERENCES polytopia_player(game_player_uuid)
);
GRANT ALL PRIVILEGES ON TABLE game_player_scores TO discordBot;