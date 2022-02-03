CREATE USER discordBot LOGIN 
ENCRYPTED PASSWORD 'password123' 
NOSUPERUSER NOCREATEDB NOCREATEROLE;

DROP EXTENSION IF EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE DATABASE polytopiaHelper_dev;
GRANT ALL PRIVILEGES ON DATABASE polytopiaHelper_dev TO discordBot;

/* 
CREATE DATABASE polytopiaHelper_test;
GRANT ALL PRIVILEGES ON DATABASE polytopiaHelper_dev TO discordBot;
*/

CREATE TABLE IF NOT EXISTS discord_server(
	server_discord_id BIGINT UNIQUE NOT NULL,
	server_name VARCHAR(40),
	PRIMARY KEY (server_discord_id)
);
GRANT ALL PRIVILEGES ON TABLE discord_server TO discordBot;

CREATE TABLE IF NOT EXISTS discord_channel(
	channel_discord_id BIGINT UNIQUE NOT NULL,
    server_discord_id BIGINT NOT NULL,
	channel_name VARCHAR(40),
	date_added TIMESTAMP default now() NOT NULL,
	is_active BOOLEAN default FALSE,
	PRIMARY KEY (channel_discord_id),
	CONSTRAINT fk_server FOREIGN KEY (server_discord_id) REFERENCES discord_server(server_discord_id)
);
GRANT ALL PRIVILEGES ON TABLE discord_channel TO discordBot;

CREATE TABLE IF NOT EXISTS polytopia_game(
	server_discord_id BIGINT NOT NULL,
	channel_discord_id BIGINT UNIQUE NOT NULL,
	game_name VARCHAR(40),
	n_players INT,
	map_size INT,
	latest_turn INT DEFAULT -1,
	PRIMARY KEY (channel_discord_id),
	CONSTRAINT fk_server FOREIGN KEY (server_discord_id) REFERENCES discord_server(server_discord_id),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES discord_channel(channel_discord_id)
);
GRANT ALL PRIVILEGES ON TABLE polytopia_game TO discordBot;

CREATE TABLE IF NOT EXISTS polytopia_player(
	game_player_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	discord_player_id BIGINT UNIQUE,
	discord_player_name VARCHAR(40),
	polytopia_player_name VARCHAR(40),
	PRIMARY KEY (game_player_uuid)
);
GRANT ALL PRIVILEGES ON TABLE polytopia_player TO discordBot;

CREATE TABLE IF NOT EXISTS game_players(
	discord_player_id BIGINT NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	is_alive BOOLEAN,
	PRIMARY KEY (discord_player_id, channel_discord_id),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES discord_channel(channel_discord_id),
	CONSTRAINT fk_player FOREIGN KEY (discord_player_id) REFERENCES polytopia_player(discord_player_id)
);
GRANT ALL PRIVILEGES ON TABLE game_players TO discordBot;

CREATE TABLE IF NOT EXISTS game_player_scores(
	score_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	discord_player_id BIGINT,
	turn INT,
	score INT,
	confirmed BOOLEAN,
	PRIMARY KEY (score_uuid),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES polytopia_game(channel_discord_id),
	CONSTRAINT fk_player FOREIGN KEY (discord_player_id) REFERENCES polytopia_player(discord_player_id)
);
GRANT ALL PRIVILEGES ON TABLE game_player_scores TO discordBot;

CREATE TABLE IF NOT EXISTS message_resources(
	filename uuid DEFAULT uuid_generate_v4() UNIQUE,
	source_channel_id BIGINT NOT NULL,
	source_message_id BIGINT NOT NULL,
	resource_number INT DEFAULT 0,
	author_id BIGINT,
	operation INT,
	date_added TIMESTAMP default now() NOT NULL,
	PRIMARY KEY (filename),
	CONSTRAINT fk_channel FOREIGN KEY (source_channel_id) REFERENCES discord_channel(channel_discord_id),
	CONSTRAINT fk_player FOREIGN KEY (author_id) REFERENCES polytopia_player(discord_player_id)
);
GRANT ALL PRIVILEGES ON TABLE message_resources TO discordBot;
