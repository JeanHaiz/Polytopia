/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.5_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "drop table score_visualisation; drop score_visualisation_input;"
 */

CREATE TABLE IF NOT EXISTS score_visualisation(
	visualisation_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	visualisation_author_discord_id BIGINT NOT NULL,
	status TEXT,
	started_on TIMESTAMP default now() NOT NULL,
	ended_on TIMESTAMP,
	PRIMARY KEY (visualisation_uuid),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES polytopia_game(channel_discord_id)
);
GRANT ALL PRIVILEGES ON TABLE score_visualisation TO discordBot;

CREATE TABLE IF NOT EXISTS score_visualisation_input(
	visualisation_input_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	visualisation_uuid uuid,
	score_uuid uuid,
	status TEXT,
	input_order INT,
	PRIMARY KEY(visualisation_input_uuid),
	CONSTRAINT fk_patch FOREIGN KEY (visualisation_uuid) REFERENCES score_visualisation(visualisation_uuid),
	CONSTRAINT fk_input FOREIGN KEY (score_uuid) REFERENCES game_player_scores(score_uuid)
);
GRANT ALL PRIVILEGES ON TABLE score_visualisation_input TO discordBot;
