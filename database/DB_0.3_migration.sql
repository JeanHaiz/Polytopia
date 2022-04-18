/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.3_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "drop table map_patching_process_input; drop table map_patching_process;"
 */

CREATE TABLE IF NOT EXISTS map_patching_process(
	patch_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	channel_discord_id BIGINT NOT NULL,
	process_author_discord_id BIGINT NOT NULL,
	status TEXT,
	output_filename uuid,
	started_on TIMESTAMP default now() NOT NULL,
	ended_on TIMESTAMP,
	PRIMARY KEY (patch_uuid),
	CONSTRAINT fk_channel FOREIGN KEY (channel_discord_id) REFERENCES polytopia_game(channel_discord_id),
	CONSTRAINT fk_output FOREIGN KEY (output_filename) REFERENCES message_resources(filename)
);
GRANT ALL PRIVILEGES ON TABLE map_patching_process TO discordBot;

CREATE TABLE IF NOT EXISTS map_patching_process_input(
	patch_input_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	patch_uuid uuid,
	input_filename uuid,
	status TEXT,
	input_order INT,
	PRIMARY KEY(patch_input_uuid),
	CONSTRAINT fk_patch FOREIGN KEY (patch_uuid) REFERENCES map_patching_process(patch_uuid),
	CONSTRAINT fk_input FOREIGN KEY (input_filename) REFERENCES message_resources(filename)
);
GRANT ALL PRIVILEGES ON TABLE map_patching_process_input TO discordBot;
