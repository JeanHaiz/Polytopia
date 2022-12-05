/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.6_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "drop table map_patching_process_requirement;"
 */

CREATE TABLE IF NOT EXISTS map_patching_process_requirement(
	patch_requirement_uuid uuid DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
	patch_uuid uuid,
	filename uuid,
	type TEXT,
	status TEXT,
	complete BOOLEAN default false,
	PRIMARY KEY(patch_requirement_uuid),
	CONSTRAINT fk_patch FOREIGN KEY (patch_uuid) REFERENCES map_patching_process(patch_uuid)
);
GRANT ALL PRIVILEGES ON TABLE map_patching_process_requirement TO discordBot;

CREATE TABLE IF NOT EXISTS message_resource_header(
    filename uuid UNIQUE NOT NULL,
    turn_value integer,
    score_value integer,
    stars_count integer,
    start_per_turn integer,
    tribe TEXT,
    rank integer,
    PRIMARY KEY(filename),
    CONSTRAINT fk_message_resource FOREIGN KEY (filename) REFERENCES message_resources(filename)
);
GRANT ALL PRIVILEGES ON TABLE message_resource_header TO discordBot;