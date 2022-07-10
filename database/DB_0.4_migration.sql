/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.4_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "drop table map_patching_input_param; drop table map_patching_background_param;"
 */

CREATE TABLE IF NOT EXISTS map_patching_input_param (
	filename uuid UNIQUE NOT NULL,
	cloud_scale FLOAT,
	corners integer[4][3],
	PRIMARY KEY (filename),
	CONSTRAINT fk_resource FOREIGN KEY (filename) REFERENCES message_resources(filename)
);
GRANT ALL PRIVILEGES ON TABLE map_patching_input_param TO discordBot;

CREATE TABLE IF NOT EXISTS map_patching_background_param(
	map_size int NOT NULL,
	cloud_scale FLOAT,
	corners integer[4][3],
	PRIMARY KEY (map_size)
);
GRANT ALL PRIVILEGES ON TABLE map_patching_background_param TO discordBot;
