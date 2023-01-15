/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.10_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "ALTER TABLE map_patching_process_requirement RENAME TO map_patching_process; ALTER TABLE map_patching_process_input RENAME TO map_patching_process;"
 */

ALTER TABLE map_patching_process_requirement
RENAME COLUMN patch_uuid TO process_uuid;

ALTER TABLE map_patching_process_input
RENAME COLUMN patch_uuid TO process_uuid;
