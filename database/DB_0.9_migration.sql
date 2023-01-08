/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.9_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "ALTER TABLE operation_process DROP COLUMN process_type; ALTER TABLE operation_process RENAME TO map_patching_process;"
 */

ALTER TABLE map_patching_process
RENAME TO operation_process;

ALTER TABLE operation_process
ADD COLUMN process_type TEXT;

UPDATE operation_process
SET process_type='MAP_PATCHING';

ALTER TABLE operation_process
ALTER COLUMN process_type SET NOT NULL;

ALTER TABLE operation_process
RENAME COLUMN patch_uuid TO process_uuid;