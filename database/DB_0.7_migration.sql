/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.7_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "ALTER TABLE map_patching_process DROP COLUMN interaction_id;"
 */

ALTER TABLE map_patching_process
ADD COLUMN interaction_id BIGINT;