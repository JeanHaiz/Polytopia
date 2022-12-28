/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -f docker-entrypoint-initdb.d/DB_0.8_migration.sql
 */
/*
 * psql --dbname=polytopiaHelper_dev -U discordBot --echo-all -c "ALTER TABLE map_patching_process DROP COLUMN patreon_role;"
 */

ALTER TABLE polytopia_player
ADD COLUMN patreon_role TEXT default '',
ADD COLUMN white_list BOOLEAN default false;