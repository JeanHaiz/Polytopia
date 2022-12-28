import os
import re
from typing import List
from typing import Tuple
from typing import Optional

from interactions import File
from interactions import Channel
from interactions import Client
from interactions.utils.get import get

from common import image_utils
from slash_bot_client import bot_error_utils
from common.image_operation import ImageOp
from database.database_client import get_database_client
from slash_bot_client import map_patching_interface
from common.error_utils import MapPatchingErrors

DEBUG = os.getenv("DEBUG")
database_client = get_database_client()


async def on_analysis_error(
        patch_id: str,
        map_requirement_id: str,
        client: Client,
        error: str
):
    await __on_error(patch_id, map_requirement_id, client, error)


async def on_patching_error(
        patch_uuid: str,
        client: Client,
        error: str
):
    await __on_error(patch_uuid, None, client, error)


async def __on_error(
        patch_id: str,
        requirement_id: Optional[str],
        client: Client,
        error: str
):
    if DEBUG:
        print("Error message received", patch_id, requirement_id, client, error, flush=True)

    database_client.update_patching_process_status(patch_id, "ERROR - %s" % error)
    
    if requirement_id is not None:
        database_client.update_patching_process_requirement(patch_id, requirement_id, "ERROR - %s" % error)
    
    patch_info = database_client.get_patching_process(patch_id)
    error_channel = await get(client, Channel, object_id=int(os.getenv("DISCORD_ERROR_CHANNEL")))
    
    if DEBUG:
        print("Error management", patch_info is not None, patch_info, error_channel, flush=True)
        
    if patch_info is not None:
        channel_id = patch_info["channel_discord_id"]
        channel = await get(client, Channel, object_id=channel_id)
        await channel.send('There was an error. <@%s> has been notified.' % os.getenv("DISCORD_ADMIN_USER"))
        
        channel_info = database_client.get_channel_info(channel_id)
        server_name = database_client.get_server_name(channel_info["server_discord_id"])
        await error_channel.send(
            f"""Hey <@{os.getenv("DISCORD_ADMIN_USER")}>,\n""" +
            f"""Error in channel {channel.name} on server {server_name}:\n{error}\n""")
    else:
        print(error, flush=True)
        await error_channel.send(
            f"""Hey <@{os.getenv("DISCORD_ADMIN_USER")}>,\n""" +
            f"""Error in unknown channel for\npatch {patch_id}, \nrequirement {requirement_id}""")


def on_map_analysis_complete(
        patch_id: str,
        map_requirement_id: str
):
    database_client.complete_patching_process_requirement(
        map_requirement_id
    )
    
    if DEBUG:
        print("complete analysis", patch_id, map_requirement_id, flush=True)
    
    if __check_patching_complete(patch_id):
        if DEBUG:
            print("sending patching request")
        
        map_patching_interface.send_map_patching_request(
            patch_id,
            number_of_images=None
        )


async def on_map_patching_complete(
        client: Client,
        patch_uuid: str,
        channel_id: int,
        filename: str
):
    if DEBUG:
        print("Done patching, callback completed", patch_uuid, flush=True)
    
    turn = database_client.get_last_turn(
        channel_id
    )
    channel = await get(client, Channel, object_id=channel_id)
    # await channel.send("Done patching")
    channel_info = database_client.get_channel_info(channel_id)
    
    patch_path = image_utils.__get_file_path(
        channel_info["channel_name"],
        ImageOp.MAP_PATCHING_OUTPUT,
        filename
    )
    
    patching_errors = get_patching_errors(patch_uuid)
    
    if DEBUG:
        print("patching errors", patching_errors, flush=True)
    
    with open(patch_path, "rb") as fh:
        attachment = File(fp=fh, filename=filename + ".png")
        
        if attachment is not None:
            await channel.send(files=attachment, content="Map patched for turn %s" % turn)
        else:
            patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
        fh.close()
    
    await bot_error_utils.manage_slash_patching_errors(database_client, channel, channel, patching_errors)


def get_patching_errors(
        patch_uuid: str
) -> List[Tuple[str, str]]:
    patching_status = database_client.get_patching_status(patch_uuid)
    
    if DEBUG:
        print("patching status", patching_status, flush=True)
    
    if patching_status.startswith("ERRORS - "):
        return [
            (re.search(r"([A-Z_]+)\(", status).group(1), re.search(r"([a-z0-9-]{36}|None)", status).group(0))
            for status in patching_status[len("ERRORS - "):].split(";")]


def on_turn_recognition_complete(patch_id, turn_requirement_id):
    database_client.complete_patching_process_requirement(
        turn_requirement_id
    )
    
    if __check_patching_complete(patch_id):
        map_patching_interface.send_map_patching_request(patch_id, number_of_images=None)


def __check_patching_complete(patch_process_id: str):
    requirements = database_client.get_patching_process_requirement(patch_process_id)
    all_requirement_check = all([r["complete"] for r in requirements])
    
    if DEBUG:
        print("requirements", all_requirement_check, requirements, flush=True)
    
    return all_requirement_check
