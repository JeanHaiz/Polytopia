import re
import threading
from typing import List
from typing import Tuple

from interactions import File, Channel, HTTPClient, Client
from interactions.utils.get import get

from common import image_utils
from common.image_operation import ImageOp
from database.database_client import DatabaseClient
from slash_bot_client import map_patching_interface
from slash_bot_client import bot_error_utils
from slash_bot_client import command_context_store as csx
from slash_bot_client.bot_error_utils import MapPatchingErrors

database_client = DatabaseClient(
    user="discordBot",
    password="password123",
    port="5432",
    database="polytopiaHelper_dev",
    host="database"
)


def on_map_analysis_complete(
        patch_id: str,
        map_requirement_id: str
):
    database_client.complete_patching_process_requirement(
        map_requirement_id
    )
    
    print("complete analysis", patch_id, map_requirement_id, flush=True)
    
    if __check_patching_complete(patch_id):
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
    print("Done patching, callback completed", patch_uuid, flush=True)
    # ctx = csx.get(patch_uuid)
    # print("CTX", ctx, flush=True)
    if True:  # ctx is not None:
        # channel = await ctx.get_channel()
        # print("ctx channel", channel, flush=True)
        turn = database_client.get_last_turn(
            channel_id
        )
        channel = await get(client, Channel, object_id=channel_id)
        await channel.send("Done patching")
        # patching_process = database_client.get_patching_process(patch_uuid)
        # patch_filename = str(patching_process["output_filename"])
        channel_info = database_client.get_channel_info(channel_id)
        
        patch_path = image_utils.__get_file_path(
            channel_info["channel_name"],
            ImageOp.MAP_PATCHING_OUTPUT,
            filename  # patch_filename
        )
        
        patching_errors = get_patching_errors(patch_uuid)
        
        print("patching errors", patching_errors, flush=True)
        
        with open(patch_path, "rb") as fh:
            # attachment = File(fp=fh, filename=patch_filename + ".png")
            attachment = File(fp=fh, filename=filename + ".png")
            
            if attachment is not None:
                await channel.send(files=attachment, content="Map patched for turn %s" % turn)
            else:
                patching_errors.append((MapPatchingErrors.ATTACHMENT_NOT_LOADED, None))
                # my_id = '<@338067113639936003>'  # Jean's id
                
                # await answer_message.edit('There was an error. %s has been notified.' % my_id)
                # await ctx.send('There was an error. %s has been notified.' % my_id)
            fh.close()
        
        await bot_error_utils.manage_slash_patching_errors(database_client, channel, channel, patching_errors)


def get_patching_errors(
        patch_uuid: str
) -> List[Tuple[str, str]]:
    patching_status = database_client.get_patching_status(patch_uuid)
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
    print("requirements", all_requirement_check, requirements, flush=True)
    return all_requirement_check
