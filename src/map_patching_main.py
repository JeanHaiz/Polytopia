import sys

from map_patching import map_patching
from map_patching.map_patching_error import PatchingException

from common import sender_utils
from common import receiver_utils

try:
    sender = sender_utils.Sender(
        "bot_client",
        "map_patching",
        "map_patching123"
    )
    
    
    def error_function(
            patch_id: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_PATCHING_ERROR",
                "patch_uuid": patch_id,
                "error": error
            }
        )
    
    
    def callback_function(
            patch_id: str,
            channel_id: str,
            filename: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_PATCHING_COMPLETE",
                "patch_uuid": patch_id,
                "channel_id": channel_id,
                "filename": filename
            }
        )
    
    
    receiver = receiver_utils.Receiver(
        "map_patching",
        "map_patching",
        "map_patching123",
        map_patching.generate_patched_map_bis,
        error_function,
        callback_function,
        PatchingException,
        ["patch_process_id"]
    )
    
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
