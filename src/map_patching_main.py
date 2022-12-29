import sys

from map_patching import map_patching, patching_callback_utils
from map_patching.map_patching_error import PatchingException

from common import receiver_utils

try:
    receiver = receiver_utils.Receiver(
        "map_patching",
        map_patching.generate_patched_map_bis,
        patching_callback_utils.send_error,
        PatchingException,
        ["patch_process_id"]
    )
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
