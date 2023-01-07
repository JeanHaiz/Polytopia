import sys

from map_analysis import map_analysis
from map_analysis.map_analysis_error import AnalysisException

from common import sender_utils
from common import receiver_utils

try:
    sender = sender_utils.Sender(
        "bot_client",
        "map_analysis",
        "map_analysis123"
    )
    
    
    def error_function(
            patch_uuid: str,
            map_requirement_id: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_ANALYSIS_ERROR",
                "patch_uuid": patch_uuid,
                "map_requirement_id": map_requirement_id,
                "error": error
            }
        )
    
    
    def callback_function(
            patch_uuid: str,
            map_requirement_id: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_ANALYSIS_COMPLETE",
                "patch_uuid": patch_uuid,
                "map_requirement_id": map_requirement_id
            }
        )
    
    
    receiver = receiver_utils.Receiver(
        "map_analysis",
        "map_analysis",
        "map_analysis123",
        map_analysis.map_analysis_request,
        error_function,
        callback_function,
        AnalysisException,
        ["patch_uuid", "map_requirement_id"]
    )
    
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
