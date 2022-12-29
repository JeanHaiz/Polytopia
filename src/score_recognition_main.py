import sys

from map_analysis import map_analysis
from map_analysis import analysis_callback_utils
from map_analysis.map_analysis_error import AnalysisException
from common import receiver_utils

try:
    receiver = receiver_utils.Receiver(
        "score_recognition",
        map_analysis.map_analysis_request,
        analysis_callback_utils.send_error,
        AnalysisException,
        ["patch_process_id", "map_requirement_id"]
    )
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
