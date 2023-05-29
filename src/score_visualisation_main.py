import sys

from score_visualisation import score_visualisation
from score_visualisation.score_visualisation_error import VisualisationException

from common import sender_utils
from common import receiver_utils

try:
    sender = sender_utils.Sender(
        "bot_client",
        "score_visualisation",
        "score_visualisation123"
    )
    
    
    def score_visualisation_error_function(
            process_uuid: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "SCORE_VISUALISATION_ERROR",
                "process_uuid": process_uuid,
                "error": error
            }
        )
    
    
    def score_visualisation_callback_function(
            process_uuid: str,
            channel_id: str,
            filename: str
    ) -> None:
        sender.send_message(
            {
                "action": "SCORE_VISUALISATION_COMPLETE",
                "process_uuid": process_uuid,
                "channel_id": channel_id,
                "filename": filename
            }
        )
    
    
    receiver = receiver_utils.Receiver(
        "score_visualisation",
        "score_visualisation",
        "score_visualisation123",
        score_visualisation.plot_scores,
        score_visualisation_error_function,
        score_visualisation_callback_function,
        VisualisationException,
        ["process_uuid"]
    )
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
