import sys

from score_recognition import score_recognition_utils
from score_recognition.score_recognition_error import RecognitionException

from common import sender_utils
from common import receiver_utils

try:
    sender = sender_utils.Sender(
        "bot_client",
        "score_recognition",
        "score_recognition123"
    )
    
    
    def score_recognition_error_function(
            process_uuid: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "SCORE_RECOGNITION_ERROR",
                "process_uuid": process_uuid,
                "error": error
            }
        )
    
    
    def score_recognition_callback_function(
            process_uuid: str,
            channel_id: int,
            score_requirement_id: str
    ) -> None:
        sender.send_message(
            {
                "action": "SCORE_RECOGNITION_COMPLETE",
                "process_uuid": process_uuid,
                "channel_id": channel_id,
                "score_requirement_id": score_requirement_id
            }
        )
    
    receiver = receiver_utils.Receiver(
        "score_recognition",
        "score_recognition",
        "score_recognition123",
        score_recognition_utils.score_recognition_request,
        score_recognition_error_function,
        score_recognition_callback_function,
        RecognitionException,
        ["process_uuid"]
    )
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
