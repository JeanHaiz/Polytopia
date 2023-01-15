import sys

from header_footer_recognition import header_footer_recognition
from header_footer_recognition.header_footer_recognition_error import HeaderFooterRecognitionException

from common import sender_utils
from common import receiver_utils

try:
    sender = sender_utils.Sender(
        "bot_client",
        "header_footer_recognition",
        "header_footer_recognition123"
    )
    
    
    def error_function(
            process_uuid: str,
            turn_requirement_id: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "TURN_RECOGNITION_ERROR",
                "process_uuid": process_uuid,
                "turn_requirement_id": turn_requirement_id,
                "error": error
            }
        )
    
    
    def callback_function(
            process_uuid: str,
            turn_requirement_id: str
    ) -> None:
        sender.send_message(
            {
                "action": "TURN_RECOGNITION_COMPLETE",
                "process_uuid": process_uuid,
                "turn_requirement_id": turn_requirement_id
            }
        )
    
    
    receiver = receiver_utils.Receiver(
        "header_footer_recognition",
        "header_footer_recognition",
        "header_footer_recognition123",
        header_footer_recognition.turn_recognition_request,
        error_function,
        callback_function,
        HeaderFooterRecognitionException,
        ["process_uuid", "turn_requirement_id"]
    )
    
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
