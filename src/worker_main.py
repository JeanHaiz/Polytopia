import sys

from common import sender_utils
from common import receiver_utils
from common.receiver_utils import ReceiverParams
from header_footer_recognition import header_footer_recognition
from header_footer_recognition.header_footer_recognition_error import HeaderFooterRecognitionException

from map_analysis import map_analysis
from map_analysis.map_analysis_error import AnalysisException

from map_patching import map_patching
from map_patching.map_patching_error import PatchingException
from score_recognition import score_recognition_utils
from score_recognition.score_recognition_error import RecognitionException
from score_visualisation import score_visualisation
from score_visualisation.score_visualisation_error import VisualisationException

try:
    sender = sender_utils.Sender(
        "bot_client",
        "map_analysis",
        "map_analysis123"
    )
    
    
    def map_analysis_error_function(
            process_uuid: str,
            map_requirement_id: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_ANALYSIS_ERROR",
                "process_uuid": process_uuid,
                "map_requirement_id": map_requirement_id,
                "error": error
            }
        )
    
    
    def map_analysis_callback_function(
            process_uuid: str,
            map_requirement_id: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_ANALYSIS_COMPLETE",
                "process_uuid": process_uuid,
                "map_requirement_id": map_requirement_id
            }
        )
    
    
    def map_patching_error_function(
            process_uuid: str,
            error: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_PATCHING_ERROR",
                "process_uuid": process_uuid,
                "error": error
            }
        )
    
    
    def map_patching_callback_function(
            process_uuid: str,
            channel_id: str,
            filename: str
    ) -> None:
        sender.send_message(
            {
                "action": "MAP_PATCHING_COMPLETE",
                "process_uuid": process_uuid,
                "channel_id": channel_id,
                "filename": filename
            }
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
    
    
    def turn_recognition_error_function(
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
    
    
    def turn_recognition_callback_function(
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
        "worker",
        "map_analysis",
        "map_analysis123",
        {
            "MAP_ANALYSIS": ReceiverParams(
                map_analysis.map_analysis_request,
                map_analysis_error_function,
                map_analysis_callback_function,
                AnalysisException,
                ["process_uuid", "map_requirement_id"]
            ),
            "MAP_PATCHING": ReceiverParams(
                map_patching.generate_patched_map_bis,
                map_patching_error_function,
                map_patching_callback_function,
                PatchingException,
                ["process_uuid"]
            ),
            "SCORE_RECOGNITION": ReceiverParams(
                score_recognition_utils.score_recognition_request,
                score_recognition_error_function,
                score_recognition_callback_function,
                RecognitionException,
                ["process_uuid"]
            ),
            "SCORE_VISUALISATION": ReceiverParams(
                score_visualisation.plot_scores,
                score_visualisation_error_function,
                score_visualisation_callback_function,
                VisualisationException,
                ["process_uuid"]
            ),
            "TURN_RECOGNITION": ReceiverParams(
                header_footer_recognition.turn_recognition_request,
                turn_recognition_error_function,
                turn_recognition_callback_function,
                HeaderFooterRecognitionException,
                ["process_uuid", "turn_requirement_id"]
            )
        }
    )
    
    receiver.run()
except KeyboardInterrupt:
    print('Interrupted', flush=True)
    sys.exit(0)
