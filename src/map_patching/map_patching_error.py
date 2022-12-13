from typing import Optional

from common.error_utils import MapPatchingErrors


class PatchingException(Exception):
    
    def __init__(self, message: str, patching_error: Optional[MapPatchingErrors]):
        super()
        self.message = message
        self.patching_error = patching_error
