class RecognitionException(Exception):
    
    def __init__(self, message: str):
        super()
        self.message = message
