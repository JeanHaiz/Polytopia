class AnalysisException(Exception):
    
    def __init__(self, message):
        super()
        self.message = message
    
    