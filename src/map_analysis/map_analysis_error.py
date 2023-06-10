class AnalysisException(Exception):
    
    def __init__(self, message):
        super()
        print("AnalysisException", message)
        self.message = message
    
    