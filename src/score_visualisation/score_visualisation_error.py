class VisualisationException(Exception):
    
    def __init__(self, message):
        super()
        self.message = message

