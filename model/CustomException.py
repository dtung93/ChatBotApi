class CustomException(Exception):
    def __init__(self, message, statusCode=500, error=None):
        self.message = message
        self.statusCode = statusCode
        self.error = error
