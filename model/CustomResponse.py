class CustomResponse:
    def __init__(self, statusCode, message, data=None):
        self.statusCode = statusCode
        self.message = message
        self.data = data

        @property
        def statusCode():
            return self.statusCode

        @statusCode.setter
        def statusCode(value):
            self.statusCode = value

        @property
        def data():
            return self.data

        @data.setter
        def data(value):
            self.data = data

        @property
        def message():
            return self.message
