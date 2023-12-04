class DoesNotMatchError(Exception):
    def __init(self, value, message):
        self.value = value
        self.message = message
        super().__init__(message)
