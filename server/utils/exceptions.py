
class AppError(Exception):

    def __init__(self, message, status_code=400, error_code="APP_ERROR"):
        super().__init__(message)

        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ValidationError(AppError):

    def __init__(self, message):
        super().__init__(
            message,
            status_code=400,
            error_code="VALIDATION_ERROR"
        )


class NotFoundError(AppError):

    def __init__(self, message="Resource not found"):
        super().__init__(
            message,
            status_code=404,
            error_code="NOT_FOUND"
        )


class UnauthorizedError(AppError):

    def __init__(self, message="Authentication required"):
        super().__init__(
            message,
            status_code=401,
            error_code="UNAUTHORIZED"
        )
