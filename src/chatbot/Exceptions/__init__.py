"""This module contains custom exceptions for the chatbot application."""


class UploadFileException(RuntimeError):
    """Exception raised when there is an error uploading a file.

    This exception is a base class for file upload related errors.
    It inherits from RuntimeError and provides a custom message.

    Attributes:
        message: The error message describing the upload failure
    """
    def __init__(self, message: str):
        super().__init__(message)


class InvalidFileExtensionError(UploadFileException):
    """Exception raised when a file has an invalid extension.

    This exception is raised when attempting to upload a file with an
    unsupported or invalid file extension. It inherits from UploadFileException
    and provides specific error handling for file extension validation.

    Attributes:
        message: The error message describing the invalid file extension
    """
    def __init__(self, message: str = "The file should be a pdf"):
        super().__init__(message)
