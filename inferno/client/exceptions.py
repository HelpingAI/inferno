"""
Exceptions for the Inferno client.
"""

class InfernoError(Exception):
    """Base exception for all Inferno client errors."""
    pass


class InfernoAPIError(InfernoError):
    """Exception raised when the Inferno API returns an error."""
    
    def __init__(self, status_code, message, response=None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"API error (status: {status_code}): {message}")


class InfernoConnectionError(InfernoError):
    """Exception raised when there's a connection error to the Inferno API."""
    
    def __init__(self, message, original_error=None):
        self.original_error = original_error
        super().__init__(f"Connection error: {message}")


class InfernoTimeoutError(InfernoError):
    """Exception raised when a request to the Inferno API times out."""
    
    def __init__(self, timeout, operation=None):
        self.timeout = timeout
        self.operation = operation
        message = f"Request timed out after {timeout} seconds"
        if operation:
            message += f" during {operation}"
        super().__init__(message)
