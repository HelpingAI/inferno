import logging
import sys
import os
from typing import Optional

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class InfernoLogger:
    """
    Logger utility for Inferno server.
    Provides consistent logging across all modules.
    """

    def __init__(self, name: str, level: str = "info", log_file: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            name: Logger name (usually module name)
            level: Log level (debug, info, warning, error, critical)
            log_file: Optional file path to write logs to
        """
        self.logger = logging.getLogger(name)

        # Check if this logger already has handlers to avoid duplicate logging
        if not self.logger.handlers:
            self.set_level(level)

            # Create formatter
            formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Create file handler if log_file is provided
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir:  # Only create directory if there's a path component
                    os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        else:
            # If handlers exist, just update the level
            self.set_level(level)

    def set_level(self, level: str):
        """Set the log level."""
        log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        self.logger.setLevel(log_level)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(message)

    def exception(self, message: str):
        """Log an exception message with traceback."""
        self.logger.exception(message)


# Create a default logger instance
logger = InfernoLogger("inferno")

def get_logger(name: str, level: str = "info", log_file: Optional[str] = None) -> InfernoLogger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually module name)
        level: Log level (debug, info, warning, error, critical)
        log_file: Optional file path to write logs to

    Returns:
        InfernoLogger instance
    """
    return InfernoLogger(name, level, log_file)