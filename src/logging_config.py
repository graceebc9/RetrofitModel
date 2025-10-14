import logging
import os
import sys

# Global flag to ensure we only configure once
_logging_configured = False

def setup_logging(log_level='INFO', log_path='logs/processing.log'):
    """
    Setup logging configuration that can only be called once.
    Prevents multiple configurations from overriding each other.
    """
    global _logging_configured
    
    if _logging_configured:
        logging.getLogger(__name__).warning("Logging already configured, skipping setup")
        return
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Clear any existing handlers (in case basicConfig was called elsewhere)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    _logging_configured = True
    
    # Log that setup is complete
    logging.getLogger(__name__).info(f"Logging configured at {log_level} level")


def get_logger(name):
    """Get a logger instance for the module"""
    # Ensure logging is set up
    global _logging_configured
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)