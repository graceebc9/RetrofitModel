import logging
import os

def setup_logging(log_level='INFO'):
    """Simple logging setup that can be used across all scripts"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/processing.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    """Get a logger instance for the module"""
    return logging.getLogger(name)