
import logging
from datetime import datetime
import os 

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir, budget, prob_loft, equity_factor):
    """
    Set up logging with separate files for detailed logs and summary statistics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed log
    detail_log_path = os.path.join(
        output_dir, 
        f'detailed_log_budget{budget}_loft{prob_loft}_equity{equity_factor}_{timestamp}.log'
    )
    detail_logger = logging.getLogger(f'detail_{budget}_{prob_loft}_{equity_factor}_{timestamp}')
    detail_logger.setLevel(logging.DEBUG)
    detail_logger.handlers.clear()
    
    detail_handler = logging.FileHandler(detail_log_path)
    detail_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    detail_logger.addHandler(detail_handler)
    
    # Create summary log
    summary_log_path = os.path.join(
        output_dir, 
        f'SUMMARY_budget{budget}_loft{prob_loft}_equity{equity_factor}_{timestamp}.txt'
    )
    summary_logger = logging.getLogger(f'summary_{budget}_{prob_loft}_{equity_factor}_{timestamp}')
    summary_logger.setLevel(logging.INFO)
    summary_logger.handlers.clear()
    
    summary_handler = logging.FileHandler(summary_log_path)
    summary_handler.setFormatter(logging.Formatter('%(message)s'))
    summary_logger.addHandler(summary_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    summary_logger.addHandler(console_handler)
    
    return summary_logger, detail_logger
