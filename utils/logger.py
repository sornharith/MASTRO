"""
Logging utilities for the multi-agent dropout prediction system
"""
import logging
import os
from datetime import datetime


def setup_logger(filename=os.path.join("logs", "agent_timeseries.log")):
    """Setup logging configuration"""
    import os
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def log(msg):
    """Log a message with timestamp"""
    logging.info(msg)
    print(f"{datetime.now().isoformat()} - {msg}")


def log_p(msg):
    """Print a message with timestamp (without logging to file)"""
    print(f"{datetime.now().isoformat()} - {msg}")