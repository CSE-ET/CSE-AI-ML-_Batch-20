import logging
from pathlib import Path

def setup_logger():
    # Lazy load to avoid circular import
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader().config['logging']
    except Exception:
        # Fallback if config missing
        config = {'level': 'INFO', 'file': 'logs/app.log'}
    
    log_dir = Path(config['file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)