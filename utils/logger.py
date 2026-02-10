import logging
import sys
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

    # Ensure stdout uses UTF-8 where possible (helps Windows consoles)
    try:
        # Python 3.7+: reconfigure is available
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        # If not available, ignore — best-effort
        pass

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(config['file'], encoding='utf-8')
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config['level']))
    # Remove any existing handlers to avoid duplicate logs during repeated imports
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return logging.getLogger(__name__)