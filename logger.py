import logging, logging.handlers, pathlib
from config import load_config

def get_logger(name: str = "app") -> logging.Logger:
    cfg = load_config()
    log_path = pathlib.Path(cfg.log.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.log.level.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
