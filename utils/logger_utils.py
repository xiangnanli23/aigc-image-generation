import os

from loguru import logger



def build_logger(
    log_dir: str = None,
    add_info_log: bool = True,
    add_error_log: bool = True,
    rotation: str = '10 MB',
    retention: str = '6 months',
    ) -> None:
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    if add_info_log:
        log_info_path  = os.path.join(log_dir, "info.log")
        logger.add(log_info_path, level="INFO", rotation=rotation, retention=retention)
    if add_error_log:
        log_error_path = os.path.join(log_dir, "error.log")
        logger.add(log_error_path, level="ERROR", rotation=rotation, retention=retention)
    
w