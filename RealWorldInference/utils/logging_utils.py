"""
统一日志工具
所有组件共用，保证日志格式一致：[时间.毫秒] [组件名] [级别] 消息
"""

import logging
import sys
from pathlib import Path
from threading import Lock
from typing import Optional


_LOG_FORMAT = "[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_GLOBAL_FILE_HANDLER: Optional[logging.Handler] = None
_GLOBAL_LOCK = Lock()


class ColorFormatter(logging.Formatter):
    """Colorize console warnings/errors without polluting file logs."""

    COLORS = {
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = self.COLORS.get(record.levelno)
        if color:
            return f"{color}{text}{self.RESET}"
        return text


def setup_logger(
    component: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(component)
    if logger.handlers:
        _attach_global_handler_if_set(logger)
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(ColorFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(stdout_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _attach_global_handler_if_set(logger)
    return logger


def attach_file_log(path: str) -> None:
    """注册进程级共享 file handler，补挂到所有已创建的 logger。幂等。"""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    new_handler = logging.FileHandler(str(target), encoding="utf-8")
    new_handler.setFormatter(formatter)
    new_handler.set_name(f"_global_file_log:{target}")

    global _GLOBAL_FILE_HANDLER
    with _GLOBAL_LOCK:
        old = _GLOBAL_FILE_HANDLER
        if old is not None and getattr(old, "baseFilename", None) == str(target):
            new_handler.close()
            return
        _GLOBAL_FILE_HANDLER = new_handler

    manager = logging.Logger.manager
    for name in list(manager.loggerDict.keys()):
        candidate = logging.getLogger(name)
        if not isinstance(candidate, logging.Logger):
            continue
        if old is not None and old in candidate.handlers:
            candidate.removeHandler(old)
        if new_handler not in candidate.handlers:
            candidate.addHandler(new_handler)

    if old is not None:
        try:
            old.close()
        except Exception:
            pass


def _attach_global_handler_if_set(logger: logging.Logger) -> None:
    handler = _GLOBAL_FILE_HANDLER
    if handler is not None and handler not in logger.handlers:
        logger.addHandler(handler)
