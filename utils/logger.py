"""
日志工具模块
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import settings


class Logger:
    """日志管理类"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str = "wechat_custody") -> logging.Logger:
        """获取日志记录器"""
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(settings.get("logging.level", "INFO"))

        # 清除已有的处理器
        logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        log_dir = Path(__file__).parent.parent / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        max_bytes = settings.get("logging.max_file_size", 10 * 1024 * 1024)
        backup_count = settings.get("logging.backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


# 默认日志记录器
logger = Logger.get_logger()
