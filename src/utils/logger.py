# -*- coding: utf-8 -*-
"""日志工具，统一控制控制台日志格式。"""

import logging


def get_logger(name: str) -> logging.Logger:
    """按名称创建或复用 logger，避免重复挂载 handler。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
