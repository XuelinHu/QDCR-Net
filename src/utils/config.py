# -*- coding: utf-8 -*-
"""配置加载工具。"""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置文件并返回字典。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
