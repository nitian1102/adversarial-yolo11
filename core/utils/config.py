import os
import yaml
from pathlib import Path
from typing import Dict, Any
import fractions


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """加载并验证配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 路径验证
    required_paths = ['model_path', 'save_dir']
    for path_key in required_paths:
        if not os.path.exists(config['paths'][path_key]) and path_key != 'save_dir':
            raise ValueError(f"Invalid path in config: {config['paths'][path_key]}")

    # 设置绝对路径并递归创建目录
    config['paths']['save_dir'] = os.path.abspath(config['paths'].get('save_dir', 'output'))
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    # 处理 attack.params 中的分数
    def convert_to_float(value):
        if isinstance(value, str):
            # 检查是否是分数格式
            if '/' in value:
                try:
                    return float(fractions.Fraction(value))
                except ValueError:
                    raise ValueError(f"Invalid fraction format: {value}")
            else:
                try:
                    return float(value)
                except ValueError:
                    raise ValueError(f"Invalid numeric value: {value}")
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise ValueError(f"Unexpected value type: {type(value)} for value {value}")

    # 遍历 attack.params 并转换值
    for key, value in config['attack']['params'].items():
        config['attack']['params'][key] = convert_to_float(value)

    return config