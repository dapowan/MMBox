import json
import os
from typing import Any, Union, List, Dict

import yaml


def json_to_log_lines(obj: Union[dict, list, str], indent: int = 0) -> str:
    """
    把 JSON 对象转换成每行一个 key:value 的字符串。
    支持嵌套 dict / list，会自动缩进。
    """
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)  # 如果传入是 JSON 字符串，先解析
        except json.JSONDecodeError:
            return obj  # 普通字符串

    lines = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(" " * indent + f"{k}:")
                lines.append(json_to_log_lines(v, indent + 2))
            else:
                lines.append(" " * indent + f"{k}: {v}")

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                lines.append(" " * indent + f"-")
                lines.append(json_to_log_lines(item, indent + 2))
            else:
                lines.append(" " * indent + f"- {item}")

    else:
        lines.append(" " * indent + str(obj))

    return "\n".join(lines)


def save_json_to_log(obj: Union[dict, list, str], path: str = "output.log"):
    """
    将 JSON 对象转换为多行字符串并保存到日志文件。
    如果文件不存在，会自动创建；如果目录不存在，会先创建目录。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    content = json_to_log_lines(obj)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return path

def diff_list(a: List[str], b: List[str]) -> List[str]:
    """
    返回在列表 a 中但不在列表 b 中的字符串
    """
    return [x for x in a if x not in b]


def string_to_json(s: str):
    """
    把 JSON 格式字符串解析成 Python 对象
    """
    try:
        obj = json.loads(s)
        return obj
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return None


def save_dict_to_json(data: Dict[str, Any], path: str, indent: int = 2, overwrite: bool = True):
    """
    保存一个字典到 JSON 文件

    Parameters
    ----------
    data : dict
        要保存的字典
    path : str
        保存路径，例如 "output.json"
    indent : int
        缩进空格数，默认 2，方便阅读
    overwrite : bool
        True 表示覆盖写入，False 表示追加写入一行（JSONL）
    """
    mode = "w" if overwrite else "a"
    with open(path, mode, encoding="utf-8") as f:
        if overwrite:
            json.dump(data, f, indent=indent)
        else:
            f.write(json.dumps(data) + "\n")


def find_objects(data: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """
    在 JSON 对象列表中查找指定 key == value 的对象

    Parameters
    ----------
    data : List[Dict[str, Any]]
        JSON list of object
    key : str
        要匹配的 key
    value : Any
        要匹配的 value

    Returns
    -------
    List[Dict[str, Any]]
        所有匹配的对象组成的列表
    """
    return [obj for obj in data if obj.get(key) == value]

def read_yaml_file(path: str):
    """读取 YAML 文件，返回 Python 对象（dict 或 list）"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data