import os
import json
import random
from typing import List, Dict, Any, Tuple

from utils import write_jsonl, read_jsonl


def convert_and_split_messages(
    data: List[Dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
    save_dir: str = None,
    save_name: str = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将对象列表转换为 messages 格式，并随机划分为训练和测试集，
    如果提供保存路径，则保存为 train.jsonl / test.jsonl。

    Parameters
    ----------
    data : list[dict]
        包含 prompt 和 response_workflow 的对象列表。
    test_ratio : float
        测试集占比 (默认 0.2)。
    seed : int
        随机种子，保证可复现。
    save_dir : str
        保存目录（例如 "./output"）。若为 None，则不保存。

    Returns
    -------
    (train_set, test_set)
    """
    # 转换格式
    formatted = [
        {
            "messages": [
                {"role": "user", "content": obj.get("prompt", "")},
                {"role": "assistant", "content": obj.get("response_workflow", "")}
            ]
        }
        for obj in data
    ]

    # 随机划分
    random.seed(seed)
    random.shuffle(formatted)
    split_idx = int(len(formatted) * (1 - test_ratio))
    train_set = formatted[:split_idx]
    test_set = formatted[split_idx:]

    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        train_path = os.path.join(save_dir, f"{save_name}_train.jsonl")
        test_path = os.path.join(save_dir, f"{save_name}_test.jsonl")
        write_jsonl(train_set, train_path)
        write_jsonl(test_set, test_path)

    return train_set, test_set


if __name__ == "__main__":
    input_file = f"qa_v3/results_generate_workflow.jsonl"
    results = read_jsonl(input_file)
    convert_and_split_messages(results, save_dir='.', save_name='v1')