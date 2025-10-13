import types
from math import ceil
import httpx
import yaml
import tiktoken
import base64
import openai
from openai import OpenAI
import re
from string import Formatter
from typing import Any, Dict, List, Tuple, Union

base_url = 'https://api.bltcy.ai/v1'

class LLMProxy:

    def __init__(self, model, key, chat_config=None, thinking_mode=False, running_mode='real'):
        self.client = OpenAI(api_key=key, base_url=base_url) #
        self.running_mode = running_mode
        self.model = model
        self.seed = chat_config.seed if chat_config is not None else None
        self.chat_config = chat_config
        self.thinking_mode = thinking_mode

    def generate_text(self, input_text, system_text=None):
        input_text_gpt = self.wrap_input(input_text, system_text)
        if self.running_mode == 'test':
            response = "test"
            return response, (0, 0)
        try:
            response = self.client.chat.completions.create(model=self.model, messages=input_text_gpt,
                                                           extra_body={"enable_thinking": self.thinking_mode})
            return response.choices[0].message.content, self.extract_usage(response.usage)
        except Exception as e:
            print(e)
            return None, (0, 0)

    def extract_usage(self, usage):
        return usage.prompt_tokens, usage.completion_tokens

    def wrap_input(self, input_text, system_text):
        if system_text:
            input_text_gpt = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": input_text}
            ]
        else:
            input_text_gpt = [
                {"role": "user", "content": input_text}
            ]
        return input_text_gpt


def generate_and_extract(
    model: Any,
    prompt_template: str,
    prompt_values: Dict[str, Any],
    regex: str,
    regex_flags: int = re.DOTALL,
):
    """
    使用大模型生成文本并提取目标值，同时把记录追加保存到 JSONL 文件。

    Parameters
    ----------
    model : Any
        具备 `response, usage = model.generate_text(prompt)` 接口的模型对象。
        其中 usage 是 (input_tokens, output_tokens) 的 tuple。
    prompt_template : str
        包含占位符的模板，如 "City: {city}, Date: {date}"。
    prompt_values : Dict[str, Any]
        用于替换模板占位符的字典，键名需与模板中的占位符一致。
    regex : str
        用于从模型回复中提取目标内容的正则表达式。
        - 若包含捕获分组：返回每个分组的匹配（多个分组则返回 tuple）。
        - 若不包含分组：返回完整匹配字符串。
    output_path : str, default "generated_outputs.jsonl"
        生成记录的保存路径（JSON Lines，每条一行）。
    regex_flags : int, default 0
        传递给 `re.compile` 的 flags，例如 re.DOTALL | re.MULTILINE。

    Returns
    -------
    Extracted : List[Union[str, Tuple[str, ...]]]
        提取到的值列表。可能是字符串列表，或分组构成的 tuple 列表。
    """
    # 1) Find all placeholders like {{...}}
    field_names = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", prompt_template))

    # 2) Check missing values
    missing = field_names - set(prompt_values.keys())
    if missing:
        raise KeyError(f"Missing prompt values for placeholders: {sorted(missing)}")

    # 3) Replace them safely
    def replace_placeholder(match):
        key = match.group(1).strip()
        try:
            return str(prompt_values[key])
        except KeyError:
            raise ValueError(f"Missing value for placeholder: {key}")

    try:
        prompt = re.sub(r"\{\{\s*(\w+)\s*\}\}", replace_placeholder, prompt_template)
    except Exception as e:
        raise ValueError(f"Failed to render prompt_template with prompt_values: {e}") from e

    # 2) 调用大模型
    response, usage = model.generate_text(prompt)

    # 3) 正则提取
    pattern = re.compile(regex, flags=regex_flags)
    matches = pattern.findall(response)

    # 规范化 findall 的返回：
    if not matches:
        extracted = []
    else:
        if isinstance(matches[0], tuple):
            extracted = [tuple(m) for m in matches]
        else:
            extracted = list(matches)

    # 4) 组装记录并保存（追加写入 JSONL）
    rec = {
        "prompt_template": prompt_template,
        "prompt_values": prompt_values,
        "rendered_prompt": prompt,
        "response": response,
        "regex": regex,
        "extracted": extracted,
    }

    # 兼容 usage 是 tuple 或 dict 的情况
    if isinstance(usage, tuple) and len(usage) == 2:
        rec["usage"] = {"input_tokens": usage[0], "output_tokens": usage[1]}
    else:
        rec["usage"] = usage
    return rec

if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06'
    # model_config = types.SimpleNamespace(**{'name': name})
    model = LLMProxy(name, key, None, running_mode='real')
    response, usage = model.generate_text('Hello', 'you are a helpful assistant')
