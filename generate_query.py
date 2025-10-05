import json
import os
import json
import random
import re
from string import Formatter
from typing import Any, Dict, List, Tuple, Union

from utils import save_json_to_log, find_objects, diff_list, read_yaml_file, save_dict_to_json, string_to_json
from utils_llm import LLMProxy


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
    # 1) 校验并渲染模板
    field_names = {fname for _, fname, _, _ in Formatter().parse(prompt_template) if fname}
    missing = field_names - set(prompt_values.keys())
    if missing:
        raise KeyError(f"Missing prompt values for placeholders: {sorted(missing)}")

    try:
        prompt = prompt_template.format(**prompt_values)
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

def generate_tags(model, tags_seed, gen_tag_meta, config, log_path):

    def sample_tags_examples(tag_num, example_num):
        chosen_tags = random.sample(tags_seed, min(tag_num, len(tags_seed)))

        results = []
        for tag_obj in chosen_tags:
            examples = tag_obj.get("examples", [])
            chosen_examples = random.sample(examples, min(example_num, len(examples)))
            results.append({
                "tag": tag_obj["tag"],
                "description": tag_obj["description"],
                "examples": chosen_examples
            })
        return results

    prompt_values = {}
    tags_examples = sample_tags_examples(config["gen_tag_num"], config["gen_tag_example_num"])
    prompt_values["tag"] = tags_examples[0]["tag"]
    prompt_values["description"] = tags_examples[0]["description"]
    prompt_values["examples"] = str(tags_examples[0]["examples"])
    prompt_values["n_output"] = config["gen_tag_example_output"]

    tag_target = tags_examples[0]["tag"]
    print(f"Generating tags for {tag_target} with examples from: {tags_examples}")
    generated_results = generate_and_extract(model, gen_tag_meta["prompt_template"], prompt_values,
                                             gen_tag_meta["target_output"]["regex_extract_target_output"]["pattern"])
    tags_examples_generated = string_to_json(generated_results["extracted"][0])
    if tags_examples_generated:
        print(f"Generating tags examples for {tag_target}: {tags_examples_generated}")
        save_json_to_log(generated_results, os.path.join(log_path, "generated_outputs_tags.log"))
        tag_examples = find_objects(tags_seed, "tag", tag_target)[0]
        tags_examples_new = diff_list(tags_examples_generated, tag_examples["examples"])
        print(f"New tag examples insert for {tag_target}: {tags_examples_new}")
        tag_examples["examples"] = tag_examples["examples"] + tags_examples_new
    else:
        print("No tags examples were generated.")
    return tags_seed

def main(model, query_meta_seed, gen_tag_meta, gen_template_meta, gen_query_meta, config, output_dir):
    query_tags = query_meta_seed["tags"]
    for i in range(2):
        output_dir_i = os.path.join(output_dir, f"round_{i}")
        query_tags = generate_tags(model, query_tags, gen_tag_meta, config, output_dir_i)

        query_meta_seed["tags"] = query_tags
        save_dict_to_json(query_meta_seed, os.path.join(output_dir_i, "query_meta.json"))
    pass

if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06'
    model = LLMProxy(name, key)
    output_dir = "log/t0"
    config = {"gen_tag_num":1, "gen_tag_example_num": 5, "gen_tag_example_output": 10}
    with open('dataset/query_meta.json', "r", encoding="utf-8") as f:
        query_meta_seed = json.load(f)
        gen_tag_meta = read_yaml_file("prompt/gen_tag_example.yaml")
        main(model, query_meta_seed, gen_tag_meta, None, None, config, output_dir)