import json
import os
from typing import List

from utils import load_json, read_yaml_file, save_json_to_log, write_jsonl, string_to_json
from utils_llm import LLMProxy, generate_and_extract
import re
import random

QUERY_TYPES = ["Time", "Duration", "Frequency", "Existence", "Action", "Place", "Summary"]


def render_query(template: str, tags_dict: List[dict]) -> str:
    """
    输入:
      template: 带有<tag>的query模板，比如 "How long was I at <place> on <time>?"
      tags_dict: 一个dict，key是"tags"，value是包含(tag, description, examples)的列表
    输出:
      替换后的query字符串
    """
    # 建立 tag -> examples 的索引
    tag_map = {item["tag"]: item["examples"] for item in tags_dict}

    # 找到所有 <...> 模式的tag
    matches = re.findall(r"<[^<>]+>", template)

    rendered = template
    for tag in matches:
        if tag in tag_map:
            example = random.choice(tag_map[tag])  # 从例子中随机采样
            rendered = rendered.replace(tag, example, 1)  # 只替换一次，避免多个相同tag都被替换成同一个例子

    return rendered


def generate_query(model, query_prompt_meta, query_meta, types_to_gen, config, output_path=None, log_dir=None, num_per_template=1):
    results = []
    if types_to_gen is None:
        types_to_gen = [q["type"] for q in query_meta.get("questions", [])]

    total = len(types_to_gen)
    tags_values = query_meta["tags"]
    num_per_template = config["num_per_template"]
    id = 0
    for idx, q_type in enumerate(types_to_gen, start=1):
        print(f"[{idx}/{total}] Processing type: {q_type}")

        matched = next((q for q in query_meta["questions"] if q["type"] == q_type), None)
        if matched:
            # Combine templates and backup templates
            templates = matched.get("templates", []) + matched.get("templates_backup", [])
            for t, template in enumerate(templates):
                print(f"[{t}/{len(templates)}] Processing template: {template}")
                for n in range(num_per_template):
                    query = render_query(template, tags_values)

                    prompt_values = {}
                    prompt_values["user_query"] = query

                    generated_results = generate_and_extract(model, query_prompt_meta["prompt_template"], prompt_values,
                                                             query_prompt_meta["target_output"]["regex_extractors"]
                                                             ["json_object"]["pattern"])
                    query_generated = string_to_json(generated_results["extracted"][0])
                    if query_generated:
                        save_json_to_log(generated_results, os.path.join(log_dir, f"generated_outputs_query_{id}.log"))
                        query_new = query_generated["final_query"]
                        results.append({
                            "query_id": id,
                            "query_type": q_type,
                            "query_template": template,
                            "query": query_new
                        })
                        print(f"Extracting query: \"{query}\", get new query: {query_new}")
                    else:
                        print("No new query were generated.")
                    print("-----------------------------------")
                    id += 1
        else:
            print(f"  [!] Type '{q_type}' not found in data")
    write_jsonl(results, os.path.join(output_path, "query.jsonl"))
    return results


if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06'
    model = LLMProxy(name, key)
    output_dir = "dataset/qa_v0"
    log_dir = "log/gen_query_v0"
    config = {"num_per_template": 20}
    query_meta = load_json("dataset/query_meta.json")
    query_prompt_meta = read_yaml_file("prompt/gen_query_rewrite.yaml")
    results = generate_query(model, query_prompt_meta, query_meta, QUERY_TYPES, config,
                             log_dir=log_dir, output_path=output_dir)