import json
import os
from typing import List, Dict, Any

from program_analyzer import PythonProgramAnalyzer
from statistic import evaluate_from_reports
from utils import load_json, read_yaml_file, save_json_to_log, string_to_json, save_dict_to_json, write_jsonl, \
    read_jsonl
from utils_llm import LLMProxy, generate_and_extract
import random

TOOLS_LIST = []
QUERY_TYPES = ["Time", "Duration", "Frequency", "Existence", "Action", "Place", "Summary"]


def filter_queries(
    query_list: List[Dict],
    allowed_types: List[str]
) -> List[Dict]:
    """
    Filter a list of query dicts by allowed query_types and remove redundant queries.

    Parameters
    ----------
    query_list : List[Dict]
        A list of query dictionaries, each containing at least 'query_type' and 'query' keys.
    allowed_types : List[str]
        A list of query_type strings to keep.

    Returns
    -------
    List[Dict]
        A filtered list of unique queries (by 'query'), preserving input order.
    """
    seen_queries = set()
    filtered = []
    for q in query_list:
        q_type = q.get("query_type")
        q_text = q.get("query")
        # Filter by allowed type and avoid duplicates
        if q_type in allowed_types and q_text not in seen_queries:
            seen_queries.add(q_text)
            filtered.append(q)
    return filtered


def extract_tool_names(tools_json: Dict[str, Any]) -> List[str]:
    """
    提取 JSON 中所有工具（internal + external）的名称。

    Parameters
    ----------
    tools_json : dict
        含有 "internal_tools" 和/或 "external_tools" 的 JSON 对象。

    Returns
    -------
    List[str]
        所有工具名称的列表。
    """
    names = []
    for key in ["internal_tools", "external_tools"]:
        if key in tools_json and isinstance(tools_json[key], list):
            for tool in tools_json[key]:
                if isinstance(tool, dict) and "name" in tool:
                    names.append(tool["name"])
    return names

def build_query_response_workflow(model, agent_prompt_meta, tools_meta, queries, types_to_check, num_target=None,
                                  output_path=None, log_dir=None):
    results = []
    query_filtered = filter_queries(queries, types_to_check)
    if num_target:
        query_filtered = random.sample(query_filtered, num_target)
    total = len(query_filtered)

    tool_names = extract_tool_names(tools_meta)
    analyzer = PythonProgramAnalyzer(tool_names)

    for idx, q in enumerate(query_filtered, start=1):
        q_id = q["query_id"]
        q_type = q["query_type"]
        query = q["query"]
        print(f"[{idx}/{total}] Processing type: {q_type}, query: \"{query}\"")

        prompt_values = {}
        prompt_values["user_query"] = query
        prompt_values["internal_tools"] = tools_meta["internal_tools"]
        prompt_values["external_tools"] = tools_meta["external_tools"]

        generated_results = generate_and_extract(model, agent_prompt_meta["prompt_template"], prompt_values,
                                                 agent_prompt_meta["target_output"]["regex_extractors"]
                                                 ["prog_block"]["pattern"])
        try:
            workflow_generated = generated_results["extracted"][0]
            if workflow_generated:
                print(f"Extracting workflow for query: \"{query}\"")
                analyzer_report = analyzer.analyze(workflow_generated)
                save_json_to_log(generated_results, os.path.join(log_dir, f"generated_outputs_workflow_{q_id}.log"))
                results.append({
                    "query_id": q_id,
                    "query_type": q_type,
                    "query": query,
                    "response_workflow": workflow_generated,
                    "report":  analyzer_report,
                    "prompt": generated_results["rendered_prompt"]
                })
                # tag_examples = find_objects(tags_seed, "tag", tag_target)[0]
                # tags_examples_new = diff_list(tags_examples_generated, tag_examples["examples"])
                print(f"New workflow for query: \"{query}\", errors:{analyzer_report['errors']}")
                # tag_examples["examples"] = tag_examples["examples"] + tags_examples_new
            else:
                print("No workflow examples were generated.")
                results.append({
                    "query_id": q_id,
                    "query_type": q_type,
                    "query": query,
                    "response_workflow": "",
                    "report": {},
                    "prompt": generated_results["rendered_prompt"]
                })
        except Exception as e:
            print("No workflow examples were generated.")
            results.append({
                "query_id": q_id,
                "query_type": q_type,
                "query": query,
                "response_workflow": "",
                "report": {},
                "prompt": generated_results["rendered_prompt"]
            })
        print("-----------------------------------")
        write_jsonl(results, os.path.join(output_path, "results_generate_workflow.jsonl"))
    return results


if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06' #  gpt-oss-120b gpt-4o-2024-08-06 qwen3-8b
    model = LLMProxy(name, key)
    version = 'gpt_v1'
    output_dir = f"dataset/qa_{version}"
    log_dir = f"log/gen_response_workflow_{version}"
    queries = read_jsonl("dataset/qa_gpt_v1/query.jsonl")
    tools_meta = load_json("dataset/tools/tools_v1.json")
    agent_prompt_meta = read_yaml_file("prompt/agent_workflow_template_v3.yaml")
    results = build_query_response_workflow(model, agent_prompt_meta, tools_meta, queries, types_to_check=QUERY_TYPES,
                                            log_dir=log_dir, output_path=output_dir) # , num_target=500
    report_summary = evaluate_from_reports(results)
    print(report_summary["error_summary"])