import json
import os
import re
import numpy as np
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template, load_dataset

from generate_workflow_from_query import extract_tool_names
from program_analyzer import PythonProgramAnalyzer
from statistic import compare_report_sets
from utils import write_jsonl, read_yaml_file, load_json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_new_tokens = 1024
temperature = 0



def summarize_metrics(summary: dict):
    # 收集所有 metric key
    keys = set()
    for v in summary.values():
        keys.update(v["metrics"].keys())
    keys = sorted(keys)

    results = {}
    for k in keys:
        values = [v["metrics"][k] for v in summary.values() if k in v["metrics"]]
        arr = np.array(values, dtype=float)
        mean, std = arr.mean(), arr.std()
        results[k] = {"avg": mean, "std": std}

    # 打印结果
    print("\n===== Metrics Summary =====")
    for k, v in results.items():
        print(f"{k}: avg={v['avg']:.4f}, std={v['std']:.4f}")

    return results

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    # print(f'query: {query}')
    # print(f'response: {response}')
    return response


def infer_stream(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=True)
    gen_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end='', flush=True)
    print()


def extract_query_response_from_messages(messages):
    query, gt = None, None
    for i, m in enumerate(messages):
        if m['role'] == 'user':
            query = m['content']
            gt = None
            for j in range(i + 1, len(messages)):
                if messages[j]['role'] == 'assistant':
                    gt = messages[j]['content']
                    break
    if query and gt:
        return query, gt
    return None, None

def evaluate_metric_func(response, gt):
    return {'a': 1.0, 'b': 2.0}

def evaluate(model, checkpoint, agent_prompt_meta, tools_meta, test_dataset=None, output_path=None, query_list=None, infer_backend='pt', stream=False):
    # Get model and template, and load LoRA weights.
    engine = PtEngine(model, adapters=[checkpoint])
    template = get_template(engine.model_meta.template, engine.processor) # , default_system=system
    # You can modify the `default_template` directly here, or pass it in during `engine.infer`.
    engine.default_template = template

    tool_names = extract_tool_names(tools_meta)
    analyzer = PythonProgramAnalyzer(tool_names)

    infer_func = infer_stream if stream else infer

    response_target_pattern = agent_prompt_meta["target_output"]["regex_extractors"]["prog_block"]["pattern"]
    summary = {}
    if query_list is not None:
        for query, i in enumerate(query_list):
            response = infer_func(engine, InferRequest(messages=[{'role': 'user', 'content': query}]))
            print('-' * 50)
            # summary[i] = {"system": system, "query": query, "response": response}
            summary[i] = {"query": query, "response": response}
    elif test_dataset is not None:
        test_dataset, _ = load_dataset(test_dataset, split_dataset_ratio=0.0, num_proc=1, seed=42)

        n = 0
        results_gt = []
        results_es = []
        for ex in test_dataset:
            messages = ex['messages']
            query, gt = extract_query_response_from_messages(messages)

            if query and gt:
                req = InferRequest(messages=[{'role': 'user', 'content': query}])
                response = infer_func(engine, req)

                try:
                    pattern = re.compile(response_target_pattern, flags=re.DOTALL)
                    matches = pattern.findall(response)
                    if not matches:
                        extracted = []
                    else:
                        if isinstance(matches[0], tuple):
                            extracted = [tuple(m) for m in matches]
                        else:
                            extracted = list(matches)
                    workflow_generated = extracted[0]
                    if workflow_generated:
                        analyzer_report = analyzer.analyze(workflow_generated)
                        results_es.append({
                            "prompt": query,
                            "response_workflow": workflow_generated,
                            "report": analyzer_report,
                            "response": response
                        })
                except:
                    results_es.append({
                        "prompt": query,
                        "response_workflow": gt,
                        "report": {},
                        "response": response
                    })
                analyzer_report_gt = analyzer.analyze(gt)
                results_gt.append({
                    "prompt": query,
                    "response_workflow": gt,
                    "report": analyzer_report_gt,
                })
            n += 1
        write_jsonl(results_es, os.path.join(output_path, "test_report_es.jsonl"))
        write_jsonl(results_gt, os.path.join(output_path, "test_report_gt.jsonl"))
        summary = compare_report_sets(results_gt, results_es)
    if output_path:
        with open(os.path.join(output_path, 'summary.json'), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # query_list = [
    #     'who are you?',
    #     "What should I do if I can't sleep at night?",
    #     '你是谁训练的？',
    # ]
    system = '''You are a helpful assistant that can analyze user's query and call correct tools to handle it.'''
    model_id = 'Qwen/Qwen3-8B-Base'
    output_dir = 'output/t1/'
    dataset_test = ['./dataset/v1_test.jsonl']
    agent_prompt_meta = read_yaml_file("prompt/agent_workflow_template_v3.yaml")
    tools_meta = load_json("dataset/tools/tools_v1.json")
    evaluate(model_id, 'output/t1/checkpoint-240', agent_prompt_meta, tools_meta, test_dataset=dataset_test)
    # summary = {}
    # test_dataset, _ = load_dataset(['./dataset/v0_test.jsonl'], split_dataset_ratio=0.0, num_proc=1, seed=42)
    # engine = PtEngine(model)
    # n = 0
    # for ex in test_dataset:
    #     messages = ex['messages']
    #     query, gt = extract_query_response_from_messages(messages)
    #
    #     if query and gt:
    #         req = InferRequest(messages=[{'role': 'user', 'content': query}])
    #         response = infer(engine, req)
    #         me = evaluate_metric_func(response, gt)
    #         summary[n] = {"system": system, "query": query, "gt": gt, "response": response, "metrics": me}
    #
    # if output_dir:
    #
    #     with open(os.path.join(output_dir, 'summary.json'), "w", encoding="utf-8") as f:
    #         json.dump(summary, f, ensure_ascii=False, indent=2)