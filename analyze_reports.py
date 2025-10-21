#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/21
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : analyze_reports.py
# @Description :
from program_analyzer import PythonProgramAnalyzer
from statistic import compare_report_sets
from utils import read_jsonl, load_json, extract_tool_names
import re

def remove_code_fence(s: str) -> str:
    # 去掉开头的```Python或```python（可选换行），以及结尾的```
    return re.sub(r'(?is)^```python\s*\n?(.*?)\n?```$', r'\1', s.strip())


if __name__ == "__main__":
    p1 = "output/t3/test_report_es.jsonl"
    p2 = "output/t3/test_report_gt.jsonl"

    tools_meta = load_json("dataset/tools/tools_v1.json")
    tool_names = extract_tool_names(tools_meta)
    analyzer = PythonProgramAnalyzer(tool_names)

    reports_es = read_jsonl(p1)
    reports_gt = read_jsonl(p2)
    for report in reports_gt:
        workflow_clean = remove_code_fence(report["response_workflow"])
        analyzer_report_gt = analyzer.analyze(workflow_clean)
        report["report"] = analyzer_report_gt
    summary = compare_report_sets(reports_gt, reports_es)