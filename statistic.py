from collections import Counter, defaultdict
from typing import List, Dict, Any

from utils import read_jsonl


def evaluate_from_reports(data: List[Dict[str, Any]], print_info=True) -> Dict[str, Any]:
    """
    分析一组包含 'report' 字段的 JSON 对象列表，统计各类指标。
    增加了 error_ratio: 含有错误的 report 占比。
    """
    if not data:
        return {"count": 0, "message": "Empty data"}

    total = len(data)
    valid_count = 0
    error_report_count = 0  # 新增：含有错误的报告数
    tool_seq_lens = []
    vars_all_counts = []
    vars_invalid_counts = []
    error_counter: Counter = Counter()
    error_positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _normalize_error_key(err_item: Any) -> str:
        """归一化错误项为字符串 key"""
        if isinstance(err_item, dict):
            etype = str(err_item.get("type", "Error"))
            emsg = str(err_item.get("message", "")) if err_item.get("message") is not None else ""
            return f"{etype}: {emsg}" if emsg else etype
        elif isinstance(err_item, str):
            return f"Error: {err_item}"
        else:
            return f"Error: {repr(err_item)}"

    for idx, item in enumerate(data):
        report = item.get("report", {})
        if not isinstance(report, dict):
            continue

        # validity
        if report.get("validity", False):
            valid_count += 1

        # tool_sequence length
        tools = report.get("tool_sequence", [])
        if isinstance(tools, list):
            tool_seq_lens.append(len(tools))

        # variables
        vars_info = report.get("variables", {})
        all_vars = vars_info.get("all", [])
        invalid_vars = vars_info.get("invalid", [])
        if isinstance(all_vars, list):
            vars_all_counts.append(len(all_vars))
        if isinstance(invalid_vars, list):
            vars_invalid_counts.append(len(invalid_vars))

        # errors（兼容 dict / str）
        errs = report.get("errors", [])
        if isinstance(errs, list) and errs:
            error_report_count += 1  # 新增统计：有错误的报告
            for e in errs:
                key = _normalize_error_key(e)
                error_counter.update([key])
                pos = {"report_index": idx}
                if isinstance(e, dict):
                    if "lineno" in e:
                        pos["lineno"] = e.get("lineno")
                    if "col_offset" in e:
                        pos["col_offset"] = e.get("col_offset")
                error_positions[key].append(pos)

    # aggregate stats
    avg_tool_seq_len = sum(tool_seq_lens) / len(tool_seq_lens) if tool_seq_lens else 0
    avg_vars_all = sum(vars_all_counts) / len(vars_all_counts) if vars_all_counts else 0
    avg_vars_invalid = sum(vars_invalid_counts) / len(vars_invalid_counts) if vars_invalid_counts else 0
    avg_invalid_ratio = (
        (sum(vars_invalid_counts) / sum(vars_all_counts))
        if vars_all_counts and sum(vars_all_counts) > 0
        else 0
    )

    # build detailed error summary
    error_summary = {
        key: {"count": count, "positions": error_positions[key]}
        for key, count in error_counter.items()
    }

    # build summary
    summary = {
        "total_reports": total,
        "valid_reports": valid_count,
        "valid_ratio": valid_count / total if total > 0 else 0,
        "error_reports": error_report_count,
        "error_ratio": error_report_count / total if total > 0 else 0,
        "avg_tool_sequence_length": avg_tool_seq_len,
        "avg_variables_total": avg_vars_all,
        "avg_variables_invalid": avg_vars_invalid,
        "avg_invalid_ratio": avg_invalid_ratio,
    }

    if print_info:
        print("\n=== Report Evaluation Summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k:30s}: {v:.4f}")
            else:
                print(f"{k:30s}: {v}")
        print("=================================\n")

    return {**summary, "error_summary": error_summary}


def compare_report_sets(gt_reports: List[Dict[str, Any]], est_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对比两组 reports（长度相同、逐条对应）：
    - 复用 evaluate_from_reports 打印并返回两组 summary
    - 比较每条 report 的 tool_sequence：
        * 是否完全一致
        * EST 比 GT 多/少了多少个 tool（基于长度差）
    - 打印对比指标，并返回整体与逐条结果
    """
    if len(gt_reports) != len(est_reports):
        raise ValueError(f"gt_reports ({len(gt_reports)}) and est_reports ({len(est_reports)}) must have the same length.")

    def _normalize_item(item: Any) -> Dict[str, Any]:
        """
        兼容输入既可能是 {'report': {...}}，也可能直接是 report dict。
        统一包装为 {'report': <dict>}；若不是 dict，则包装空 dict。
        """
        if isinstance(item, dict) and "report" in item:
            return item
        elif isinstance(item, dict):
            return {"report": item}
        else:
            return {"report": {}}

    # 统一结构，便于 evaluate_from_reports 与对比逻辑复用
    gt_norm = [_normalize_item(x) for x in gt_reports]
    est_norm = [_normalize_item(x) for x in est_reports]

    # 1) 打印并获取两个 summary（evaluate_from_reports 内部已负责打印）
    gt_summary = evaluate_from_reports(gt_norm)
    est_summary = evaluate_from_reports(est_norm)

    # 2) 逐对比较 tool_sequence
    per_pair = []
    exact_match_count = 0
    extra_sum = 0     # EST 相对 GT 多出来的个数（仅正差计入）
    missing_sum = 0   # EST 相对 GT 缺少的个数（仅正差计入）

    def _get_tools(report_item: Dict[str, Any]) -> List[Any]:
        r = report_item.get("report", {})
        tools = r.get("tool_sequence", [])
        return tools if isinstance(tools, list) else []

    for idx, (g_item, e_item) in enumerate(zip(gt_norm, est_norm)):
        g_tools = _get_tools(g_item)
        e_tools = _get_tools(e_item)

        exact_match = (g_tools == e_tools)
        if exact_match:
            exact_match_count += 1

        # 以“长度差”衡量多/少的数量（不做集合去重，保持顺序敏感的长度定义）
        diff = len(e_tools) - len(g_tools)
        extra = diff if diff > 0 else 0
        missing = (-diff) if diff < 0 else 0

        extra_sum += extra
        missing_sum += missing

        per_pair.append({
            "index": idx,
            "gt_len": len(g_tools),
            "est_len": len(e_tools),
            "exact_match": exact_match,
            "extra_tools": extra,      # EST 相对 GT 多出的数量
            "missing_tools": missing,  # EST 相对 GT 缺少的数量
        })

    n = len(gt_norm) if gt_norm else 0
    exact_match_ratio = (exact_match_count / n) if n > 0 else 0.0
    avg_extra = (extra_sum / n) if n > 0 else 0.0
    avg_missing = (missing_sum / n) if n > 0 else 0.0

    # 3) 单独打印对比指标
    print("\n=== Tool Sequence Comparison (GT vs EST) ===")
    print(f"{'pairs':30s}: {n}")
    print(f"{'exact_match_count':30s}: {exact_match_count}")
    print(f"{'exact_match_ratio':30s}: {exact_match_ratio:.4f}")
    print(f"{'avg_extra_tools (EST-GT)':30s}: {avg_extra:.4f}")
    print(f"{'avg_missing_tools (GT-EST)':30s}: {avg_missing:.4f}")
    print("============================================\n")

    # 4) 返回汇总
    return {
        "gt_summary": gt_summary,
        "est_summary": est_summary,
        "comparison": {
            "pairs": n,
            "exact_match_count": exact_match_count,
            "exact_match_ratio": exact_match_ratio,
            "avg_extra_tools": avg_extra,
            "avg_missing_tools": avg_missing,
            "per_pair": per_pair,
        }
    }

if __name__ == "__main__":
    input_file1 = f"dataset/qa_v3/results_generate_workflow.jsonl"
    input_file2 = f"dataset/qa_gpt-oss_v1/results_generate_workflow.jsonl"

    results1 = read_jsonl(input_file1)
    results2 = read_jsonl(input_file2)
    report_summary = compare_report_sets(results1, results2)
