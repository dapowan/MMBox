import copy
from typing import List, Dict, Any

from generate_workflow_from_query import extract_tool_names
from program_analyzer import PythonProgramAnalyzer
from statistic import evaluate_from_reports
from utils import read_jsonl, write_jsonl, load_json


def analyze_workflows_in_file(input_jsonl_path: str, output_jsonl_path: str, tools_meta) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file, analyzes the 'response_workflow' field in each object,
    adds the analysis to a 'report' field, and writes the updated objects
    to a new JSONL file.

    Args:
        input_jsonl_path (str): The path to the input JSONL file.
        output_jsonl_path (str): The path where the output JSONL file will be saved.

    Returns:
        List[Dict[str, Any]]: The list of updated dictionary objects.
    """
    print(f"Starting analysis of file: {input_jsonl_path}")
    tool_names = extract_tool_names(tools_meta)
    # 1. Initialize the analyzer
    analyzer = PythonProgramAnalyzer(tool_names)

    # 2. Read the source data
    records = read_jsonl(input_jsonl_path)
    if not records:
        print("No records found or file could not be read. Exiting.")
        return []

    updated_records = []

    # 3. Iterate over each object, analyze, and update
    for i, record in enumerate(records):
        workflow_string = record.get('response_workflow')

        report = record['report']
        if workflow_string and isinstance(workflow_string, str):
            # Analyze the workflow string
            report = analyzer.analyze(workflow_string)
            record['report'] = copy.deepcopy(report)
        updated_records.append(record)
        print(f"  - Processed record {i + 1} -> Validity: {report['validity']}, Number of errors: {len(report['errors'])}")

    # 4. Write the updated data to the output file
    write_jsonl(updated_records, output_jsonl_path)
    print(f"\nAnalysis complete. {len(updated_records)} records saved to: {output_jsonl_path}")

    # 5. Return the new list of objects
    return updated_records


if __name__ == "__main__":
    # --- Create a sample JSONL file for demonstration ---

    # Define file paths for the demo
    tools_meta = load_json("tools/tools_v1.json")
    version = "v2"
    input_file = f"qa_{version}/results_generate_workflow.jsonl"
    output_file = f"qa_{version}/results_generate_workflow_t.jsonl"

    results = analyze_workflows_in_file(input_file, output_file, tools_meta)
    report_summary = evaluate_from_reports(results)
    print(report_summary["error_summary"])