"""
Parse test report JSON produced by pytest --json-report and generate
v3 format benchmark results for upload to PyTorch HUD.

Metrics produced per suite:
  - pass_rate(%)   : percentage of passing tests (skips excluded from denominator)
  - total_pass     : number of passing tests
  - total_fail     : number of failing tests
  - total_skip     : number of skipped tests
"""

import argparse
import json
import sys


def parse_test_report(json_path: str) -> dict:
    """
    Parse a test report JSON file and return pass/fail/skip counts.

    The JSON is produced by test_backend.sh via pytest --json-report and has the
    structure used by generate_markdown_summary_json.py:
      { "tests": [ { "metadata": { "subtests": [ { "Result": "Pass"|"Fail"|"Skip", ... } ] } } ] }
    """
    with open(json_path) as f:
        data = json.load(f)

    passes = 0
    fails = 0
    skips = 0

    for test_data in data["tests"]:
        for subtest in test_data["metadata"]["subtests"]:
            result = subtest["Result"]
            if result == "Pass":
                passes += 1
            elif result == "Fail":
                fails += 1
            elif result == "Skip":
                skips += 1

    return {"passes": passes, "fails": fails, "skips": skips}


def build_v3_record(
    metric_name: str,
    value: float,
    suite: str,
    flow: str,
    git_sha: str,
    workflow_run_id: str,
    workflow_run_url: str,
    runner_name: str,
) -> dict:
    """Build a single v3 format benchmark record."""
    return {
        "benchmark": {
            "name": "ExecuTorch",
            "mode": "test",
            "extra_info": {
                "backend": "cuda",
                "suite": suite,
                "flow": flow,
                "git_sha": git_sha,
                "workflow_run_id": workflow_run_id,
                "workflow_run_url": workflow_run_url,
            },
        },
        "model": {
            "name": f"cuda_backend_tests_{suite}",
            "type": "OSS backend test",
            "backend": "cuda",
        },
        "metric": {
            "name": metric_name,
            "benchmark_values": [value],
            "target_value": 0,
            "extra_info": {},
        },
        "runners": [{"name": runner_name, "type": "linux"}],
    }


def generate_v3_records(
    counts: dict,
    suite: str,
    flow: str,
    git_sha: str,
    workflow_run_id: str,
    workflow_run_url: str,
    runner_name: str,
) -> list:
    """Generate v3 format records for all metrics."""
    total_excluding_skips = counts["passes"] + counts["fails"]
    pass_rate = (
        (counts["passes"] / total_excluding_skips * 100)
        if total_excluding_skips > 0
        else 0.0
    )

    common = dict(
        suite=suite,
        flow=flow,
        git_sha=git_sha,
        workflow_run_id=workflow_run_id,
        workflow_run_url=workflow_run_url,
        runner_name=runner_name,
    )

    return [
        build_v3_record("pass_rate(%)", pass_rate, **common),
        build_v3_record("total_pass", counts["passes"], **common),
        build_v3_record("total_fail", counts["fails"], **common),
        build_v3_record("total_skip", counts["skips"], **common),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate v3 format benchmark results from test report JSON"
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to the test report JSON file",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Test suite name (e.g. models, operators)",
    )
    parser.add_argument(
        "--flow",
        required=True,
        help="Test flow name (e.g. cuda)",
    )
    parser.add_argument(
        "--git-sha",
        required=True,
        help="Git commit SHA",
    )
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="GitHub workflow run ID",
    )
    parser.add_argument(
        "--workflow-run-url",
        default="",
        help="GitHub workflow run URL",
    )
    parser.add_argument(
        "--runner-name",
        default="linux.g5.4xlarge.nvidia.gpu",
        help="CI runner name",
    )
    parser.add_argument(
        "--output-v3",
        required=True,
        help="Path to write v3 format JSON output",
    )
    args = parser.parse_args()

    counts = parse_test_report(args.report_json)

    total_excluding_skips = counts["passes"] + counts["fails"]
    pass_rate = (
        (counts["passes"] / total_excluding_skips * 100)
        if total_excluding_skips > 0
        else 0.0
    )

    print(f"Suite: {args.suite}")
    print(
        f"  Pass: {counts['passes']}, Fail: {counts['fails']}, Skip: {counts['skips']}"
    )
    print(f"  Pass rate: {pass_rate:.2f}%")

    records = generate_v3_records(
        counts=counts,
        suite=args.suite,
        flow=args.flow,
        git_sha=args.git_sha,
        workflow_run_id=args.workflow_run_id,
        workflow_run_url=args.workflow_run_url,
        runner_name=args.runner_name,
    )

    with open(args.output_v3, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} v3 records to {args.output_v3}")


if __name__ == "__main__":
    main()
