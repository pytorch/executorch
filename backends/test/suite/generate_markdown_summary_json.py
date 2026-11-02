import argparse
import json

from dataclasses import dataclass, field


@dataclass
class ResultCounts:
    """
    Represents aggregated result counts for each status.
    """

    total: int = 0
    passes: int = 0
    fails: int = 0
    expected_fails: int = 0
    skips: int = 0
    by_detail: dict[str, int] = field(default_factory=lambda: {})

    def add_row(
        self, result_value: str, result_detail: str, expected_fail: bool = False
    ) -> None:
        """
        Update the result counts for the specified row. A failure in a test that
        a flow lists in xfail_patterns counts as an expected failure.
        """

        self.total += 1

        if result_value == "Pass":
            self.passes += 1
        elif result_value == "Fail" and expected_fail:
            self.expected_fails += 1
            return
        elif result_value == "Fail":
            self.fails += 1
        elif result_value == "Skip":
            self.skips += 1
        else:
            raise RuntimeError(f"Unknown result value {result_value}")

        if result_detail:
            if result_detail not in self.by_detail:
                self.by_detail[result_detail] = 0

            self.by_detail[result_detail] += 1


@dataclass
class AggregatedSummary:
    """
    Represents aggegrated summary data for the test run.
    """

    counts: ResultCounts
    counts_by_params: dict[str, ResultCounts]
    failed_tests: list[list[str]]
    expected_failures: list[list[str]]
    unexpected_passes: list[str]


#
# A standalone script to generate a Markdown representation of a test report.
# This is primarily intended to be used with GitHub actions to generate a nice
# representation of the test results when looking at the action run.
#
# Usage: python executorch/backends/test/suite/generate_markdown_summary.py <path to test report CSV file>
# Markdown is written to stdout.
#


def aggregate_results(json_path: str) -> AggregatedSummary:
    with open(json_path) as f:
        data = json.load(f)

    # Count results and prepare data
    counts = ResultCounts()
    failed_tests = []
    counts_by_param = {}
    expected_failures = []
    unexpected_passes = []

    for test_data in data["tests"]:
        # The subtest rows hold the runner's own result. Only pytest's outcome
        # knows whether the test is in a flow's xfail_patterns.
        outcome = test_data.get("outcome")
        expected_fail = outcome == "xfailed"
        # Under pytest-xdist the first line names the worker, so check every line.
        longrepr = str((test_data.get("call") or {}).get("longrepr") or "")
        if outcome == "failed" and any(
            line.startswith("[XPASS(strict)]") for line in longrepr.splitlines()
        ):
            unexpected_passes.append(test_data["nodeid"].split("::")[-1])

        result_meta = test_data["metadata"]
        for subtest_meta in result_meta["subtests"]:
            result = subtest_meta["Result"]
            result_detail = subtest_meta.get("Result Detail") or ""

            counts.add_row(result, result_detail, expected_fail)

            test_id = subtest_meta["Test ID"]
            base_test = subtest_meta["Test Case"]
            params = test_id[len(base_test) + 1 : -1]

            if params:
                if params not in counts_by_param:
                    counts_by_param[params] = ResultCounts()
                counts_by_param[params].add_row(result, result_detail, expected_fail)

            if result.lower() == "fail":
                if expected_fail:
                    expected_failures.append(subtest_meta)
                else:
                    failed_tests.append(subtest_meta)

    return AggregatedSummary(
        counts=counts,
        failed_tests=failed_tests,
        counts_by_params=counts_by_param,
        expected_failures=expected_failures,
        unexpected_passes=unexpected_passes,
    )


def escape_for_markdown(text: str) -> str:
    """
    Modify a string to properly display in a markdown table cell.
    """
    if not text:
        return text

    # Replace newlines with <br /> tags
    escaped = text.replace("\n", "<br />")

    # Escape backslashes.
    escaped = escaped.replace("\\", "\\\\")

    # Escape pipe characters that would break table structure
    escaped = escaped.replace("|", "\\|")

    return escaped


def generate_markdown(json_path: str, exit_code: int = 0):  # noqa (C901)
    results = aggregate_results(json_path)

    # Generate Summary section
    print("# Summary\n")
    total_excluding_skips = (
        results.counts.passes + results.counts.fails + results.counts.expected_fails
    )
    pass_fraction = results.counts.passes / total_excluding_skips
    fail_fraction = results.counts.fails / total_excluding_skips
    print(
        f"- **Pass**: {results.counts.passes}/{total_excluding_skips} ({pass_fraction*100:.2f}%)"
    )
    print(
        f"- **Fail**: {results.counts.fails}/{total_excluding_skips} ({fail_fraction*100:.2f}%)"
    )
    expected_fail_fraction = results.counts.expected_fails / total_excluding_skips
    print(
        f"- **Expected Fail**: {results.counts.expected_fails}/{total_excluding_skips} ({expected_fail_fraction*100:.2f}%)"
    )
    print(f"- **Skip**: {results.counts.skips}")

    if results.counts_by_params:
        print("\n## Results by Parameters\n")

        if len(results.counts_by_params) > 0:
            # Create table header
            header_cols = ["Params", "Pass", "Fail", "Expected Fail", "Skip", "Pass %"]
            print("| " + " | ".join(header_cols) + " |")
            print("|" + "|".join(["---"] * len(header_cols)) + "|")

            # Create table rows
            for params_str, counts in results.counts_by_params.items():
                row_values = [params_str]

                # Add parameter values
                pass_fraction = counts.passes / (
                    counts.passes + counts.fails + counts.expected_fails
                )

                # Add count values
                row_values.extend(
                    [
                        str(counts.passes),
                        str(counts.fails),
                        str(counts.expected_fails),
                        str(counts.skips),
                        f"{pass_fraction*100:.2f}%",
                    ]
                )

                print("| " + " | ".join(row_values) + " |")

        print()

    print("## Failure Breakdown:")
    total_rows_with_result_detail = sum(results.counts.by_detail.values())
    for detail, count in sorted(results.counts.by_detail.items()):
        print(f"- **{detail}**: {count}/{total_rows_with_result_detail}")

    if results.unexpected_passes:
        print("# Unexpected Passes\n")
        print(
            "These tests are in a flow's xfail_patterns but passed. Remove them from the list."
        )
        for test_id in results.unexpected_passes:
            print(f"- {test_id}")
        print()

    # Generate Failed Tests section
    print("# Failed Tests\n")
    print(
        "To reproduce, run the following command from the root of the ExecuTorch repository:"
    )
    print("```")
    print('pytest -c /dev/nul backends/test/suite/ -k "<test_id>"')
    print("```")
    if results.failed_tests:
        print_table(results.failed_tests)
    else:
        print("No failed tests.\n")

    if results.expected_failures:
        # Shown with their details, so an expected failure that starts failing
        # in a different way is still visible.
        print("# Expected Failures\n")
        print_table(results.expected_failures)


def print_table(records) -> None:
    header = build_header(records)

    escaped_header = [escape_for_markdown(col) for col in header.keys()]
    print("| " + " | ".join(escaped_header) + " |")
    print("|" + "|".join(["---"] * len(escaped_header)) + "|")
    for rec in records:
        row = build_row(rec, header)
        print("| " + " | ".join(row) + " |")


def build_header(data) -> dict[str, int]:
    """
    Find the union of all keys and return a dict of header keys and indices. Try to preserve
    ordering as much as possible.
    """

    keys = max(data, key=len)

    header = {k: i for (i, k) in enumerate(keys)}

    for rec in data:
        keys = set(rec.keys())
        for k in keys:
            if k not in header:
                header[k] = len(header)

    return header


def build_row(rec, header: dict[str, int]) -> list[str]:
    row = [""] * len(header)
    for k, v in rec.items():
        row[header[k]] = escape_for_markdown(str(v))
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown representation of a test report."
    )
    parser.add_argument("json_path", help="Path to the test report CSV file.")
    parser.add_argument(
        "--exit-code", type=int, default=0, help="Exit code from the test process."
    )
    args = parser.parse_args()
    generate_markdown(args.json_path, args.exit_code)


if __name__ == "__main__":
    main()
