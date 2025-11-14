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
    skips: int = 0
    by_detail: dict[str, int] = field(default_factory=lambda: {})

    def add_row(self, result_value: str, result_detail: str) -> None:
        """
        Update the result counts for the specified row.
        """

        self.total += 1

        if result_value == "Pass":
            self.passes += 1
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

    for test_data in data["tests"]:
        result_meta = test_data["metadata"]
        for subtest_meta in result_meta["subtests"]:
            result = subtest_meta["Result"]
            result_detail = subtest_meta.get("Result Detail") or ""

            counts.add_row(result, result_detail)

            test_id = subtest_meta["Test ID"]
            base_test = subtest_meta["Test Case"]
            params = test_id[len(base_test) + 1 : -1]

            if params:
                if params not in counts_by_param:
                    counts_by_param[params] = ResultCounts()
                counts_by_param[params].add_row(result, result_detail)

            if result.lower() == "fail":
                failed_tests.append(subtest_meta)

    return AggregatedSummary(
        counts=counts,
        failed_tests=failed_tests,
        counts_by_params=counts_by_param,
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
    total_excluding_skips = results.counts.passes + results.counts.fails
    pass_fraction = results.counts.passes / total_excluding_skips
    fail_fraction = results.counts.fails / total_excluding_skips
    print(
        f"- **Pass**: {results.counts.passes}/{total_excluding_skips} ({pass_fraction*100:.2f}%)"
    )
    print(
        f"- **Fail**: {results.counts.fails}/{total_excluding_skips} ({fail_fraction*100:.2f}%)"
    )
    print(f"- **Skip**: {results.counts.skips}")

    if results.counts_by_params:
        print("\n## Results by Parameters\n")

        if len(results.counts_by_params) > 0:
            # Create table header
            header_cols = ["Params", "Pass", "Fail", "Skip", "Pass %"]
            print("| " + " | ".join(header_cols) + " |")
            print("|" + "|".join(["---"] * len(header_cols)) + "|")

            # Create table rows
            for params_str, counts in results.counts_by_params.items():
                row_values = [params_str]

                # Add parameter values
                pass_fraction = counts.passes / (counts.passes + counts.fails)

                # Add count values
                row_values.extend(
                    [
                        str(counts.passes),
                        str(counts.fails),
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

    # Generate Failed Tests section
    print("# Failed Tests\n")
    print(
        "To reproduce, run the following command from the root of the ExecuTorch repository:"
    )
    print("```")
    print('pytest -c /dev/nul backends/test/suite/ -k "<test_id>"')
    print("```")
    if results.failed_tests:
        header = build_header(results.failed_tests)

        escaped_header = [escape_for_markdown(col) for col in header.keys()]
        print("| " + " | ".join(escaped_header) + " |")
        print("|" + "|".join(["---"] * len(escaped_header)) + "|")
        for rec in results.failed_tests:
            row = build_row(rec, header)
            print("| " + " | ".join(row) + " |")
    else:
        print("No failed tests.\n")


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
