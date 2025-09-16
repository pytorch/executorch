import argparse
import csv
import json
import sys

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
    header: list[str]


#
# A standalone script to generate a Markdown representation of a test report.
# This is primarily intended to be used with GitHub actions to generate a nice
# representation of the test results when looking at the action run.
#
# Usage: python executorch/backends/test/suite/generate_markdown_summary.py <path to test report CSV file>
# Markdown is written to stdout.
#


def aggregate_results(csv_path: str) -> AggregatedSummary:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]

    header_indices_by_name = {n.lower(): i for (i, n) in enumerate(header)}
    params_column_index = header_indices_by_name.get("params", None)
    result_column_index = header_indices_by_name["result"]
    result_detail_column_index = header_indices_by_name["result detail"]

    # Count results and prepare data
    counts = ResultCounts()
    failed_tests = []
    counts_by_param = {}

    for row in data_rows:
        result = row[result_column_index]
        result_detail = row[result_detail_column_index]

        counts.add_row(result, result_detail)

        params = row[params_column_index] if params_column_index else None
        if params:
            if params not in counts_by_param:
                counts_by_param[params] = ResultCounts()
            counts_by_param[params].add_row(result, result_detail)

        # Make a copy of the row to avoid modifying the original
        processed_row = [escape_for_markdown(cell) for cell in row]

        # Count results and collect failed tests
        if result_column_index is not None and result_column_index < len(row):
            result_value = row[result_column_index].strip().lower()
            if result_value == "pass":
                processed_row[result_column_index] = (
                    '<span style="color:green">Pass</span>'
                )
            elif result_value == "fail":
                processed_row[result_column_index] = (
                    '<span style="color:red">Fail</span>'
                )
                failed_tests.append(processed_row.copy())
            elif result_value == "skip":
                processed_row[result_column_index] = (
                    '<span style="color:gray">Skip</span>'
                )

    return AggregatedSummary(
        counts=counts,
        failed_tests=failed_tests,
        counts_by_params=counts_by_param,
        header=header,
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


def generate_markdown(csv_path: str, exit_code: int = 0):  # noqa (C901)
    # Print warning if exit code is non-zero
    if exit_code != 0:
        print("> [!WARNING]")
        print(
            f"> Exit code {exit_code} was non-zero. Test process may have crashed. Check the job logs for more information.\n"
        )

    results = aggregate_results(csv_path)

    # Generate Summary section
    total_rows = results.counts.total
    print("# Summary\n")
    print(f"- **Pass**: {results.counts.passes}/{total_rows}")
    print(f"- **Fail**: {results.counts.fails}/{total_rows}")
    print(f"- **Skip**: {results.counts.skips}/{total_rows}")

    if results.counts_by_params:
        print("\n## Results by Parameters\n")

        # Extract all unique parameter keys from the JSON strings
        all_param_keys = set()
        parsed_params = {}

        for params_str in results.counts_by_params.keys():
            # Parse the JSON string (it's a string representation of a dict)
            params_dict = json.loads(params_str)
            parsed_params[params_str] = params_dict
            all_param_keys.update(params_dict.keys())

        if parsed_params:
            # Sort parameter keys for consistent column ordering
            sorted_param_keys = sorted(all_param_keys)

            # Create table header
            header_cols = sorted_param_keys + ["Pass", "Fail", "Skip", "Pass %"]
            print("| " + " | ".join(header_cols) + " |")
            print("|" + "|".join(["---"] * len(header_cols)) + "|")

            # Create table rows
            for params_str, counts in results.counts_by_params.items():
                if params_str in parsed_params:
                    params_dict = parsed_params[params_str]
                    row_values = []

                    # Add parameter values
                    for key in sorted_param_keys:
                        value = params_dict.get(key, "")
                        row_values.append(str(value))

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
    if results.failed_tests:
        escaped_header = [escape_for_markdown(col) for col in results.header]
        print("| " + " | ".join(escaped_header) + " |")
        print("|" + "|".join(["---"] * len(results.header)) + "|")
        for row in results.failed_tests:
            print("| " + " | ".join(row) + " |")
    else:
        print("No failed tests.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown representation of a test report."
    )
    parser.add_argument("csv_path", help="Path to the test report CSV file.")
    parser.add_argument(
        "--exit-code", type=int, default=0, help="Exit code from the test process."
    )
    args = parser.parse_args()
    try:
        generate_markdown(args.csv_path, args.exit_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
