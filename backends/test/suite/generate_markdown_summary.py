import argparse
import csv
import sys

#
# A standalone script to generate a Markdown representation of a test report.
# This is primarily intended to be used with GitHub actions to generate a nice
# representation of the test results when looking at the action run.
#
# Usage: python executorch/backends/test/suite/generate_markdown_summary.py <path to test report CSV file>
# Markdown is written to stdout.
#


def escape_for_markdown(text: str) -> str:
    """
    Modify a string to properly display in a markdown table cell.
    """
    if not text:
        return text
    
    # Replace newlines with <br /> tags
    escaped = text.replace('\n', '<br />')

    # Escape backslashes.
    escaped = escaped.replace('\\', '\\\\')
    
    # Escape pipe characters that would break table structure
    escaped = escaped.replace('|', '\\|')
    
    return escaped


def generate_markdown(csv_path: str, exit_code: int = 0):  # noqa (C901)
    # Print warning if exit code is non-zero
    if exit_code != 0:
        print("> [!WARNING]")
        print(
            f"> Exit code {exit_code} was non-zero. Test process may have crashed. Check the job logs for more information.\n"
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]

    # Find the Result and Result Detail column indices
    result_column_index = None
    result_detail_column_index = None
    for i, col in enumerate(header):
        if col.lower() == "result":
            result_column_index = i
        elif col.lower() == "result detail":
            result_detail_column_index = i

    # Count results and prepare data
    pass_count = 0
    fail_count = 0
    skip_count = 0
    failed_tests = []
    processed_rows = []
    result_detail_counts = {}

    for row in data_rows:
        # Make a copy of the row to avoid modifying the original
        processed_row = [escape_for_markdown(cell) for cell in row]

        # Count results and collect failed tests
        if result_column_index is not None and result_column_index < len(row):
            result_value = row[result_column_index].strip().lower()
            if result_value == "pass":
                pass_count += 1
                processed_row[result_column_index] = (
                    '<span style="color:green">Pass</span>'
                )
            elif result_value == "fail":
                fail_count += 1
                processed_row[result_column_index] = (
                    '<span style="color:red">Fail</span>'
                )
                failed_tests.append(processed_row.copy())
            elif result_value == "skip":
                skip_count += 1
                processed_row[result_column_index] = (
                    '<span style="color:gray">Skip</span>'
                )

        # Count result details (excluding empty ones)
        if result_detail_column_index is not None and result_detail_column_index < len(
            row
        ):
            result_detail_value = row[result_detail_column_index].strip()
            if result_detail_value:  # Only count non-empty result details
                if result_detail_value in result_detail_counts:
                    result_detail_counts[result_detail_value] += 1
                else:
                    result_detail_counts[result_detail_value] = 1

        processed_rows.append(processed_row)

    # Generate Summary section
    total_rows = len(data_rows)
    print("# Summary\n")
    print(f"- **Pass**: {pass_count}/{total_rows}")
    print(f"- **Fail**: {fail_count}/{total_rows}")
    print(f"- **Skip**: {skip_count}/{total_rows}")

    print("## Failure Breakdown:")
    total_rows_with_result_detail = sum(result_detail_counts.values())
    for detail, count in sorted(result_detail_counts.items()):
        print(f"- **{detail}**: {count}/{total_rows_with_result_detail}")

    # Generate Failed Tests section
    print("# Failed Tests\n")
    if failed_tests:
        escaped_header = [escape_for_markdown(col) for col in header]
        print("| " + " | ".join(escaped_header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in failed_tests:
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
