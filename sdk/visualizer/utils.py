from typing import Any, List


def make_markdown_table(table: List[List[Any]]) -> str:
    if table is None or len(table) <= 0:
        # Empty table
        return ""

    # Add table header row
    header_md = "|"
    for col_name in table[0]:
        header_md += " " + col_name + " |"
    header_md += "\n|"

    # Add the divider line
    for _ in range(len(table[0])):
        header_md += " ------- |"

    if len(table) <= 1:
        # Table with just the header row
        return header_md

    # Add the table data
    running_md = header_md
    for row in table[1:]:
        running_md += "\n|"
        for col in row:
            running_md += " " + str(col) + " |"

    return running_md
