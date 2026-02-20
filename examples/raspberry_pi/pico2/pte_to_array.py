# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Automated PTE to Array Injector for ExecuTorch
Reads a PTE file and either:
1. Updates model_pte[] array in a C file (when --file points to .c file)
2. Dumps hex array to text file (when --file points to .txt or is not provided)

Supports multiple array declaration patterns via regex map.
"""

import argparse
import os
import re
import sys


# Define regex patterns for different array declarations
# Each pattern should have 3 capture groups:
# 1. Declaration part (before opening brace)
# 2. Array contents (to be replaced)
# 3. Closing part (closing brace and semicolon)
ARRAY_PATTERNS = {
    "const_uint8_model_pte": r"(const\s+uint8_t\s+model_pte\[\].*?=\s*\{)(.*?)(\};)",
    "network_model_sec": r'(__attribute__\(\(section\("network_model_sec"\).*?\)\)\s+\w+\s+\w+\[\].*?=\s*\{)(.*?)(\};)',
    # Add more patterns here as needed
}


def generate_hex_array(pte_file_path):
    """Generate formatted hex array contents from PTE file."""
    with open(pte_file_path, "rb") as f:
        data = f.read()

    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = []
    bytes_per_line = 12  # Format with 12 bytes per line for readability

    for i in range(0, len(hex_bytes), bytes_per_line):
        line = ", ".join(hex_bytes[i : i + bytes_per_line])
        lines.append(f"    {line}")

    return ",\n".join(lines)


def dump_to_text_file(output_file, hex_array_contents):
    """Dump hex array contents to a text file."""
    with open(output_file, "w") as f:
        f.write(hex_array_contents)

    byte_count = len(hex_array_contents.split(","))
    print(f"Successfully wrote {byte_count} bytes to {output_file}")


def find_and_replace_array(content, hex_array_contents):
    """Try each regex pattern until one matches, then replace array contents."""
    for pattern_name, pattern in ARRAY_PATTERNS.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            print(f"Found array using pattern: '{pattern_name}'")
            # Replace the array contents (keeping the declaration and closing brace)
            new_content = (
                content[: match.start(2)]
                + f"\n{hex_array_contents}\n"
                + content[match.start(3) :]
            )
            return new_content, pattern_name
    # No pattern matched
    return None, None


def update_model_file(c_file_path, hex_array_contents):
    """Update the model array in the C file with new hex contents."""
    with open(c_file_path, "r") as f:
        content = f.read()

    new_content, matched_pattern = find_and_replace_array(content, hex_array_contents)
    if new_content is None:
        available_patterns = ", ".join(ARRAY_PATTERNS.keys())
        raise ValueError(
            f"Could not find any model array pattern in {c_file_path}.\n"
            f"Available patterns: {available_patterns}\n"
            "Add a new pattern to ARRAY_PATTERNS if needed."
        )

    with open(c_file_path, "w") as f:
        f.write(new_content)

    byte_count = len(hex_array_contents.split(","))
    print(
        f"Successfully updated {c_file_path} with {byte_count} bytes using pattern '{matched_pattern}'"
    )


def add_custom_pattern(name, pattern):
    """Add a custom regex pattern to the pattern map."""
    ARRAY_PATTERNS[name] = pattern
    print(f"Added custom pattern '{name}': {pattern}")


def list_patterns():
    """List all available regex patterns."""
    print("Available regex patterns:")
    for name, pattern in ARRAY_PATTERNS.items():
        print(f"  {name}: {pattern}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hex array from PTE file and either inject into C file or dump to text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Inject into C file (auto-detects array pattern):\n"
            "  python3 pte_to_array.py --model model.pte --file model_pte.c\n\n"
            "  # Dump to text file (default):\n"
            "  python3 pte_to_array.py --model model.pte\n"
            "  python3 pte_to_array.py --model model.pte --file my_array.txt\n\n"
            "  # List available patterns:\n"
            "  python3 pte_to_array.py --list-patterns\n\n"
            "  # Add custom pattern:\n"
            '  python3 pte_to_array.py --model model.pte --file model.c --add-pattern "my_pattern" "regex_here"\n'
        ),
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Path to the input PTE model file",
    )

    parser.add_argument(
        "--file",
        "-f",
        default="model_array.txt",
        help="Output file: .c file to inject array into, or .txt file to dump array (default: model_array.txt)",
    )

    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List all available regex patterns and exit",
    )

    parser.add_argument(
        "--add-pattern",
        nargs=2,
        metavar=("NAME", "REGEX"),
        help="Add a custom regex pattern: --add-pattern 'name' 'regex_pattern'",
    )

    args = parser.parse_args()

    # Handle list patterns command
    if args.list_patterns:
        list_patterns()
        sys.exit(0)

    # Add custom pattern if provided
    if args.add_pattern:
        add_custom_pattern(args.add_pattern[0], args.add_pattern[1])

    # Validate required arguments
    if not args.model:
        print("Error: --model argument is required", file=sys.stderr)
        sys.exit(1)

    # Validate input PTE file
    if not os.path.exists(args.model):
        print(f"Error: PTE file '{args.model}' does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        # Generate hex array from PTE file
        print(f"Reading PTE file: {args.model}")
        hex_array = generate_hex_array(args.model)

        # Determine action based on file extension
        file_ext = os.path.splitext(args.file)[1].lower()

        if file_ext == ".c":
            # Inject into C file
            if not os.path.exists(args.file):
                print(f"Error: C file '{args.file}' does not exist", file=sys.stderr)
                sys.exit(1)

            print(f"Injecting into C file: {args.file}")
            update_model_file(args.file, hex_array)

        else:
            # Dump to text file (default behavior)
            print(f"Dumping hex array to: {args.file}")
            dump_to_text_file(args.file, hex_array)

        sys.exit(0)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
