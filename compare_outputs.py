#!/usr/bin/env python3
"""
Comparison script to calculate max absolute tolerance (atol) and max relative tolerance (rtol)
between runtime outputs and label outputs.
"""

import os
import sys

import numpy as np


def read_csv_file(filepath):
    """Read a comma-separated values file and return as numpy array."""
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: {filepath} is empty")
                return np.array([])

            # Split by comma and convert to float
            values = [float(x.strip()) for x in content.split(",") if x.strip()]
            return np.array(values)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        return None
    except ValueError as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def calculate_tolerances(runtime_outputs, label_outputs):
    """Calculate max absolute and relative tolerances."""
    if runtime_outputs is None or label_outputs is None:
        return None, None

    if len(runtime_outputs) == 0 or len(label_outputs) == 0:
        print("Warning: One of the output arrays is empty")
        return None, None

    if len(runtime_outputs) != len(label_outputs):
        print(
            f"Warning: Array lengths don't match: runtime={len(runtime_outputs)}, label={len(label_outputs)}"
        )
        # Pad shorter array with zeros or truncate longer array
        min_len = min(len(runtime_outputs), len(label_outputs))
        runtime_outputs = runtime_outputs[:min_len]
        label_outputs = label_outputs[:min_len]

    # Calculate absolute differences
    abs_diff = np.abs(runtime_outputs - label_outputs)
    max_atol = np.max(abs_diff)

    # Calculate relative differences (avoid division by zero)
    # rel_diff = |a - b| / max(|a|, |b|, eps) where eps is a small number
    eps = 1e-8
    denominator = np.maximum(
        np.maximum(np.abs(runtime_outputs), np.abs(label_outputs)), eps
    )
    rel_diff = abs_diff / denominator
    max_rtol = np.max(rel_diff)

    return max_atol, max_rtol


def main():
    """Main function to compare outputs and print tolerances."""
    # File paths
    runtime_file = "aoti_debug_data/final_runtime_output.txt"
    label_file = "aoti_debug_data/label_output.txt"

    print("=" * 60)
    print("AOTI Runtime vs Label Output Comparison")
    print("=" * 60)

    # Check if files exist
    if not os.path.exists(runtime_file):
        print(f"Error: {runtime_file} not found")
        sys.exit(1)

    if not os.path.exists(label_file):
        print(f"Error: {label_file} not found")
        sys.exit(1)

    # Read the files
    print(f"Reading runtime outputs from: {runtime_file}")
    runtime_outputs = read_csv_file(runtime_file)

    print(f"Reading label outputs from: {label_file}")
    label_outputs = read_csv_file(label_file)

    if runtime_outputs is None or label_outputs is None:
        print("Failed to read one or both files")
        sys.exit(1)

    print(f"Runtime outputs shape: {runtime_outputs.shape}")
    print(f"Label outputs shape: {label_outputs.shape}")

    if runtime_outputs.shape != label_outputs.shape:
        print("Error: Output shapes don't match")
        sys.exit(1)

    # Calculate tolerances
    max_atol, max_rtol = calculate_tolerances(runtime_outputs, label_outputs)

    if max_atol is None or max_rtol is None:
        print("Failed to calculate tolerances")
        sys.exit(1)

    # Print results
    print("-" * 60)
    print("COMPARISON RESULTS:")
    print(f"Max Absolute Tolerance (atol): {max_atol:.10f}")
    print(f"Max Relative Tolerance (rtol): {max_rtol:.10f}")
    print("-" * 60)

    # Print some statistics
    print("ADDITIONAL STATISTICS:")
    print(f"Total elements compared: {len(runtime_outputs)}")
    print(
        f"Runtime output range: [{np.min(runtime_outputs):.6f}, {np.max(runtime_outputs):.6f}]"
    )
    print(
        f"Label output range: [{np.min(label_outputs):.6f}, {np.max(label_outputs):.6f}]"
    )

    # Calculate mean absolute difference
    abs_diff = np.abs(runtime_outputs - label_outputs)
    mean_atol = np.mean(abs_diff)
    print(f"Mean Absolute Tolerance: {mean_atol:.10f}")

    # Check if outputs are close within common tolerances
    is_close_1e5 = np.allclose(
        runtime_outputs,
        label_outputs,
        atol=1e-5,
        rtol=1e-5,
    )
    is_close_1e6 = np.allclose(
        runtime_outputs,
        label_outputs,
        atol=1e-6,
        rtol=1e-6,
    )

    print(f"Close within atol=1e-5, rtol=1e-5: {is_close_1e5}")
    print(f"Close within atol=1e-6, rtol=1e-6: {is_close_1e6}")

    print("=" * 60)


if __name__ == "__main__":
    main()
