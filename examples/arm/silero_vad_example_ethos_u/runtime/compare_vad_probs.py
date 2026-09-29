# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import re
import struct
from pathlib import Path


def read_float32_file(path: str | Path) -> tuple[float, ...]:
    with open(path, "rb") as input_file:
        data = input_file.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: size is not a multiple of float32")
    return struct.unpack(f"<{len(data) // 4}f", data)


def read_probability_log(path: str | Path) -> tuple[float, ...]:
    probabilities = []
    prob_pattern = re.compile(
        r"\bPROB\s+[-+]?\d+(?:\.\d+)?\s+"
        r"([-+]?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf))\s+"
        r"(?:speech|silence)\b",
        re.IGNORECASE,
    )
    with open(path) as input_file:
        for line in input_file:
            match = prob_pattern.search(line)
            if match:
                probabilities.append(float(match.group(1)))
    if not probabilities:
        raise ValueError(f"{path}: no PROB lines found")
    return tuple(probabilities)


def compare(
    expected_path: str | Path,
    actual_path: str | Path | None,
    actual_log_path: str | Path | None,
    threshold: float,
    atol: float,
    mean_atol: float,
    max_threshold_mismatches: int,
) -> None:
    expected = read_float32_file(expected_path)
    if actual_log_path is not None:
        actual = read_probability_log(actual_log_path)
    elif actual_path is not None:
        actual = read_float32_file(actual_path)
    else:
        raise ValueError("Either actual_path or actual_log_path is required")

    if len(expected) != len(actual):
        raise ValueError(
            f"Length mismatch: expected {len(expected)} values, got {len(actual)}"
        )
    if not expected:
        raise ValueError("No probability values found to compare")

    max_abs_error = 0.0
    total_abs_error = 0.0
    threshold_mismatches = 0
    worst_index = 0
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            raise AssertionError(
                f"Non-finite probability at frame {index}: "
                f"expected={expected_value}, actual={actual_value}"
            )
        abs_error = abs(expected_value - actual_value)
        total_abs_error += abs_error
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            worst_index = index
        if (expected_value > threshold) != (actual_value > threshold):
            threshold_mismatches += 1

    mean_abs_error = total_abs_error / len(expected)

    print(f"Compared {len(expected)} probability values")
    print(f"Max abs error: {max_abs_error:.6f} at frame {worst_index}")
    print(f"Mean abs error: {mean_abs_error:.6f}")
    print(f"Threshold mismatches at {threshold:.3f}: {threshold_mismatches}")

    if max_abs_error > atol:
        raise AssertionError(
            f"Max abs error {max_abs_error:.6f} exceeds tolerance {atol:.6f}"
        )
    if mean_abs_error > mean_atol:
        raise AssertionError(
            f"Mean abs error {mean_abs_error:.6f} exceeds tolerance " f"{mean_atol:.6f}"
        )
    if threshold_mismatches > max_threshold_mismatches:
        raise AssertionError(
            f"{threshold_mismatches} threshold decisions differed from expected; "
            f"maximum allowed is {max_threshold_mismatches}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, help="Path to expected_probs.bin")
    actual_group = parser.add_mutually_exclusive_group(required=True)
    actual_group.add_argument("--actual", help="Path to vad_probs.bin")
    actual_group.add_argument(
        "--actual-log",
        help="Path to FVP serial log containing PROB lines",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold used by the runtime",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.05,
        help="Maximum allowed absolute probability error",
    )
    parser.add_argument(
        "--mean-atol",
        type=float,
        default=0.01,
        help="Maximum allowed mean absolute probability error",
    )
    parser.add_argument(
        "--max-threshold-mismatches",
        type=int,
        default=0,
        help="Maximum allowed speech/silence decision mismatches",
    )
    args = parser.parse_args()

    compare(
        args.expected,
        args.actual,
        args.actual_log,
        args.threshold,
        args.atol,
        args.mean_atol,
        args.max_threshold_mismatches,
    )
