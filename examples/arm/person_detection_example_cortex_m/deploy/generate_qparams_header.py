# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate target C++ constants for quantized model I/O."""

import argparse
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from utils.pte_metadata import (  # type: ignore[import-not-found]  # noqa: E402
    read_io_qparams,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pte", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qparams = read_io_qparams(args.pte)
    contents = f"""// Generated from {args.pte.name}; do not edit.
#pragma once

#include <cstdint>

namespace model_qparams {{
constexpr float kInputScale = {float(qparams['input0_scale']):.17g}f;
constexpr int32_t kInputZeroPoint = {int(qparams['input0_zp'])};
constexpr int32_t kInputMinimum = {int(qparams['input0_quant_min'])};
constexpr int32_t kInputMaximum = {int(qparams['input0_quant_max'])};
constexpr float kOutputScale = {float(qparams['output0_scale']):.17g}f;
constexpr int32_t kOutputZeroPoint = {int(qparams['output0_zp'])};
}} // namespace model_qparams
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(contents)


if __name__ == "__main__":
    main()
