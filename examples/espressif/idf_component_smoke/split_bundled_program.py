# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split a bundled program into the plain .pte and an .io file for idf_component_smoke.

Writes <name>.pte and <name>.io next to <name>.bpte. The .io file holds the
forward test cases, which the device runs in order, little-endian: atol and
rtol (float32) and a uint32 case count, then per case the inputs and the
expected outputs, each as a uint32 count followed by uint32 byte size and raw
data per tensor, with data aligned to 16 bytes.
"""

import argparse
import struct
from pathlib import Path

from executorch.devtools.bundled_program.schema.bundled_program_schema import Tensor
from executorch.devtools.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
)

ALIGNMENT = 16


def tensors(blob, values):
    blob += struct.pack("<I", len(values))
    for value in values:
        if not isinstance(value.val, Tensor):
            raise SystemExit(
                f"only tensor inputs and outputs are supported, not {value.val}"
            )
        data = value.val.data
        blob += struct.pack("<I", len(data))
        blob += b"\0" * (-len(blob) % ALIGNMENT) + data
    return blob


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bpte", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="write the cases this many times; only for models without state",
    )
    args = parser.parse_args()

    bundle = deserialize_from_flatbuffer_to_bundled_program(args.bpte.read_bytes())
    (suite,) = [s for s in bundle.method_test_suites if s.method_name == "forward"]
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    cases = suite.test_cases * args.repeat
    if not cases:
        raise SystemExit(f"{args.bpte} has no forward test cases")
    blob = struct.pack("<ffI", args.atol, args.rtol, len(cases))
    for case in cases:
        blob = tensors(blob, case.inputs)
        blob = tensors(blob, case.expected_outputs)
    args.bpte.with_suffix(".pte").write_bytes(bundle.program)
    args.bpte.with_suffix(".io").write_bytes(blob)


if __name__ == "__main__":
    main()
