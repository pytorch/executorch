# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import struct
import subprocess
import zipfile
from pathlib import Path


def make_graph() -> dict[str, object]:
    meta = {
        "dtype": "FLOAT",
        "sizes": [{"min": 1, "max": 1}, {"min": 6, "max": 6}],
        "dim_order": [0, 1],
    }
    tensor = {"value_type": "TensorArg", "value": {"name": "x"}}
    return {
        "nodes": [
            {"name": "x", "op_kind": "PLACEHOLDER", "outputs": [{"name": "x"}]},
            {
                "name": "add",
                "op_kind": "CALL_FUNCTION",
                "target": "torch.ops.aten.add.Tensor",
                "inputs": [
                    {"name": "self", "arg": tensor},
                    {"name": "other", "arg": tensor},
                    {
                        "name": "alpha",
                        "arg": {"value_type": "IntArg", "value": {"value": 1}},
                    },
                ],
                "outputs": [{"name": "add"}],
            },
            {
                "name": "output",
                "op_kind": "OUTPUT",
                "inputs": [
                    {"arg": {"value_type": "TensorArg", "value": {"name": "add"}}}
                ],
            },
        ],
        "inputs": ["x"],
        "outputs": ["add"],
        "tensor_values": [{"name": name, "meta": meta} for name in ("x", "add")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Native SDK add fixture")
    parser.add_argument("--flatc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = Path(__file__).resolve().parents[2] / "serialization/native_graph.fbs"
    program = {
        "version": "1",
        "methods": [
            {
                "name": "forward",
                "graph": make_graph(),
                "output_specs": [{"name": "add", "kind": "USER_OUTPUT"}],
            }
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "program.json"
    source.write_text(json.dumps(program))
    subprocess.run(
        [args.flatc, "-b", "-o", str(args.output), str(schema), str(source)],
        check=True,
    )
    with zipfile.ZipFile(args.output / "add.ptn", "w") as package:
        package.write(args.output / "program.bin", "program.ptg")
    (args.output / "input.bin").write_bytes(struct.pack("<6f", 1, 2, 3, 4, 5, 6))
    (args.output / "expected.bin").write_bytes(struct.pack("<6f", 2, 4, 6, 8, 10, 12))


if __name__ == "__main__":
    main()
