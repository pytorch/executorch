# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors

"""Read µYOLO I/O quantization metadata embedded in a PTE."""

from pathlib import Path


def io_method_names() -> tuple[str, ...]:
    return (
        "input0_scale",
        "input0_zp",
        "input0_quant_min",
        "input0_quant_max",
        "output0_scale",
        "output0_zp",
        "output0_quant_min",
        "output0_quant_max",
    )


def read_io_qparams(path: Path) -> dict[str, float | int]:
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(path)
    values: dict[str, float | int] = {}
    for name in io_method_names():
        result = program.load_method(name).execute(())[0]
        values[name] = result.item() if hasattr(result, "item") else result
    return values
