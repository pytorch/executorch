#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the example models against the library that will run them.

    python verify_models.py arduino_lib/ExecuTorch

A .pte records, per operator call, how many values it puts on the stack. The
generated kernel wrappers check that count and reject anything else. The two
only agree when the model and the library came from the same ExecuTorch
commit, because Cortex-M operator schemas change between releases -- `scratch`
was added to the conv operators in #19636 and #19825.

A mismatch is invisible until far too late: the program loads, every operator
resolves, and then Method::execute returns InvalidProgram (0x23) naming
nothing useful. This compares the two directly, in about a second, with no
board and no toolchain.
"""

import argparse
import pathlib
import re
import sys

from executorch.exir._serialize._program import deserialize_pte_binary

KERNEL_RE = re.compile(r'Kernel\(\s*"([^"]+)"(.*?)stack\.size\(\)\s*==\s*(\d+)', re.S)


def library_expectations(library: pathlib.Path) -> dict[str, int]:
    """Stack size each registered kernel wrapper demands, by operator name."""
    generated = list((library / "src/executorch/codegen").glob("Register*Kernels*.cpp"))
    if not generated:
        sys.exit(f"no kernel registration found under {library}/src/executorch/codegen")
    expectations: dict[str, int] = {}
    for source in generated:
        for match in KERNEL_RE.finditer(source.read_text(encoding="utf-8")):
            name, size = match.group(1), int(match.group(3))
            previous = expectations.get(name)
            if previous is not None and previous != size:
                sys.exit(
                    f"{name} is registered twice with different stack sizes "
                    f"({previous} and {size}); the library is inconsistent"
                )
            expectations[name] = size
    if not expectations:
        sys.exit(
            f"parsed no stack-size checks out of {len(generated)} generated "
            "file(s). The codegen output format has probably changed, so this "
            "script needs updating -- treat that as a bug here, not a bad model."
        )
    return expectations


def model_calls(pte: pathlib.Path) -> list[tuple[str, int]]:
    """Operator name and stack size for every kernel call in a .pte."""
    parsed = deserialize_pte_binary(pte.read_bytes())
    plan = getattr(parsed, "program", parsed).execution_plan[0]
    calls = []
    for chain in plan.chains:
        for instruction in chain.instructions:
            args = getattr(instruction.instr_args, "args", None)
            index = getattr(instruction.instr_args, "op_index", None)
            if args is None or index is None:
                continue  # not a kernel call
            op = plan.operators[index]
            name = f"{op.name}.{op.overload}" if op.overload else op.name
            calls.append((name, len(args)))
    return calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=pathlib.Path, help="Generated library root")
    args = parser.parse_args()

    expectations = library_expectations(args.library)
    models = sorted(args.library.glob("examples/*/model.pte"))
    if not models:
        # The build converts each model.pte to model.h and removes it, so run
        # this against the source tree rather than the packaged library.
        models = sorted(pathlib.Path(__file__).parent.glob("examples/*/model.pte"))
    if not models:
        sys.exit("no example models found")

    failures = 0
    for pte in models:
        problems = []
        for name, provided in model_calls(pte):
            expected = expectations.get(name)
            if expected is None:
                problems.append(f"{name}: not registered in the library")
            elif expected != provided:
                problems.append(
                    f"{name}: model supplies {provided}, library expects {expected}"
                )
        if problems:
            failures += len(problems)
            print(f"FAIL {pte.parent.name}")
            for problem in dict.fromkeys(problems):
                print(f"       {problem}")
        else:
            print(f"ok   {pte.parent.name}")

    if failures:
        print(
            "\nThe models and the library came from different ExecuTorch commits.\n"
            "Re-export the models from the same checkout that built the library;\n"
            "see the pin in extras/PROVENANCE.txt."
        )
        return 1
    print(f"\n{len(models)} models match the library")
    return 0


if __name__ == "__main__":
    sys.exit(main())
