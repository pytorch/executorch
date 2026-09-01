# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stage 4: run model.pte through the ExecuTorch runtime.
#
# Loads the .pte produced by stage 3 and executes it, asserting the output
# shape matches expectations.  If the ExecuTorch pybindings are not available
# in this environment a warning is printed and the script exits cleanly; the
# .pte produced by stage 3 is still valid.

import argparse
import os

from model import get_example_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: run model.pte through the ExecuTorch runtime."
    )
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    pte_path = os.path.join(args.workdir, "model.pte")
    assert os.path.isfile(
        pte_path
    ), f"model.pte not found at {pte_path}  (run 3_lower.py first)"

    try:
        from executorch.runtime import Runtime
    except ModuleNotFoundError:
        print(
            "WARNING: executorch.runtime is not available in this environment. "
            "Build and install the ExecuTorch pybindings to run the .pte file. "
            "Skipping runtime execution."
        )
        return

    print(f"Loading {pte_path} ...")
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    ex = get_example_inputs()
    outputs = method.execute(ex)

    assert len(outputs) == 1, f"Expected 1 output tensor, got {len(outputs)}"
    out_tensor = outputs[0]
    assert out_tensor.shape == (1, 10), f"Unexpected output shape: {out_tensor.shape}"
    print(
        f"Runtime execution succeeded.  Output shape: {out_tensor.shape}  (assertion passed)"
    )
    print("\nStage 4 done.")


if __name__ == "__main__":
    main()
