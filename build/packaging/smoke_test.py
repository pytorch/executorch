#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is run by CI after building the executorch wheel. Before running
this, the job will install the matching torch package as well as the newly-built
executorch package and its dependencies.
"""

# Import this first. If it can't find the torch.so libraries, the dynamic load
# will fail and the process will exit.
from executorch.extension.pybindings import portable_lib  # usort: skip

# Import this after importing the ExecuTorch pybindings. If the pybindings
# links against a different torch.so than this uses, there will be a set of
# symbol comflicts; the process will either exit now, or there will be issues
# later in the smoke test.
import torch  # usort: skip

# Import everything else later to help isolate the critical imports above.
import os
import tempfile
from typing import Tuple

from executorch.exir import to_edge
from torch.export import export


class LinearModel(torch.nn.Module):
    """Runs Linear on its input, which should have shape [4]."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor):
        """Expects a single tensor of shape [4]."""
        return self.linear(x)


def linear_model_inputs() -> Tuple[torch.Tensor]:
    """Returns some example inputs compatible with LinearModel."""
    # The model takes a single tensor of shape [4] as an input.
    return (torch.ones(4),)


def export_linear_model() -> bytes:
    """Exports LinearModel and returns the .pte data."""

    # This helps the exporter understand the shapes of tensors used in the model.
    # Since our model only takes one input, this is a one-tuple.
    example_inputs = linear_model_inputs()

    # Export the pytorch model and process for ExecuTorch.
    print("Exporting program...")
    exported_program = export(LinearModel(), example_inputs)
    print("Lowering to edge...")
    edge_program = to_edge(exported_program)
    print("Creating ExecuTorch program...")
    et_program = edge_program.to_executorch()

    return et_program.buffer


def main():
    """Tests the export and execution of a simple model."""

    # If the pybindings loaded correctly, we should be able to ask for the set
    # of operators.
    ops = portable_lib._get_operator_names()
    assert len(ops) > 0, "Empty operator list"
    print(f"Found {len(ops)} operators; first element '{ops[0]}'")

    # Export LinearModel to .pte data.
    pte_data: bytes = export_linear_model()

    # Try saving to and loading from a file.
    with tempfile.TemporaryDirectory() as tempdir:
        pte_file = os.path.join(tempdir, "linear.pte")

        # Save the .pte data to a file.
        with open(pte_file, "wb") as file:
            file.write(pte_data)
            print(f"ExecuTorch program saved to {pte_file} ({len(pte_data)} bytes).")

        # Load the model from disk.
        m = portable_lib._load_for_executorch(pte_file)

        # Run the model.
        outputs = m.forward(linear_model_inputs())

        # Should see a single output with shape [2].
        assert len(outputs) == 1, f"Unexpected output length {len(outputs)}: {outputs}"
        assert outputs[0].shape == (2,), f"Unexpected output size {outputs[0].shape}"

    print("PASS")


if __name__ == "__main__":
    main()
