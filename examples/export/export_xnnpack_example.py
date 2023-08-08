# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse

import executorch.exir as exir
import torch
from executorch.backends.backend_api import to_backend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackFloatingPointPartitioner

# from ..models import MODEL_NAME_TO_MODEL



def export_add_module_with_lower_graph():
    """

    AddMulModule:

        input -> torch.mm -> torch.add -> output

    this module can be lowered to the demo backend as a delegate

        input -> [lowered module (delegate)] -> output

    the lowered module can be used to composite with other modules

        input -> [lowered module (delegate)] -> sub  -> output
               |--------  composite module    -------|

    """
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + x

    capture_config = exir.CaptureConfig(pt2_mode=True, enable_dynamic_shape=False)
    edge_compile_config = exir.EdgeCompileConfig()
    sample_inputs = (torch.ones(1, 2, 3, 4),)
    print("Running the example to export a composite module with lowered graph...")
    edge = exir.capture(AddModule(), sample_inputs, capture_config).to_edge(edge_compile_config)
    print("Exported graph:\n", edge.exported_program.graph)

    # Lower AddMulModule to the demo backend
    print("Lowering to the demo backend...")
    edge.exported_program = to_backend(
        edge.exported_program, XnnpackFloatingPointPartitioner
    )


    # The graph module is still runnerable
    edge.exported_program.graph_module(*sample_inputs)

    print("Lowered graph:\n", edge.exported_program.graph)

    exec_prog = edge.to_executorch()
    buffer = exec_prog.buffer

    model_name = "xnnpack_add"
    filename = f"{model_name}.ff"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)



OPTIONS_TO_LOWER = {
    "add": export_add_module_with_lower_graph
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--option",
        required=True,
        choices=list(OPTIONS_TO_LOWER.keys()),
        help=f"Provide the flow name. Valid ones: {list(OPTIONS_TO_LOWER.keys())}",
    )

    args = parser.parse_args()

    # Choose one option
    option = OPTIONS_TO_LOWER[args.option]

    # Run the example flow
    option()
