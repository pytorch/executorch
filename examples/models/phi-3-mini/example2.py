# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pprint import pprint

import torch.nn

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge, ExecutorchBackendConfig

from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


class ExampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
    ):
        x.copy_(y)


def main() -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        model = ExampleModel()
        example_inputs = (
            torch.zeros((1, 10), dtype=torch.long),
            torch.ones((1, 10), dtype=torch.long)
        )

        model = torch.export.export(
            model, example_inputs, strict=False
        )
        print(model)
        edge_manager = to_edge(model, compile_config=get_xnnpack_edge_compile_config())
        print("Graph:")
        print(edge_manager.exported_program().graph_module.graph)
        print("Graph signature:")
        pprint(edge_manager.exported_program().graph_signature)
        edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        et_program = edge_manager.to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            )
        )
        print("ExecuTorch program:")
        pprint(et_program.executorch_program)
        print("Graph:")
        print(et_program.exported_program().graph_module.graph)
        print("Graph signature:")
        pprint(et_program.exported_program().graph_signature)


        with open("example2.pte", "wb") as file:
            file.write(et_program.buffer)


def main2():
    x = torch.zeros((1, 10), dtype=torch.long)
    y = torch.ones((1, 10), dtype=torch.long)
    model = ExampleModel()
    model.forward(x, y)
    print(f"x: {x}")
    print(f"y: {y}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--export",
        default=True,
        action="store_true",
        help="Whether or not to export",
    )
    args = parser.parse_args()
    if args.export:
        main()
    else:
        main2()
