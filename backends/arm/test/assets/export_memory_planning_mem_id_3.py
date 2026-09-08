# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import Optional

import executorch.exir as exir
import torch
from executorch.exir import to_edge
from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
from executorch.exir.pass_base import PassResult
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule


class CustomPoolMemoryPlanningPass(MemoryPlanningPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                if node.op == "placeholder":
                    node.meta["spec"].mem_id = 1
                    continue

                if node.op != "call_function":
                    continue

                if node.target == torch.ops.aten.add.out:
                    node.meta["spec"].mem_id = 3
                elif node.target == torch.ops.aten.mul.out:
                    node.meta["spec"].mem_id = 1

        return super().run(graph_module)

    def run(
        self,
        graph_module: GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        return self.call(graph_module)


class MultiplePoolsToyModel(torch.nn.Module):
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        b = a + a
        c = a * b
        d = c + b
        e = c * d
        return e


def export_pte(output_path: Path) -> None:
    edge_program = to_edge(
        export(MultiplePoolsToyModel(), (torch.ones(1),), strict=True)
    )
    mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
    executorch_program = edge_program.to_executorch(
        exir.ExecutorchBackendConfig(
            memory_planning_pass=CustomPoolMemoryPlanningPass(
                memory_planning_algo=mem_algo,
                alignment=1,
            ),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        executorch_program.write_to_file(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a portable .pte with mem_id=3 planned tensor storage."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the .pte file to write.",
    )
    args = parser.parse_args()

    export_pte(args.output)


if __name__ == "__main__":
    main()
