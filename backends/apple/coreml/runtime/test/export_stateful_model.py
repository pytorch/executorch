# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import os
from pathlib import Path

import coremltools as ct
import executorch.exir as exir

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from torch.export import export


class StatefulModel(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.register_buffer(
            "cache", torch.zeros((max_seq_len, embedding_dim), dtype=torch.float32)
        )

    def forward(
        self,
        q: torch.Tensor,
        k_val: torch.Tensor,
        input_pos: torch.Tensor,
    ):
        q_T = q.transpose(0, 1)
        k = torch.ops.aten.index_put_(self.cache, [input_pos, None], k_val)
        attn = k.mm(q_T)
        return attn


def main() -> None:
    embedding_dim = 3
    max_seq_len = 2
    model = StatefulModel(embedding_dim=embedding_dim, max_seq_len=max_seq_len)
    example_inputs = (
        torch.randn((1, embedding_dim)),
        torch.randn((1, embedding_dim)),
        torch.tensor([0]),
    )
    exported_model = export(model, example_inputs)
    edge_program_manager = exir.to_edge(exported_model)
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16,
        compute_unit=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    partitioner = CoreMLPartitioner(
        skip_ops_for_coreml_delegation=None,
        compile_specs=compile_specs,
    )

    delegated_program_manager = edge_program_manager.to_backend(partitioner)
    exec_program = delegated_program_manager.to_executorch(
        config=exir.ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    buffer = exec_program.buffer
    models_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "models"
    models_dir.mkdir(parents=False, exist_ok=True)
    file_path = models_dir / "state_coreml_all.pte"
    with open(file_path.resolve(), "wb") as file:
        file.write(buffer)


if __name__ == "__main__":
    main()  # pragma: no cover
