# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from executorch.extension.llm.batching.sampler import BatchSampler


def export_sampler(directory: Path, vocab: int):
    rows = torch.export.Dim("rows", min=1, max=64)
    program = torch.export.export(
        BatchSampler(),
        (torch.zeros(64, vocab), torch.zeros(64, 4)),
        dynamic_shapes=({0: rows}, {0: rows}),
        strict=True,
    )
    specs = [CudaBackend.generate_method_name_compile_spec("sample")]
    edge = to_edge_transform_and_lower(
        {"sample": program},
        partitioner={"sample": [CudaPartitioner(specs)]},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    result = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            propagate_device_config={
                "sample": PropagateDeviceConfig(skip_h2d_for_method_inputs=True)
            },
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.pte").open("wb") as output:
        result.write_to_file(output)
    return directory / "model.pte"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class CudaBatchSamplerTest(unittest.TestCase):
    def test_dynamic_rows_with_masked_reduction_lanes(self):
        from executorch.runtime import Runtime

        # A vocabulary spanning several reduction tiles exposes padding lanes.
        vocab = 202048
        with tempfile.TemporaryDirectory() as temporary:
            path = export_sampler(Path(temporary), vocab)
            method = Runtime.get().load_program(path).load_method("sample")
            for count in (1, 3, 4, 5, 64):
                with self.subTest(rows=count):
                    logits = torch.full((count, vocab), -20.0)
                    logits[:, :8] = torch.arange(8).float() / 2
                    settings = [(0.0, 1.0, 0.0, 0.0), (0.7, 0.8, 4.0, 0.37)]
                    params = torch.tensor([settings[i % 2] for i in range(count)])
                    if count == 1:
                        params[0] = torch.tensor(settings[1])
                    expected = []
                    for temperature, _, top_k, coin in params.tolist():
                        if temperature == 0:
                            expected.append(7)
                            continue
                        order = list(range(7, 7 - int(top_k), -1))
                        mass = [math.exp((i / 2 - 3.5) / temperature) for i in order]
                        threshold = coin * sum(mass)
                        cumulative = 0.0
                        for token, probability in zip(order, mass):
                            cumulative += probability
                            if cumulative > threshold:
                                expected.append(token)
                                break
                    actual = method.execute([logits.cuda(), params.cuda()])[0]
                    self.assertEqual(actual.cpu().tolist(), expected)
