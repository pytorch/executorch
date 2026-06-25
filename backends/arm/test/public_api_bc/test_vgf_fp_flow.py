# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
from executorch.backends.arm import VgfBackend, VgfCompileSpec, VgfPartitioner
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program


class TinyAddSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x + y)


def _configured_compile_spec(tmp_path: Path) -> VgfCompileSpec:
    compile_spec = VgfCompileSpec("TOSA-1.0+FP")

    assert compile_spec == VgfCompileSpec("TOSA-1.0+FP")
    assert "VgfCompileSpec" in repr(compile_spec)

    compile_spec.dump_intermediate_artifacts_to(str(tmp_path / "vgf_fp_intermediates"))
    returned = compile_spec.dump_debug_info(VgfCompileSpec.DebugMode.TOSA)
    assert returned is compile_spec
    return compile_spec


def test_vgf_fp_public_api_scenario(tmp_path: Path) -> None:
    backend = VgfBackend()
    assert isinstance(backend, VgfBackend)

    compile_spec = _configured_compile_spec(tmp_path)
    partitioner = VgfPartitioner(compile_spec)

    example_inputs = (
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
    )
    exported_program_for_partition = torch.export.export(
        TinyAddSigmoid().eval(),
        example_inputs,
    )
    ops_to_preserve, filter_fn = partitioner.ops_to_not_decompose(
        exported_program_for_partition
    )
    partition_result = partitioner.partition(exported_program_for_partition)

    assert isinstance(ops_to_preserve, list)
    assert filter_fn is None or callable(filter_fn)
    assert partition_result.tagged_exported_program is exported_program_for_partition

    exported_program = torch.export.export(TinyAddSigmoid().eval(), example_inputs)
    edge_manager = to_edge_transform_and_lower(
        programs=exported_program,
        partitioner=[partitioner],
    )
    executorch_program_manager = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    pte_path = tmp_path / "vgf_fp_public_api_bc.pte"
    save_pte_program(executorch_program_manager, str(pte_path))

    assert pte_path.is_file()
    assert pte_path.stat().st_size > 0
    assert any((tmp_path / "vgf_fp_intermediates").rglob("*"))
