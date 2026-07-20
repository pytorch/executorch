# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
from executorch.backends.arm import (
    get_symmetric_quantization_config,
    VgfCompileSpec,
    VgfPartitioner,
    VgfQuantizer,
)
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TinyAddSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x + y)


def _configured_compile_spec(tmp_path: Path) -> VgfCompileSpec:
    compile_spec = VgfCompileSpec("TOSA-1.0+INT")

    assert compile_spec == VgfCompileSpec("TOSA-1.0+INT")
    assert "VgfCompileSpec" in repr(compile_spec)

    compile_spec.dump_intermediate_artifacts_to(str(tmp_path / "vgf_int_intermediates"))
    returned = compile_spec.dump_debug_info(VgfCompileSpec.DebugMode.TOSA)
    assert returned is compile_spec
    return compile_spec


def _exercise_quantizer_api(compile_spec: VgfCompileSpec) -> None:
    quantizer = VgfQuantizer(compile_spec)
    symmetric_config = get_symmetric_quantization_config(is_per_channel=False)

    quantizer.set_global(symmetric_config)
    quantizer.set_io(symmetric_config)
    quantizer.set_module_name("sigmoid", symmetric_config)
    quantizer.set_module_type(torch.nn.Sigmoid, symmetric_config)

    example_inputs = (
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
    )
    graph_module = torch.export.export(
        TinyAddSigmoid().eval(),
        example_inputs,
    ).module(check_guards=False)
    transformed = quantizer.transform_for_annotation(graph_module)
    annotated = quantizer.annotate(transformed)
    quantizer.validate(annotated)


def _build_quantized_program(compile_spec: VgfCompileSpec):
    model = TinyAddSigmoid().eval()
    example_inputs = (
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
    )
    exported_program = torch.export.export(model, example_inputs)
    graph_module = exported_program.module(check_guards=False)

    quantizer = VgfQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    prepared = prepare_pt2e(graph_module, quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    return torch.export.export(converted, example_inputs)


def test_vgf_int_public_api_scenario(tmp_path: Path) -> None:
    compile_spec = _configured_compile_spec(tmp_path)
    _exercise_quantizer_api(compile_spec)

    partitioner = VgfPartitioner(compile_spec)
    quantized_program_for_partition = _build_quantized_program(compile_spec)
    ops_to_preserve, filter_fn = partitioner.ops_to_not_decompose(
        quantized_program_for_partition
    )
    partition_result = partitioner.partition(quantized_program_for_partition)

    assert isinstance(ops_to_preserve, list)
    assert filter_fn is None or callable(filter_fn)
    assert partition_result.tagged_exported_program is quantized_program_for_partition

    quantized_program = _build_quantized_program(compile_spec)
    edge_manager = to_edge_transform_and_lower(
        programs=quantized_program,
        partitioner=[partitioner],
    )
    executorch_program_manager = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    pte_path = tmp_path / "vgf_int_public_api_bc.pte"
    save_pte_program(executorch_program_manager, str(pte_path))

    assert pte_path.is_file()
    assert pte_path.stat().st_size > 0
    assert any((tmp_path / "vgf_int_intermediates").rglob("*"))
