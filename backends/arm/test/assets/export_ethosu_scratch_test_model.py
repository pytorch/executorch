# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from executorch.backends.arm import (
    EthosUCompileSpec,
    EthosUPartitioner,
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm._passes import EthosUMemoryPlanning
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.schema import Tensor as BundledTensor
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
    to_edge_transform_and_lower,
)
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision import models  # type: ignore[import-untyped]


def _build_quantized_program(compile_spec: EthosUCompileSpec, model_name: str):
    torch.manual_seed(0)
    if model_name == "conv":
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU()
        ).eval()
        example_inputs = (torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16),)
    else:
        model = models.mobilenetv2.mobilenet_v2(weights=None).eval()
        example_inputs = (torch.rand(1, 3, 224, 224),)
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    example_inputs = (example_inputs[0].to(memory_format=torch.channels_last),)
    exported_program = torch.export.export(model, example_inputs)
    graph_module = exported_program.module(check_guards=False)

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    prepared = prepare_pt2e(graph_module, quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    with torch.no_grad():
        expected_output = converted(*example_inputs)
    return (
        torch.export.export(converted, example_inputs),
        example_inputs,
        expected_output,
    )


def _assert_ethosu_scratch_plan(executorch_program_manager) -> None:
    graph = executorch_program_manager.exported_program().graph_module.graph
    delegate_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == executorch_call_delegate
    ]
    if len(delegate_nodes) == 0:
        raise AssertionError("Expected at least one Ethos-U delegate")

    for delegate_node in delegate_nodes:
        scratch = memory.delegate_scratch(delegate_node)
        if len(scratch) != 1:
            raise AssertionError(
                f"Expected one planned scratch buffer, got {len(scratch)}"
            )
        spec = scratch[0].spec
        if spec.dtype != torch.uint8 or spec.nbytes() == 0 or spec.mem_offset is None:
            raise AssertionError(f"Invalid Ethos-U scratch spec: {spec}")
        io_specs = [arg.meta["spec"] for arg in delegate_node.args[1:]]
        outputs = delegate_node.meta["spec"]
        io_specs.extend(outputs if isinstance(outputs, (list, tuple)) else [outputs])
        if any(io_spec.storage_base is not spec for io_spec in io_specs):
            raise AssertionError("Expected all delegate I/O to alias planned scratch")


def export_pte(
    output_path: Path,
    target: str,
    planning_mode: str = "shared",
    model_name: str = "mobilenet",
    bundle_output: Path | None = None,
) -> None:
    """Exports an Ethos-U program and optional reference output bundle.

    Args:
        output_path (Path): Destination for the exported program.
        target (str): Ethos-U accelerator configuration passed to Vela.
        planning_mode (str): Shared I/O, ordinary scratch, or legacy planning.
        model_name (str): Model to quantize and lower.
        bundle_output (Path | None): Optional bundled program destination.

    """
    # Exercise COP1 shared scratch on both U55 and U85.
    compile_spec = EthosUCompileSpec(
        target,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        plan_delegate_scratch=planning_mode != "legacy",
    )
    program, example_inputs, expected_output = _build_quantized_program(
        compile_spec, model_name
    )
    edge_manager = to_edge_transform_and_lower(
        programs=program,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_manager = edge_manager.transform([ReplaceQuantNodesPass()])
    executorch_program_manager = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=(
                EthosUMemoryPlanning()
                if planning_mode == "shared"
                else MemoryPlanningPass()
            ),
            extract_delegate_segments=False,
        )
    )
    if planning_mode == "shared":
        _assert_ethosu_scratch_plan(executorch_program_manager)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pte_program(executorch_program_manager, str(output_path))
    if bundle_output is not None:
        test_suite = MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(inputs=example_inputs, expected_outputs=expected_output)
            ],
        )
        bundle = BundledProgram(executorch_program_manager, [test_suite])
        # Bundling makes tensors contiguous; restore the PTE's I/O layouts.
        schema = bundle.serialize_to_schema()
        (plan,) = executorch_program_manager.executorch_program.execution_plan
        (test_case,) = schema.method_test_suites[0].test_cases
        for values, tensors, indices in (
            (test_case.inputs, example_inputs, plan.inputs),
            (test_case.expected_outputs, (expected_output,), plan.outputs),
        ):
            for value, tensor, index in zip(values, tensors, indices):
                dim_order = plan.values[index].val.dim_order
                bundled_tensor = cast(BundledTensor, value.val)
                bundled_tensor.dim_order = dim_order
                bundled_tensor.data = (
                    tensor.permute(dim_order).contiguous().numpy().tobytes()
                )
        bundle_output.parent.mkdir(parents=True, exist_ok=True)
        bundle_output.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))


def main() -> None:
    """Exports the program selected by the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export an Ethos-U PTE with planned scratch and shared COP1 I/O."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="ethos-u55-128")
    parser.add_argument(
        "--planning-mode", choices=("shared", "scratch", "legacy"), default="shared"
    )
    parser.add_argument("--model", choices=("mobilenet", "conv"), default="mobilenet")
    parser.add_argument("--bundle-output", type=Path)
    args = parser.parse_args()

    export_pte(
        args.output, args.target, args.planning_mode, args.model, args.bundle_output
    )


if __name__ == "__main__":
    main()
