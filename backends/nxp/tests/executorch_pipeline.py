# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Callable, Iterable

import executorch.export.export
import numpy as np
import torch
from executorch import exir
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.export.export_utils import (
    get_default_quantizer,
    handle_kernel_selection,
    ModelInputSpec,
    to_model_input_spec,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import (
    core_aten_ops_exception_list,
    default_preserve_ops,
    generate_neutron_compile_spec,
)
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.nxp.recipes import (
    NeutronRecipeConfig,
    NXPRecipeProvider,
    NXPRecipeType,
)
from executorch.backends.nxp.recipes.nxp_recipe_provider import (
    NEUTRON_RECIPE_CONFIG_KEY,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)

from executorch.export import ExportRecipe, ExportSession, StageType
from torch import nn
from torch.export import export
from torchao.quantization.pt2e.quantizer import Quantizer


neutron_target_spec = NeutronTargetSpec(target="imxrt700")


def get_random_calibration_inputs(
    input_spec: Iterable[ModelInputSpec], num_samples: int = 4
) -> list[tuple[torch.Tensor, ...]]:
    return [
        tuple([torch.randn(spec.shape, dtype=spec.dtype) for spec in input_spec])
        for _ in range(num_samples)
    ]


GetCalibrationInputsFn = Callable[
    [tuple[ModelInputSpec, ...]], Iterable[tuple[torch.Tensor, ...]]
]


def get_calibration_inputs_fn_from_dataset_dir(dataset_dir) -> GetCalibrationInputsFn:
    def _nested(
        input_spec: tuple[ModelInputSpec, ...],
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        data = sorted(os.listdir(dataset_dir))
        inputs_needed = len(input_spec)

        for path in data:
            path = os.path.join(dataset_dir, path)
            files = []

            if os.path.isdir(path):
                files = [os.path.join(path, x) for x in sorted(os.listdir(path))]
            else:
                files.append(path)

            input_data = []
            for idx, file in enumerate(files):
                if len(input_data) == inputs_needed:
                    break

                tensor = np.fromfile(
                    file, dtype=torch_type_to_numpy_type(input_spec[idx].dtype)
                ).reshape(input_spec[idx].shape)
                input_data += (torch.from_numpy(tensor),)
                continue

            if len(input_data) < inputs_needed:
                continue

            yield tuple(input_data)

    return _nested


def get_example_input(
    input_spec: tuple[ModelInputSpec, ...],
) -> tuple[torch.Tensor, ...]:
    example_input = []
    for spec in input_spec:
        match spec.dim_order:
            case torch.contiguous_format:
                sample = torch.ones(spec.shape, dtype=spec.dtype)
            case torch.channels_last:
                sample = torch.ones(spec.shape, dtype=spec.dtype).to(
                    memory_format=torch.channels_last
                )
            case _:
                raise ValueError(f"Unsupported dim_order: {spec.dim_order}")
        # noinspection PyUnboundLocalVariable
        example_input.append(sample)

    return tuple(example_input)


def get_recipe(
    rc: NeutronRecipeConfig,
    use_qat: bool = False,
    delegate_to_npu: bool = False,
) -> ExportRecipe:
    match [use_qat, delegate_to_npu]:
        case [False, False]:
            recipe_type = NXPRecipeType.INT8_PTQ_NO_DELEGATE
        case [False, True]:
            recipe_type = NXPRecipeType.INT8_PTQ_NEUTRON
        case [True, False]:
            recipe_type = NXPRecipeType.INT8_QAT_NO_DELEGATE
        case [True, True]:
            recipe_type = NXPRecipeType.INT8_QAT_NEUTRON
        case _:
            raise RuntimeError(
                "export_with_recipe: `use_qat` and `delegate_to_npu` must be booleans."
            )

    recipe = NXPRecipeProvider().create_recipe(
        recipe_type, **{NEUTRON_RECIPE_CONFIG_KEY: rc}
    )
    return recipe


def export_with_recipe(
    model: torch.nn.Module,
    example_inputs: list[tuple[torch.Tensor, ...]],
    rc: NeutronRecipeConfig,
    use_qat: bool = False,
    delegate_to_npu: bool = False,
) -> ExportSession:
    recipe = get_recipe(rc, use_qat, delegate_to_npu)
    return executorch.export.export(model, example_inputs, recipe)


def to_quantized_edge_program(
    model: torch.nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
    operators_not_to_delegate: list[str] = None,
    get_calibration_inputs_fn: GetCalibrationInputsFn = get_random_calibration_inputs,
    target: str = "imxrt700",
    intermediates_dir: str | None = None,
    use_qat: bool = False,
    train_fn: Callable[[torch.fx.GraphModule], None] | None = None,
    remove_quant_io_ops: bool = False,
    custom_delegation_options: CustomDelegationOptions = CustomDelegationOptions(),  # noqa B008
    get_quantizer_fn: Callable[[], Quantizer] | None = None,
    use_neutron_for_format_conversion: bool = True,
    use_quant_state_dict: bool = True,
    fetch_constants_to_sram: bool = False,
    dump_kernel_selection_code: bool = False,
    use_profiling: bool = False,
    delegate_to_npu=True,
    use_recipe_export: bool = False,
) -> EdgeProgramManager:
    _neutron_target_spec = NeutronTargetSpec(target)
    input_spec = to_model_input_spec(input_spec)
    calibration_inputs = get_calibration_inputs_fn(input_spec)
    example_input = get_example_input(input_spec)

    if use_recipe_export:
        rc = NeutronRecipeConfig(
            input_spec,
            target,
            operators_not_to_delegate,
            intermediates_dir,
            get_quantizer_fn,  # None means recipe provider uses its default; avoids deepcopy issues
            custom_delegation_options,
            remove_quant_io_ops,
            use_quant_state_dict,
            use_neutron_for_format_conversion,
            fetch_constants_to_sram,
            dump_kernel_selection_code,
            use_profiling,
            train_fn,
            get_calibration_inputs_fn,  # thread calibration data into the recipe
        )
        recipe = get_recipe(rc, use_qat, delegate_to_npu)

        # We do not want to run the TO_EXECUTORCH stage, as this function should provide the edge_program_manager and
        #  the `.to_executorch()` call would mutate the model in place.
        stages = ExportSession(model, [example_input], recipe)._get_default_pipeline()
        stages.remove(StageType.TO_EXECUTORCH)
        recipe.pipeline_stages = stages

        session = executorch.export.export(model, [example_input], recipe)
        print(
            f"\n\n\n\n to_quantized_edge_program: {session._quant_recipe.train_fn} \n\n\n\n\n\n\n"
        )

        return session.get_edge_program_manager()

    else:
        # Make sure the model is in the evaluation mode.
        model.eval()

        # Build the default quantizer only for the imperative path to avoid holding
        # an SDK handle inside a partial that would later be deep-copied.
        if get_quantizer_fn is None:
            get_quantizer_fn = partial(
                get_default_quantizer, _neutron_target_spec, use_qat
            )

        exir_program_aten = torch.export.export(model, example_input, strict=True)

        exir_program_aten__module_quant = calibrate_and_quantize(
            model=exir_program_aten,
            calibration_inputs=calibration_inputs,
            quantizer=get_quantizer_fn(),
            is_qat=use_qat,
            train_fn=train_fn,
        )

        compile_spec = generate_neutron_compile_spec(
            target,
            intermediates_dir=intermediates_dir,
            operators_not_to_delegate=operators_not_to_delegate,
            use_neutron_for_format_conversion=use_neutron_for_format_conversion,
            fetch_constants_to_sram=fetch_constants_to_sram,
            dump_kernel_selection_code=dump_kernel_selection_code,
            use_profiling=use_profiling,
        )
        post_quant_state_dict = (
            exir_program_aten__module_quant.state_dict()
            if use_quant_state_dict
            else None
        )
        if delegate_to_npu:
            partitioners = [
                NeutronPartitioner(
                    compile_spec,
                    _neutron_target_spec,
                    custom_delegation_options,
                    post_quant_state_dict,
                    preserve_ops=default_preserve_ops,
                )
            ]
        else:
            partitioners = []

        edge_program_manager = to_edge_transform_and_lower(
            export(exir_program_aten__module_quant, example_input, strict=True),
            transform_passes=NeutronEdgePassManager(),
            partitioner=partitioners,
            generate_etrecord=use_profiling,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _core_aten_ops_exception_list=core_aten_ops_exception_list,
            ),
        )

        if remove_quant_io_ops:
            edge_program_manager = edge_program_manager.transform(
                [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
            )

        edge_program_manager = edge_program_manager.transform(
            NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
        )

        if dump_kernel_selection_code:
            handle_kernel_selection()

        return edge_program_manager


def to_quantized_executorch_program(
    model: torch.nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
    intermediates_dir: str | None = None,
    use_qat: bool = False,
    train_fn: Callable[[torch.fx.GraphModule], None] | None = None,
    use_neutron_for_format_conversion: bool = True,
    dataset_dir: str | None = None,
    delegate_to_npu=True,
    use_profiling: bool = False,
    operators_not_to_delegate: list[str] | None = None,
    remove_quant_io_ops: bool = False,
    use_recipe_export: bool = False,
) -> ExecutorchProgramManager:
    if dataset_dir:
        # Extract calibration data from a directory.
        get_calibration_inputs_fn = get_calibration_inputs_fn_from_dataset_dir(
            dataset_dir
        )
    else:
        get_calibration_inputs_fn = None  # use default (random) in both paths

    if use_recipe_export:
        rc = NeutronRecipeConfig(
            input_spec,
            operators_not_to_delegate=operators_not_to_delegate,
            intermediates_dir=intermediates_dir,
            remove_quant_io_ops=remove_quant_io_ops,
            use_neutron_for_format_conversion=use_neutron_for_format_conversion,
            use_profiling=use_profiling,
            train_fn=train_fn,
            calibration_inputs_fn=get_calibration_inputs_fn,
        )
        example_inputs = [get_example_input(to_model_input_spec(input_spec))]
        session = export_with_recipe(
            model, example_inputs, rc, use_qat, delegate_to_npu
        )
        print(
            f"\n\n\n\n to_quantized_executorch_program: {session._quant_recipe.train_fn} \n\n\n\n\n\n\n"
        )
        return session.get_executorch_program_manager()

    else:
        calib_fn_kwarg = (
            {"get_calibration_inputs_fn": get_calibration_inputs_fn}
            if get_calibration_inputs_fn is not None
            else {}
        )

        edge_program_manager = to_quantized_edge_program(
            model,
            input_spec,
            intermediates_dir=intermediates_dir,
            use_qat=use_qat,
            train_fn=train_fn,
            use_neutron_for_format_conversion=use_neutron_for_format_conversion,
            delegate_to_npu=delegate_to_npu,
            use_profiling=use_profiling,
            operators_not_to_delegate=operators_not_to_delegate,
            remove_quant_io_ops=remove_quant_io_ops,
            **calib_fn_kwarg,
        )

        return edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )


def to_edge_program(
    model: nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
) -> EdgeProgramManager:
    example_input = get_example_input(to_model_input_spec(input_spec))

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program = torch.export.export(model, example_input)
    return exir.to_edge(exir_program)
