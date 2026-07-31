# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast, Iterable, Optional, Sequence

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import (
    core_aten_ops_exception_list,
    default_preserve_ops,
    generate_neutron_compile_spec,
)
from executorch.backends.nxp.recipes.nxp_recipe_types import NXP_BACKEND, NXPRecipeType
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_default_quantizer,
    handle_kernel_selection,
    ModelInputSpec,
    to_model_input_spec,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExportedProgram

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import Partitioner
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    RecipeType,
)
from torchao.quantization.pt2e.quantizer import Quantizer


class NeutronEdgePassManagerWrapper:
    def __init__(self, passes: list[NeutronEdgePass] | None = None):
        self.neutron_edge_pass_manager = NeutronEdgePassManager(passes)

    def __call__(
        self, method_name: str, exported_program: ExportedProgram
    ) -> NeutronEdgePassManager:
        return self.neutron_edge_pass_manager


NEUTRON_RECIPE_CONFIG_KEY = "neutron_recipe_config"


@dataclass
class NeutronRecipeConfig:
    """Configuration shared by all NXP recipe types.

    Parameters that vary the *type* of export (delegate vs no-delegate)
    are expressed by choosing a different NXPRecipeType rather than by flags here.

    Attributes:
        input_spec: Model input description. Accepts a single shape tuple, a list of
                    shape tuples (one per input), or a list of ModelInputSpec objects.
        target: Neutron hardware target string. Default: "imxrt700".
        operators_not_to_delegate: Optional list of op names excluded from NPU delegation.
                                   For example ["aten::convolution"].
        intermediates_dir: Optional directory to dump intermediate compilation artefacts.
        get_quantizer_fn: Optional factory that returns a custom Quantizer. When None,
                          the default NeutronQuantizer is used.
        custom_delegation_options: Optional fine-grained control over which ops are
                                   delegated. Default: CustomDelegationOptions().
        remove_quant_io_ops: If True, remove quantize/dequantize ops at the IO boundary
                             (useful for integer-IO deployments).
        use_quant_state_dict: If False, the post-quantization parameter values are not
                              passed to NeutronPartitioner.
        use_neutron_for_format_conversion: Whether Neutron handles data-format conversion.
        fetch_constants_to_sram: Place constant tensors in SRAM on the target.
        dump_kernel_selection_code: Generate kernel-selection files after compilation.
        use_profiling: Enable execution profiling / ETRecord generation.
    """

    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]]
    target: str = "imxrt700"
    operators_not_to_delegate: list[str] | None = None
    intermediates_dir: str | None = None
    get_quantizer_fn: Callable[[], Quantizer] | None = None
    custom_delegation_options: CustomDelegationOptions | None = None
    remove_quant_io_ops: bool = False
    use_quant_state_dict: bool = True
    use_neutron_for_format_conversion: bool = True
    fetch_constants_to_sram: bool = False
    dump_kernel_selection_code: bool = False
    use_profiling: bool = False


class NXPRecipeProvider(BackendRecipeProvider):

    @property
    def backend_name(self) -> str:
        return NXP_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(NXPRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if recipe_type not in self.get_supported_recipes():
            logging.warning(f"NXP backend: Recipe `{recipe_type}` is not valid.")
            return None

        original_rc = kwargs.get(NEUTRON_RECIPE_CONFIG_KEY)
        if original_rc is None:
            raise KeyError(
                f"NXP backend: create_recipe() requires `{NEUTRON_RECIPE_CONFIG_KEY}=<NeutronRecipeConfig>`."
            )

        rc = cast(NeutronRecipeConfig, deepcopy(original_rc))
        if rc.custom_delegation_options is None:
            rc.custom_delegation_options = CustomDelegationOptions()

        rc.input_spec = to_model_input_spec(rc.input_spec)

        match recipe_type:
            case NXPRecipeType.INT8_PTQ_NEUTRON:
                return self._build_recipe(recipe_type, rc, is_qat=False, delegate=True)
            case NXPRecipeType.INT8_PTQ_NO_DELEGATE:
                return self._build_recipe(recipe_type, rc, is_qat=False, delegate=False)
            case _:
                raise NotImplementedError(
                    f"NXP backend: Recipe `{recipe_type}` is not supported."
                )

    def _build_recipe(
        self,
        recipe_type: NXPRecipeType,
        rc: NeutronRecipeConfig,
        *,
        is_qat: bool,
        delegate: bool,
    ) -> ExportRecipe:
        # is_qat=True is reserved for future QAT support; always False for now.
        if is_qat:
            raise NotImplementedError(
                "NXP recipe with QAT (quantization aware training) is not yet supported."
            )

        neutron_target_spec = NeutronTargetSpec(rc.target)

        if rc.get_quantizer_fn is None:
            rc.get_quantizer_fn = partial(
                get_default_quantizer, neutron_target_spec, is_qat
            )

        quantization_recipe = _build_quantization_recipe(rc)
        compile_spec = generate_neutron_compile_spec(
            rc.target,
            intermediates_dir=rc.intermediates_dir,
            operators_not_to_delegate=rc.operators_not_to_delegate,
            use_neutron_for_format_conversion=rc.use_neutron_for_format_conversion,
            fetch_constants_to_sram=rc.fetch_constants_to_sram,
            dump_kernel_selection_code=rc.dump_kernel_selection_code,
            use_profiling=rc.use_profiling,
        )
        lowering_recipe = _build_lowering_recipe(
            compile_spec, neutron_target_spec, rc, delegate=delegate
        )

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quantization_recipe,
            lowering_recipe=lowering_recipe,
        )


# ---------------------------------------------------------------------------
# Module-level builder helpers
# ---------------------------------------------------------------------------


def _build_quantization_recipe(rc: NeutronRecipeConfig) -> QuantizationRecipe:
    """Build the QuantizationRecipe for PTQ.

    PTQ uses the standard QuantizeStage flow (prepare_pt2e -> calibrate -> convert_pt2e).
    The example_inputs passed to the export session are used directly for calibration.
    Multiple PTQ recipes can be combined with ExportRecipe.combine().
    """
    _quantizer = rc.get_quantizer_fn()

    return QuantizationRecipe(
        quantizers=[_quantizer],
    )


def _build_lowering_recipe(
    compile_spec: list[CompileSpec],
    neutron_target_spec: NeutronTargetSpec,
    rc: NeutronRecipeConfig,
    *,
    delegate: bool,
) -> LoweringRecipe:
    """Build the LoweringRecipe, optionally including NPU delegation."""
    partitioners = _build_partitioners(compile_spec, neutron_target_spec, rc, delegate)
    pre_partitioning_callback = _build_pre_partitioning_callback(rc)
    post_partitioning_transforms = _build_post_partitioning_transforms(rc)

    # The edge pass manager must be wrapped: EdgeTransformAndLowerStage calls
    # edge_transform_passes with (method_name, ep) and expects a PassManager back.
    return LoweringRecipe(
        partitioners=partitioners,
        edge_transform_passes=[NeutronEdgePassManagerWrapper()],
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=core_aten_ops_exception_list,
        ),
        pre_partitioning_callback=pre_partitioning_callback,
        post_partitioning_transforms=post_partitioning_transforms,
    )


def _build_partitioners(
    compile_spec: list[CompileSpec],
    neutron_target_spec: NeutronTargetSpec,
    rc: NeutronRecipeConfig,
    delegate: bool,
) -> list:
    """Create the NeutronPartitioner list. Empty when delegate=False."""
    if not delegate:
        return []
    return [
        NeutronPartitioner(
            compile_spec,
            neutron_target_spec,
            rc.custom_delegation_options,
            preserve_ops=default_preserve_ops,
        )
    ]


def _build_pre_partitioning_callback(rc: NeutronRecipeConfig):
    """Return a callback that assigns the post-quantization state_dict to NeutronPartitioner.

    NeutronPartitioner requires static parameter data. Since the partitioner is instantiated
    during recipe creation (before model data is available), assignment is deferred to a
    callback invoked just before partitioning.
    """
    _use_quant_state_dict = rc.use_quant_state_dict

    def _callback(
        _partitioners: list[Partitioner] | None,
        programs: dict[str, ExportedProgram],
    ) -> None:
        if _partitioners is None:
            return

        if _use_quant_state_dict:
            post_quant_state_dict: dict | None = {}
            for _, program in programs.items():
                post_quant_state_dict.update(program.state_dict)
        else:
            post_quant_state_dict = None

        for _partitioner in _partitioners:
            if isinstance(_partitioner, NeutronPartitioner):
                _partitioner.post_quantization_state_dict = post_quant_state_dict

    return _callback


def _build_post_partitioning_transforms(rc: NeutronRecipeConfig) -> list:
    """Build the list of post-partitioning EdgeProgramManager transforms.

    These mirror what the imperative pipeline did after to_edge_transform_and_lower:
      - RemoveIOQuantOpsPass (optional, when remove_quant_io_ops=True)
      - RemoveAdditionalQDQClustersPass (always applied)
      - handle_kernel_selection side-effect (optional, when dump_kernel_selection_code=True)
    """
    transforms = []

    if rc.remove_quant_io_ops:

        def _remove_io_quant_ops(epm: EdgeProgramManager) -> EdgeProgramManager:
            return epm.transform([RemoveIOQuantOpsPass(edge_program_manager=epm)])

        transforms.append(_remove_io_quant_ops)

    def _remove_additional_qdq_clusters(epm: EdgeProgramManager) -> EdgeProgramManager:
        return epm.transform(
            NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
        )

    transforms.append(_remove_additional_qdq_clusters)

    if rc.dump_kernel_selection_code:

        def _handle_kernel_selection_transform(
            epm: EdgeProgramManager,
        ) -> EdgeProgramManager:
            handle_kernel_selection()
            return epm

        transforms.append(_handle_kernel_selection_transform)

    return transforms
