# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast, Iterable, Optional, Sequence

import torch

from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    FuseBatchNormWithLinearPass,
)
from executorch.backends.nxp.aten_passes.simulated_linear_bn_fusion_passes import (
    AddSimulatedLinearBatchNormFusionQATPass,
    RemoveSimulatedLinearBatchNormFusionQATPass,
)
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
from executorch.backends.nxp.quantizer.utils import (
    _replace_histogram_observers_for_integer_inputs,
)
from executorch.backends.nxp.recipes.nxp_recipe_types import NXP_BACKEND, NXPRecipeType
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_default_quantizer,
    handle_kernel_selection,
    ModelInputSpec,
    to_model_input_spec,
)
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    QuantizeFusedConvBnBiasAtenPass,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExportedProgram,
)

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

    Parameters that vary the *type* of export (delegate vs no-delegate, PTQ vs QAT)
    are expressed by choosing a different NXPRecipeType rather than by flags here.

    Attributes:
        input_spec: Model input description. Accepts a single shape tuple, a list of
                    shape tuples (one per input), or a list of ModelInputSpec objects.
        target: Neutron hardware target string. Default: "imxrt700".
        operators_not_to_delegate: Optional list of op names excluded from NPU delegation.
                                   For example ["aten::convolution"].
        intermediates_dir: Optional directory to dump intermediate compilation artifacts.
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
        use_profiling: Enable Neutron execution profiling. IMPORTANT: To also generate an
                       ETRecord, pass generate_etrecord=True to export() separately.
        train_fn: Training function required for QAT recipe types (INT8_QAT_NEUTRON and
                  INT8_QAT_NO_DELEGATE). Receives the prepared GraphModule and must
                  perform the training loop. Ignored for PTQ recipe types.
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
    train_fn: Callable[["torch.fx.GraphModule"], None] | None = None


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
        if not isinstance(original_rc, NeutronRecipeConfig):
            raise TypeError(
                f"NXP backend: `{NEUTRON_RECIPE_CONFIG_KEY}` must be a NeutronRecipeConfig, "
                f"got {type(original_rc).__name__}."
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
            case NXPRecipeType.INT8_QAT_NEUTRON:
                return self._build_recipe(recipe_type, rc, is_qat=True, delegate=True)
            case NXPRecipeType.INT8_QAT_NO_DELEGATE:
                return self._build_recipe(recipe_type, rc, is_qat=True, delegate=False)
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
        if is_qat and rc.train_fn is None:
            raise ValueError(
                f"NXP backend: Recipe `{recipe_type}` requires `train_fn` to be set in "
                f"NeutronRecipeConfig. Provide a callable that trains the prepared model."
            )

        neutron_target_spec = NeutronTargetSpec(rc.target)

        if rc.get_quantizer_fn is None:
            rc.get_quantizer_fn = partial(
                get_default_quantizer, neutron_target_spec, is_qat
            )

        quantization_recipe = _build_quantization_recipe(rc, is_qat)
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
            executorch_backend_config=ExecutorchBackendConfig(
                extract_delegate_segments=False
            ),
        )


# ---------------------------------------------------------------------------
# Pass wrappers
# ---------------------------------------------------------------------------
# ExirPassBase subclasses return a PassResult with a .graph_module attribute,
# but QuantizeStage._apply_passes expects callable(GraphModule) -> GraphModule.
# These thin wrappers bridge the two conventions.


def _wrap_exir_pass(pass_cls, *args, **kwargs):
    """Return a callable(GraphModule) -> GraphModule wrapping an ExirPass instance."""
    _pass_instance = pass_cls(*args, **kwargs)

    def _wrapped(m):
        return _pass_instance(m).graph_module

    _wrapped.__qualname__ = f"_wrap_exir_pass({pass_cls.__name__})"
    return _wrapped


def _histogram_observer_fix_pass(m):
    """Callable(GraphModule) -> GraphModule that replaces HistogramObserver for integer inputs."""
    _replace_histogram_observers_for_integer_inputs(m)
    return m


# ---------------------------------------------------------------------------
# Module-level builder helpers
# ---------------------------------------------------------------------------


def _build_quantization_recipe(
    rc: NeutronRecipeConfig, is_qat: bool
) -> QuantizationRecipe:
    """Build the QuantizationRecipe for PTQ or QAT.

    PTQ uses the standard QuantizeStage flow (prepare_pt2e -> calibrate -> convert_pt2e).
    QAT uses the QAT flow (prepare_qat_pt2e -> BN-fusion passes -> train_fn -> convert_pt2e).

    The NXP-specific passes are injected via the QuantizationRecipe hook lists so
    that QuantizeStage executes them in the correct order.
    """
    _quantizer = rc.get_quantizer_fn()

    # post_prepare_passes: always fix HistogramObserver for non-float inputs.
    # For QAT, also insert the simulated linear-BN fusion before training so
    # fake-quantize nodes see fused weights during the training loop.
    post_prepare: list[Callable] = []
    if is_qat:
        post_prepare.append(_wrap_exir_pass(AddSimulatedLinearBatchNormFusionQATPass))
    post_prepare.append(_histogram_observer_fix_pass)

    if is_qat:
        # pre_convert_passes: tear down the simulated fusion and fold BN into
        # the linear weights before convert_pt2e.
        pre_convert: list[Callable] = [
            _wrap_exir_pass(RemoveSimulatedLinearBatchNormFusionQATPass),
            _wrap_exir_pass(FuseBatchNormWithLinearPass),
        ]

        # post_convert_passes: fix up quantization parameters for fused conv+BN
        # bias nodes after convert_pt2e has inserted the quantize/dequantize ops.
        post_convert: list[Callable] = [
            _wrap_exir_pass(
                QuantizeFusedConvBnBiasAtenPass,
                default_zero_bias=False,
                symmetric_quant=True,
            )
        ]

        return QuantizationRecipe(
            quantizers=[_quantizer],
            is_qat=True,
            train_fn=rc.train_fn,
            post_prepare_passes=post_prepare,
            pre_convert_passes=pre_convert,
            post_convert_passes=post_convert,
        )
    else:
        return QuantizationRecipe(
            quantizers=[_quantizer],
            post_prepare_passes=post_prepare,
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
    edge_manager_transform_passes = _build_edge_manager_transform_passes(rc)

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
        edge_manager_transform_passes=edge_manager_transform_passes,
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
        if not _partitioners:
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


def _build_edge_manager_transform_passes(rc: NeutronRecipeConfig) -> list:
    """Build edge_manager_transform_passes for the post-partitioning graph cleanup.

    These run in EdgeProgramManagerTransformStage, after to_edge_transform_and_lower:
      - RemoveIOQuantOpsPass (optional, when remove_quant_io_ops=True)
      - RemoveAdditionalQDQClustersPass (always applied)
      - handle_kernel_selection side-effect (optional, when dump_kernel_selection_code=True)

    Each callable receives EdgeProgramManager and returns passes for epm.transform(),
    or an empty list when no graph transformation is needed (side-effect only).
    """
    passes = []

    if rc.remove_quant_io_ops:

        def _remove_io_quant_ops(epm: EdgeProgramManager) -> list:
            return [RemoveIOQuantOpsPass(edge_program_manager=epm)]

        passes.append(_remove_io_quant_ops)

    def _remove_additional_qdq_clusters(
        epm: EdgeProgramManager,
    ) -> NeutronEdgePassManager:
        return NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])

    passes.append(_remove_additional_qdq_clusters)

    if rc.dump_kernel_selection_code:

        def _handle_kernel_selection_side_effect(_epm: EdgeProgramManager) -> list:
            # Side-effect only: write kernel-selection files. No graph transform needed.
            handle_kernel_selection()
            return []

        passes.append(_handle_kernel_selection_side_effect)

    return passes
