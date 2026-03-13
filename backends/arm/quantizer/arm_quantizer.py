# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Quantizer for Arm backend
#
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer.quantization_config import (
    QuantizationConfig,
    TOSAQuantizationConfig,
)
from executorch.backends.arm.quantizer.quantizer_support import (
    TOSA_QUANTIZER_SUPPORT_DICT,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    InputNodeFinder,
    ModuleNameNodeFinder,
    ModuleTypeNodeFinder,
    NodeFinder,
    NodeNameNodeFinder,
    NodeTargetNodeFinder,
    OutputNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantizer import (
    PatternQuantizer,
    SharedQspecQuantizer,
)

from executorch.backends.cortex_m.quantizer.quantizer_reporter import (
    QuantizerReporter,
    SUPPORTED_QCONFIGS,
    SUPPORTED_QSPECS,
)
from torch._ops import OpOverload

from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    QuantizationAnnotation,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from executorch.backends.arm.common.arm_compile_spec import (
    ArmCompileSpec,
)  # isort: skip
from executorch.backends.arm._passes.arm_pass_utils import (
    get_cond_while_submodules_nested,
    is_submodule_node,
)
from executorch.backends.arm.vgf import VgfCompileSpec

from executorch.backends.cortex_m.quantizer.quantization_configs import (
    _get_int32_bias_qspec,
    _get_int32_per_channel_bias_qspec,
)
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    ObserverOrFakeQuantizeConstructor,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_module_name_filter,
    QuantizationSpec,
)

from .arm_quantizer_utils import is_annotated, mark_node_as_annotated
from .quantization_annotator import annotate_graph


__all__ = [
    "TOSAQuantizer",
    "EthosUQuantizer",
    "VgfQuantizer",
    "get_symmetric_a16w8_quantization_config",
    "get_symmetric_quantization_config",
]

logger = logging.getLogger(__name__)


@functools.lru_cache
def get_symmetric_quantization_config(
    is_per_channel: bool = True,
    is_qat: bool = False,
    is_dynamic: bool = False,
    act_qmin: int = -128,
    act_qmax: int = 127,
    weight_qmin: int = -127,
    weight_qmax: int = 127,
    eps: float = 2**-16,
) -> QuantizationConfig:
    """Create symmetric quantization config for activations and weights.

    Activations use an affine qscheme; "symmetric" refers to the weight
    quantization qscheme.

    Args:
        is_per_channel (bool): Whether to use per-channel quantization for
            weights.
        is_qat (bool): Whether the configuration targets quantization aware
            training.
        is_dynamic (bool): Whether to generate dynamic activation observers.
        act_qmin (int): Minimum activation quantization value.
        act_qmax (int): Maximum activation quantization value.
        weight_qmin (int): Minimum weight quantization value.
        weight_qmax (int): Maximum weight quantization value.

    Returns:
        QuantizationConfig: Quantization settings for activations, weights, and
        bias.

    """
    extra_args: Dict[str, Any] = {"eps": eps}
    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=act_qmin,
        quant_max=act_qmax,
        qscheme=torch.per_tensor_affine,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args,
        ),
    )

    # Setup quantization config for weights
    weight_qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    weight_observer_or_fake_quant_ctr: ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )

    # Determine the right observer/fake-quant constructor
    if is_qat:
        if is_per_channel:
            weight_observer_or_fake_quant_ctr = FakeQuantize.with_args(
                observer=PerChannelMinMaxObserver,
                quant_min=weight_qmin,
                quant_max=weight_qmax,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                reduce_range=False,
                ch_axis=0,
                **extra_args,
            )
        else:
            # Set plain fake-quant with true min/max
            weight_observer_or_fake_quant_ctr = FakeQuantize.with_args(**extra_args)
    else:
        # PTQ: set min/max observer
        weight_observer_or_fake_quant_ctr = (
            PerChannelMinMaxObserver if is_per_channel else MinMaxObserver
        )
        weight_observer_or_fake_quant_ctr = weight_observer_or_fake_quant_ctr.with_args(
            **extra_args,
        )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
    )

    if is_per_channel:
        bias_quantization_spec = _get_int32_per_channel_bias_qspec
    else:
        bias_quantization_spec = _get_int32_bias_qspec

    if is_dynamic:
        quantization_config = TOSAQuantizationConfig(
            act_quantization_spec,
            None,
            weight_quantization_spec,
            bias_quantization_spec,
        )
    else:
        quantization_config = TOSAQuantizationConfig(
            act_quantization_spec,
            act_quantization_spec,
            weight_quantization_spec,
            bias_quantization_spec,
        )
    return quantization_config


def get_symmetric_a8w4_quantization_config(
    is_per_channel: bool = True, is_qat: bool = True, is_dynamic: bool = False
):
    return get_symmetric_quantization_config(
        is_per_channel, is_qat, is_dynamic, weight_qmin=-7, weight_qmax=7
    )


@functools.lru_cache
def get_symmetric_a16w8_quantization_config(
    is_per_channel: bool = True,
    is_qat: bool = False,
    is_dynamic: bool = False,
    weight_qmin: int = -127,
    weight_qmax: int = 127,
    epsilon: float = 2**-12,
) -> QuantizationConfig:
    """16A8W quantization config: 16-bit activations, 8-bit weights.

    This configuration provides better accuracy than 8A8W while maintaining
    reasonable memory usage through 8-bit weights.

    Args:
        is_per_channel (bool): Whether to use per-channel quantization for
            weights.
        is_qat (bool): Whether this is for quantization aware training.
        is_dynamic (bool): Whether to use dynamic quantization.
        weight_qmin (int): Minimum quantization value for weights.
        weight_qmax (int): Maximum quantization value for weights.
        epsilon (float): Value used to pad observed [qmin, qmax] before initial
            zero-point and scale calculation.

    Returns:
        QuantizationConfig: Configuration with 16-bit activations and 8-bit
        weights.

    """
    extra_args: Dict[str, Any] = {"eps": epsilon}

    # Setup observer/fake-quant for 16-bit activations
    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            # HistogramObserver works well for 16-bit range
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # 16-bit activation quantization spec
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=torch.iinfo(torch.int16).min + 1,  # -32767
        quant_max=torch.iinfo(torch.int16).max,  # 32767
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args,
        ),
    )

    # Instead of reconstructing quantization_config, just clone and update as needed
    # Clone the quantization_config from get_symmetric_quantization_config and update activation spec
    base_config = get_symmetric_quantization_config(
        is_per_channel=is_per_channel,
        is_qat=is_qat,
        is_dynamic=is_dynamic,
    )
    # Replace activation quantization spec with 16-bit version
    if is_dynamic:
        quantization_config = TOSAQuantizationConfig(
            act_quantization_spec,  # 16-bit input activations
            None,
            base_config.weight,  # 8-bit weights from base config
            base_config.bias,  # bias from base config
        )
    else:
        quantization_config = TOSAQuantizationConfig(
            act_quantization_spec,  # 16-bit input activations
            act_quantization_spec,  # 16-bit output activations
            base_config.weight,  # 8-bit weights from base config
            base_config.bias,  # bias from base config
        )
    return quantization_config


# Register supported quantization configs and qspecs in the reporter for human-readable reporting
# MLETORCH-1854: Temporary solution, refactor to automatically register these instead
_symmetric_a8w4_config_per_channel = get_symmetric_a8w4_quantization_config()
_symmetric_a8w8_config_per_channel = get_symmetric_quantization_config()
_symmetric_a16w8_config_per_channel = get_symmetric_a16w8_quantization_config()
_symmetric_a8w4_config_per_tensor = get_symmetric_a8w4_quantization_config(
    is_per_channel=False
)
_symmetric_a8w8_config_per_tensor = get_symmetric_quantization_config(
    is_per_channel=False
)
_symmetric_a16w8_config_per_tensor = get_symmetric_a16w8_quantization_config(
    is_per_channel=False
)
SUPPORTED_QCONFIGS.update(
    {
        _symmetric_a8w8_config_per_channel: f"{__name__}.get_symmetric_quantization_config(is_per_channel=True)",
        _symmetric_a16w8_config_per_channel: f"{__name__}.get_symmetric_a16w8_quantization_config(is_per_channel=True)",
        _symmetric_a8w4_config_per_channel: f"{__name__}.get_symmetric_a8w4_quantization_config(is_per_channel=True)",
        _symmetric_a8w8_config_per_tensor: f"{__name__}.get_symmetric_quantization_config(is_per_channel=False)",
        _symmetric_a16w8_config_per_tensor: f"{__name__}.get_symmetric_a16w8_quantization_config(is_per_channel=False)",
        _symmetric_a8w4_config_per_tensor: f"{__name__}.get_symmetric_a8w4_quantization_config(is_per_channel=False)",
    }
)

SUPPORTED_QSPECS.update(
    {
        _symmetric_a8w4_config_per_channel.get_weight_qspec(): "INT4_PER_CHANNEL_QSPEC",
        _symmetric_a8w8_config_per_channel.get_weight_qspec(): "INT8_PER_CHANNEL_QSPEC",
        _symmetric_a8w8_config_per_tensor.get_weight_qspec(): "INT8_PER_TENSOR_QSPEC",
        _symmetric_a8w4_config_per_tensor.get_weight_qspec(): "INT4_PER_TENSOR_QSPEC",
        _symmetric_a8w8_config_per_tensor.get_input_act_qspec(): "INT8_PER_TENSOR_QSPEC",
        _symmetric_a16w8_config_per_tensor.get_input_act_qspec(): "INT16_PER_TENSOR_QSPEC",
    }
)

NodeFilterType = Callable[[Node], bool]
"""Type for a Node Filter used by annotators.

A Node filter is a function that takes a Node and returns whether the node
should be annotated or not.

"""


def _get_module_type_filter(tp: Callable) -> NodeFilterType:
    """Get the module_type_filter function for a given module type.

    The filter accepts a node and checks if the node comes from a module that
    has a certain module type.

    Args:
        tp (Callable): Module class to match against the graph node metadata.

    Returns:
        NodeFilterType: Predicate that returns True for nodes from the module
        type.

    For example:
        node: linear_op = call_function[...](...)  # type Block -> Sub -> Linear

    >> module_type_filter = _get_module_type_filter(Sub)
    >> print(module_type_filter(node))
    True  # the node is from the submodule `Sub` (same for `Block` and `Linear`)

    """
    tp_str = tp.__module__ + "." + tp.__qualname__

    def module_type_filter(n: Node) -> bool:
        """Return True if the node originates from the target module type."""
        # node_stack example: {
        #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = [t for _, t in nn_module_stack.values()]
        return tp_str in types

    return module_type_filter


def _get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> NodeFilterType:
    """Create a filter that excludes provided module types and names.

    Args:
        tp_list (List[Callable]): Module types to exclude from annotation.
        module_name_list (List[str]): Module names to exclude from annotation.

    Returns:
        NodeFilterType: Filter that returns True when the node does not match
        any provided module type or name.

    """
    module_type_filters = [_get_module_type_filter(tp) for tp in tp_list]
    module_name_list_filters = [get_module_name_filter(m) for m in module_name_list]

    def not_module_type_or_name_filter(n: Node) -> bool:
        """Return True when the node matches none of the blocked filters."""
        return not any(f(n) for f in module_type_filters + module_name_list_filters)

    return not_module_type_or_name_filter


def _for_each_filtered_node(
    model: GraphModule,
    filter_fn: Callable[[Node], bool],
):
    for node in model.graph.nodes:
        if filter_fn(node):
            yield node


class TOSAQuantizer(Quantizer):
    """Manage quantization annotations for TOSA-compatible backends."""

    def __init__(
        self,
        compile_spec_or_tosa_spec,
        use_composable_quantizer: bool = False,
    ) -> None:
        """Create a TOSA quantizer from a TOSA spec or Arm compile spec."""
        self.use_composable_quantizer = use_composable_quantizer
        self.quantizer: _TOSAQuantizerV1 | _TOSAQuantizerV2
        if use_composable_quantizer:
            logger.info(
                "Using composable quantizer implementation in the arm backend. See https://github.com/pytorch/executorch/issues/17701"
            )
            self.quantizer = _TOSAQuantizerV2(compile_spec_or_tosa_spec)
        else:
            logger.info(
                "Using default quantizer in the arm backend. This quantizer is planned to be replaced by the composable quantizer implementation in the future, see https://github.com/pytorch/executorch/issues/17701"
            )
            self.quantizer = _TOSAQuantizerV1(compile_spec_or_tosa_spec)

    @property
    def tosa_spec(self):
        return self.quantizer.tosa_spec

    @property
    def compile_spec(self):
        return self.quantizer.compile_spec

    @property
    def global_config(self):
        return self.quantizer.global_config

    def set_global(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization_config for submodules not matched by other filters.

        Args:
            quantization_config (Optional[QuantizationConfig]): Configuration to
                apply to modules that are not captured by name or type filters.
                ``None`` indicates no quantization.

        """
        self.quantizer.set_global(quantization_config)
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization_config for submodules with a given module type.

        For example, calling set_module_type(Softmax) quantizes supported
        patterns in each Softmax instance with the provided quantization_config.

        Args:
            module_type (Callable): Type whose submodules should use the
                provided quantization configuration.
            quantization_config (Optional[QuantizationConfig]): Configuration to
                apply to submodules of the given type. ``None`` indicates no
                quantization.

        """
        self.quantizer.set_module_type(module_type, quantization_config)
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization_config for submodules with a given module name.

        For example, calling set_module_name("blocks.sub") quantizes supported
        patterns for that submodule with the provided quantization_config.

        Args:
            module_name (str): Fully qualified module name to configure.
            quantization_config (Optional[QuantizationConfig]): Configuration
                applied to the named submodule. ``None`` indicates no
                quantization.

        """
        self.quantizer.set_module_name(module_name, quantization_config)
        return self

    def set_io(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization_config for input and output nodes.

        Args:
            quantization_config (Optional[QuantizationConfig]): Configuration
                describing activation quantization for model inputs and outputs.
                ``None`` indicates no quantization.

        """
        self.quantizer.set_io(quantization_config)
        return self

    def add_quantizer(self, quantizer: Quantizer) -> TOSAQuantizer:
        """Insert a quantizer with highest precedence."""
        if self.use_composable_quantizer:
            return self.quantizer.add_quantizer(quantizer)  # type: ignore[union-attr,return-value]
        raise NotImplementedError(
            "add_quantizer is only supported in the composable quantizer implementation."
        )

    def set_node_finder(
        self, quantization_config: Optional[QuantizationConfig], node_finder: NodeFinder
    ) -> TOSAQuantizer:
        """Set quantization_config for nodes matched by a custom NodeFinder.

        Args:
            quantization_config (Optional[QuantizationConfig]): Configuration
                describing quantization settings for nodes matched by the provided
                NodeFinder. ``None`` indicates no quantization.

        """
        if self.use_composable_quantizer:
            return self.quantizer.set_node_finder(quantization_config, node_finder)  # type: ignore[union-attr,return-value]
        raise NotImplementedError(
            "set_node_finder is only supported in the composable quantizer implementation."
        )

    def set_node_target(
        self, node_target: OpOverload, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization config for a specific operator target."""
        if self.use_composable_quantizer:
            return self.quantizer.set_node_target(node_target, quantization_config)  # type: ignore[union-attr,return-value]
        raise NotImplementedError(
            "set_node_target is only supported in the composable quantizer implementation."
        )

    def set_node_name(
        self, node_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization config for a specific node name."""
        if self.use_composable_quantizer:
            return self.quantizer.set_node_name(node_name, quantization_config)  # type: ignore[union-attr,return-value]
        raise NotImplementedError(
            "set_node_name is only supported in the composable quantizer implementation."
        )

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        """Transform the graph to prepare it for quantization annotation.

        Decomposes all operators where required to get correct quantization parameters.

        Args:
            model (GraphModule): Model whose graph will be transformed.

        Returns:
            GraphModule: Transformed model prepared for annotation.

        """
        return self.quantizer.transform_for_annotation(model)

    def annotate(self, model: GraphModule) -> GraphModule:
        """Annotate the graph with the configured quantization settings.

        Currently only does static quantization annotation.

        Args:
            model (GraphModule): Model to annotate statically.

        Returns:
            GraphModule: Annotated model ready for export.

        """
        return self.quantizer.annotate(model)

    def validate(self, model: GraphModule) -> None:
        """Validate the quantization results. Currently, this includes:
            - Ensure tensor inputs to each operator live on the same device.

        Args:
            model (GraphModule): GraphModule being validated.
        Raises:
            ValueError: If tensor inputs for any operator span more than one
                device.
        """
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue

            devices = set()
            for arg_node in node.all_input_nodes:
                meta_val = arg_node.meta.get("val", None)
                if meta_val is None:
                    continue
                if isinstance(meta_val, (tuple, list)):
                    for tensor in meta_val:
                        devices.add(
                            str(
                                getattr(
                                    tensor,
                                    "device",
                                    f"Could not get device from {tensor}",
                                )
                            )
                        )
                else:
                    devices.add(
                        str(
                            getattr(
                                meta_val,
                                "device",
                                f"Could not get device from {meta_val}",
                            )
                        )
                    )

                if len(devices) > 1:
                    raise ValueError(
                        f"Quantizer detected operator {node.name} with different device inputs: {devices}."
                    )

    def quantize_with_submodules(
        self,
        model: GraphModule,
        calibration_samples: list[tuple],
        is_qat: bool = False,
        fold_quantize: bool = True,
    ):
        """Quantizes a GraphModule in a way such that conditional submodules are
        handled properly.

        Note: torchao's prepare_pt2e and convert_pt2e natively handle
        while_loop body_fn submodules, so we only manually process cond
        branches and while_loop cond_fn here.

        Args:
            model (GraphModule): The model to quantize.
            calibration_samples (list[tuple]): A list of inputs to used to
                calibrate the model during quantization. To properly calibrate a
                model with submodules, at least one sample per code path is
                needed.
            is_qat (bool): Whether to do quantization aware training or not.
            fold_quantize (bool): Enables or disables constant folding when quantization
                is completed.

        Returns:
            GraphModule: The quantized model.

        """
        prepare_fn = prepare_qat_pt2e if is_qat else prepare_pt2e

        prepared = prepare_fn(model, self)
        # Prepare conditional submodules (e.g., if/while bodies)
        # prepare only cond branches and while_loop cond_fn
        for name, submodule, _ in get_cond_while_submodules_nested(
            prepared, apply_quantization=True
        ):
            prepared.set_submodule(name, prepare_fn(submodule, self), strict=True)
            for submodule_node in submodule.graph.nodes:
                if is_submodule_node(submodule_node):
                    for nested_name, nested_sub, _ in get_cond_while_submodules_nested(
                        submodule, apply_quantization=True
                    ):
                        prepared.set_submodule(
                            nested_name, prepare_fn(nested_sub, self), strict=True
                        )

        for inp in calibration_samples:
            prepared(*inp)

        # Prepare conditional submodules (e.g., if/while bodies)
        # convert only cond branches and while_loop cond_fn
        for _, submodule, _ in get_cond_while_submodules_nested(
            prepared, apply_quantization=True
        ):
            converted = convert_pt2e(submodule)
            for submodule_node in submodule.graph.nodes:
                if is_submodule_node(submodule_node):
                    for nested_name, nested_sub, _ in get_cond_while_submodules_nested(
                        submodule, apply_quantization=True
                    ):
                        converted.set_submodule(
                            nested_name, convert_pt2e(nested_sub), strict=True
                        )

        return convert_pt2e(prepared)


class _TOSAQuantizerV1(Quantizer):

    def __init__(
        self, compile_spec_or_tosa_spec: TosaSpecification | ArmCompileSpec
    ) -> None:
        super().__init__()
        self.compile_spec: ArmCompileSpec
        if isinstance(compile_spec_or_tosa_spec, TosaSpecification):
            from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec

            self.compile_spec = TosaCompileSpec(compile_spec_or_tosa_spec)
            self.tosa_spec = self.compile_spec.tosa_spec
        elif isinstance(compile_spec_or_tosa_spec, ArmCompileSpec):
            self.compile_spec = compile_spec_or_tosa_spec
            self.tosa_spec = self.compile_spec.tosa_spec
        else:
            raise TypeError(
                f"TOSAQuantizer constructor expects "
                f"a TosaSpecification or compile_spec list, "
                f"got {type(compile_spec_or_tosa_spec)}"
            )

        self.global_config: Optional[QuantizationConfig] = None
        self.io_config: Optional[QuantizationConfig] = None
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV1:

        self.global_config = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV1:

        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV1:

        # Validate that quantization_config is provided
        self.module_name_config[module_name] = quantization_config
        return self

    def set_io(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV1:
        self.io_config = quantization_config
        return self

    def _set_disallow_tfa_for_nodes(self, model: GraphModule) -> None:
        """Populate `disallow_tfa` metadata for each FX node.

        Transform-for-annotation passes inspect this flag to decide whether they
        may transform a node. Typically, a node should not be transformed in
        case it is not to be quantized, which is relevant for partially
        quantized models.

        """

        # First, set all nodes according to global config
        for node in model.graph.nodes:
            node.meta[DISALLOW_TFA_META_KEY] = self.global_config is None

        # Next, override using module type config to take precedence over global config
        for module_type, config in self.module_type_config.items():
            mod_type_filter = _get_module_type_filter(module_type)
            for node in _for_each_filtered_node(model, mod_type_filter):
                node.meta[DISALLOW_TFA_META_KEY] = config is None

        # Finally, override using module name config to take precedence over both global and type configs
        for module_name, config in self.module_name_config.items():
            mod_name_filter = get_module_name_filter(module_name)
            for node in _for_each_filtered_node(model, mod_name_filter):
                node.meta[DISALLOW_TFA_META_KEY] = config is None

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        self._set_disallow_tfa_for_nodes(model)

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm._passes import ArmPassManager

        pass_manager = ArmPassManager(self.compile_spec)
        return pass_manager.transform_for_annotation_pipeline(graph_module=model)

    def annotate(self, model: GraphModule) -> GraphModule:
        model = self._annotate_for_static_quantization_config(model)
        return model

    def _annotate_all_static_patterns(
        self,
        model: GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> GraphModule:
        """Annotate all static patterns registered for the backend.

        Args:
            model (GraphModule): Model to annotate statically.
            quantization_config (Optional[QuantizationConfig]): Quantization
                specs for input activations, output activations, weights, and
                biases.
            filter_fn (Optional[Callable[[Node], bool]]): Optional node filter
                specifying which nodes to annotate.

        Returns:
            GraphModule: Model populated with quantization annotations.

        """
        # TODO: implement the support for None to be canceling out previous annotations
        if quantization_config is None:
            return model

        annotate_graph(model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(
        self, model: GraphModule
    ) -> GraphModule:
        """Match QuantizationConfigs to modules before annotating patterns.

        Args:
            model (GraphModule): Model whose modules are being matched to
                quantization configs.

        Returns:
            GraphModule: Annotated model after applying configured filters.

        """
        if self.io_config:
            self._annotate_io(model, self.io_config)

        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, get_module_name_filter(module_name)
            )

        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_type_filter(module_type)
            )

        self._annotate_all_static_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )

        return model

    def _annotate_io(
        self,
        model: GraphModule,
        quantization_config: QuantizationConfig,
    ):
        """Annotate graph inputs and outputs with the provided configuration.

        Args:
            model (GraphModule): GraphModule being annotated.
            quantization_config (QuantizationConfig): Activation qspecs to apply
                to IO nodes.

        """
        for node in model.graph.nodes:
            if is_annotated(node):
                continue
            if node.op == "placeholder" and len(node.users) > 0:
                annotate_output_qspec(
                    node,
                    quantization_config.get_output_act_qspec(),
                )
                mark_node_as_annotated(node)
            if node.op == "output":
                for parent in node.all_input_nodes:
                    annotate_input_qspec_map(
                        node, parent, quantization_config.get_input_act_qspec()
                    )
                mark_node_as_annotated(node)

    def validate(self, model: GraphModule) -> None:
        # Validation is handled by TOSAQuantizer.validate; keep no-op for
        # Quantizer interface compatibility.
        return None


class _TOSAQuantizerV2(ComposableQuantizer):

    def __init__(
        self, compile_spec_or_tosa_spec: TosaSpecification | ArmCompileSpec
    ) -> None:
        self.compile_spec: ArmCompileSpec
        if isinstance(compile_spec_or_tosa_spec, TosaSpecification):
            from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec

            self.compile_spec = TosaCompileSpec(compile_spec_or_tosa_spec)
            self.tosa_spec = self.compile_spec.tosa_spec
        elif isinstance(compile_spec_or_tosa_spec, ArmCompileSpec):
            self.compile_spec = compile_spec_or_tosa_spec
            self.tosa_spec = self.compile_spec.tosa_spec
        else:
            raise TypeError(
                f"TOSAQuantizer constructor expects "
                f"a TosaSpecification or compile_spec list, "
                f"got {type(compile_spec_or_tosa_spec)}"
            )

        self.pattern_matcher = PatternMatcher(TOSA_QUANTIZER_SUPPORT_DICT)
        self.shared_qspec_quantizer = SharedQspecQuantizer()
        self.global_quantizer: Quantizer | None = None
        self.global_config: Optional[QuantizationConfig] = None
        self._quantizers: List[Quantizer] = []
        self._graph_annotations: dict[Node, QuantizationAnnotation] = {}

    @property
    def quantizers(self) -> List[Quantizer]:
        """Returns the configured quantizers in order of precedence, ensuring
        the global config and shared_qspec_quantizer are applied last.

        The returned list is a shallow copy; quantizer instances are shared.

        """
        quantizers = self._quantizers.copy()
        if self.global_quantizer is not None:
            quantizers.append(self.global_quantizer)
        quantizers.append(self.shared_qspec_quantizer)

        return quantizers

    @quantizers.setter
    def quantizers(self, value: List[Quantizer]) -> None:
        """Override of quantizers setter to allow for dynamic updating of
        quantizers without accessing self._quantizers.
        """
        self._quantizers = value

    def annotate(self, model):
        reporter = QuantizerReporter(self.quantizers, "FINAL QUANTIZATION REPORT")
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        return model

    def _remove_annotations(self, model: GraphModule) -> GraphModule:
        for node in model.graph.nodes:
            if Q_ANNOTATION_KEY in node.meta:
                del node.meta[Q_ANNOTATION_KEY]
            if ArmAnnotationInfo.CUSTOM_META_KEY in node.meta:
                del node.meta[ArmAnnotationInfo.CUSTOM_META_KEY]
            if DISALLOW_TFA_META_KEY in node.meta:
                del node.meta[DISALLOW_TFA_META_KEY]
            if PatternMatcher.Q_PATTERN_MATCHED_KEY in node.meta:
                del node.meta[PatternMatcher.Q_PATTERN_MATCHED_KEY]

        # Clear quantizer internal annotation tracking
        self._graph_annotations.clear()

        return model

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        # Transform_for_annotation should only decompose ops if quantized, which is
        # indicated either by node.meta['DISALLOW_TFA_META_KEY']==False or no such key
        # existing in the dict. This means that ops are assumed to be quantized by
        # default and we need to explicitly annotate all non-quantized nodes with
        # DISALLOW_TFA_META_KEY=True before calling the pass manager.

        # For _TOSAQuantizerV2 there is no simple filter which directly finds unquantized
        # nodes since nodes can be annotated by any quantizer. Instead, self.annotate is
        # run to set DISALLOW_TFA_META_KEY for quantized nodes and all nodes missing
        # this key afterwards are set to DISALLOW_TFA_META_KEY=True.
        reporter = QuantizerReporter(
            self.quantizers, "PRE-TRANSFORM_FOR_ANNOTATION QUANTIZATION REPORT"  # type: ignore[arg-type]
        )
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        for node in model.graph.nodes:
            if DISALLOW_TFA_META_KEY not in node.meta:
                node.meta[DISALLOW_TFA_META_KEY] = True

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm._passes import ArmPassManager

        pass_manager = ArmPassManager(self.compile_spec)
        transformed_model = pass_manager.transform_for_annotation_pipeline(model)

        # Remove the temporary annotations
        return self._remove_annotations(transformed_model)

    def add_quantizer(self, quantizer: Quantizer) -> _TOSAQuantizerV2:
        """Insert a quantizer with highest precedence."""
        self._quantizers.insert(0, quantizer)
        return self

    def set_node_finder(
        self, quantization_config: Optional[QuantizationConfig], node_finder: NodeFinder
    ) -> _TOSAQuantizerV2:
        """Add a quantizer targeting nodes found by the provided finder.

        ``None`` indicates no quantization for matched nodes.

        """
        quantizer = PatternQuantizer(
            quantization_config, node_finder, self.pattern_matcher
        )
        self.add_quantizer(quantizer)
        return self

    def set_global(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set the default quantization config for all nodes.

        ``None`` indicates no quantization.

        """
        node_finder = GlobalNodeFinder()
        self.global_quantizer = PatternQuantizer(
            quantization_config, node_finder, self.pattern_matcher
        )
        self.global_config = quantization_config
        return self

    def set_node_target(
        self, node_target: OpOverload, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set quantization config for a specific operator target."""
        node_finder = NodeTargetNodeFinder(node_target)
        self.set_node_finder(quantization_config, node_finder)
        return self

    def set_node_name(
        self, node_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set quantization config for a specific node name."""
        node_finder = NodeNameNodeFinder(node_name)
        self.set_node_finder(quantization_config, node_finder)
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set quantization config for nodes originating from a module type."""
        node_finder = ModuleTypeNodeFinder(module_type)
        self.set_node_finder(quantization_config, node_finder)
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set quantization config for nodes originating from a module name."""
        node_finder = ModuleNameNodeFinder(module_name)
        self.set_node_finder(quantization_config, node_finder)
        return self

    def set_io(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> _TOSAQuantizerV2:
        """Set quantization_config for input and output nodes.

        Args:
            quantization_config (Optional[QuantizationConfig]): Configuration
                describing activation quantization for model inputs and outputs.
                ``None`` indicates no quantization.

        """
        input_finder = InputNodeFinder()
        output_finder = OutputNodeFinder()
        self.set_node_finder(quantization_config, input_finder)
        self.set_node_finder(quantization_config, output_finder)
        return self


class EthosUQuantizer(TOSAQuantizer):
    """Quantizer supported by the Arm Ethos-U backend.

    Args:
        compile_spec (EthosUCompileSpec): Backend compile specification for
            Ethos-U targets.
        use_composable_quantizer (bool): Whether to use the composable quantizer implementation. See https://github.com/pytorch/executorch/issues/17701" for details.

    """

    def __init__(
        self,
        compile_spec: EthosUCompileSpec,
        use_composable_quantizer: bool = False,
    ) -> None:
        super().__init__(compile_spec, use_composable_quantizer)


class VgfQuantizer(TOSAQuantizer):
    """Quantizer supported by the Arm Vgf backend.

    Args:
        compile_spec (VgfCompileSpec): Backend compile specification for Vgf
            targets.
        use_composable_quantizer (bool): Whether to use the composable quantizer implementation. See https://github.com/pytorch/executorch/issues/17701" for details.

    """

    def __init__(
        self,
        compile_spec: VgfCompileSpec,
        use_composable_quantizer: bool = False,
    ) -> None:
        super().__init__(compile_spec, use_composable_quantizer)
