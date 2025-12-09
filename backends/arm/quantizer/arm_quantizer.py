# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Quantizer for Arm backend
#

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec

from executorch.backends.arm.quantizer import QuantizationConfig
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.common.arm_compile_spec import (
    ArmCompileSpec,
)  # isort: skip
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.exir.graph_module import get_cond_while_submodules

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
    Quantizer,
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


@functools.lru_cache
def get_symmetric_quantization_config(
    is_per_channel: bool = True,
    is_qat: bool = False,
    is_dynamic: bool = False,
    act_qmin: int = -128,
    act_qmax: int = 127,
    weight_qmin: int = -127,
    weight_qmax: int = 127,
) -> QuantizationConfig:
    """Create symmetric quantization config for activations and weights.

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
    extra_args: Dict[str, Any] = {"eps": 2**-12}
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

    bias_quantization_spec = None
    if is_dynamic:
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            None,
            weight_quantization_spec,
            bias_quantization_spec,
        )
    else:
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            act_quantization_spec,
            weight_quantization_spec,
            bias_quantization_spec,
        )
    return quantization_config


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
        quantization_config = QuantizationConfig(
            act_quantization_spec,  # 16-bit input activations
            None,
            base_config.weight,  # 8-bit weights from base config
            None,
        )
    else:
        quantization_config = QuantizationConfig(
            act_quantization_spec,  # 16-bit input activations
            act_quantization_spec,  # 16-bit output activations
            base_config.weight,  # 8-bit weights from base config
            None,
        )
    return quantization_config


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


class TOSAQuantizer(Quantizer):
    """Manage quantization annotations for TOSA-compatible backends."""

    def __init__(
        self, compile_spec_or_tosa_spec: TosaSpecification | ArmCompileSpec
    ) -> None:

        super().__init__()
        if isinstance(compile_spec_or_tosa_spec, TosaSpecification):
            self.tosa_spec = compile_spec_or_tosa_spec
            self.compile_spec = None
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

    def set_global(self, quantization_config: QuantizationConfig) -> TOSAQuantizer:
        """Set quantization_config for submodules not matched by other filters.

        Args:
            quantization_config (QuantizationConfig): Configuration to apply to
                modules that are not captured by name or type filters.

        """
        self.global_config = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ) -> TOSAQuantizer:
        """Set quantization_config for submodules with a given module type.

        For example, calling set_module_type(Sub) quantizes supported patterns
        in each Sub instance with the provided quantization_config.

        Args:
            module_type (Callable): Type whose submodules should use the
                provided quantization configuration.
            quantization_config (QuantizationConfig): Configuration to apply to
                submodules of the given type.

        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> TOSAQuantizer:
        """Set quantization_config for submodules with a given module name.

        For example, calling set_module_name("blocks.sub") quantizes supported
        patterns for that submodule with the provided quantization_config.

        Args:
            module_name (str): Fully qualified module name to configure.
            quantization_config (QuantizationConfig): Configuration applied to
                the named submodule.

        """
        # Validate that quantization_config is provided
        if quantization_config is None:
            raise ValueError("quantization_config == None is not supported yet")
        self.module_name_config[module_name] = quantization_config
        return self

    def set_io(self, quantization_config: QuantizationConfig) -> TOSAQuantizer:
        """Set quantization_config for input and output nodes.

        Args:
            quantization_config (QuantizationConfig): Configuration describing
                activation quantization for model inputs and outputs.

        """
        self.io_config = quantization_config
        return self

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        """Transform the graph to prepare it for quantization annotation.

        Currently transforms scalar values to tensor attributes.

        Args:
            model (GraphModule): Model whose graph will be transformed.

        Returns:
            GraphModule: Transformed model prepared for annotation.

        """
        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm._passes import ArmPassManager

        return ArmPassManager(self.tosa_spec).transform_for_annotation_pipeline(
            graph_module=model
        )

    def annotate(self, model: GraphModule) -> GraphModule:
        """Annotate the graph with the configured quantization settings.

        Currently only does static quantization annotation.

        Args:
            model (GraphModule): Model to annotate statically.

        Returns:
            GraphModule: Annotated model ready for export.

        """
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
        """TODO: Implement validation of annotated graph for TOSA backend."""
        pass

    def quantize_with_submodules(
        self,
        model: GraphModule,
        calibration_samples: list[tuple],
        is_qat: bool = False,
    ):
        """Quantizes a GraphModule in a way such that conditional submodules are handled properly.

        Args:
            model (GraphModule): The model to quantize.
            calibration_samples (list[tuple]): A list of inputs to used to
                calibrate the model during quantization. To properly calibrate a
                model with submodules, at least one sample per code path is
                needed.
            is_qat (bool): Whether to do quantization aware training or not.

        Returns:
            GraphModule: The quantized model.

        """
        prepare_fn = prepare_qat_pt2e if is_qat else prepare_pt2e

        prepared = prepare_fn(model, self)
        for name, submodule, _ in get_cond_while_submodules(prepared):
            prepared.set_submodule(name, prepare_fn(submodule, self), strict=True)
        for inp in calibration_samples:
            prepared(*inp)

        for name, submodule, _ in get_cond_while_submodules(prepared):
            prepared.set_submodule(name, convert_pt2e(submodule), strict=True)
        converted = convert_pt2e(prepared)
        return converted


class EthosUQuantizer(TOSAQuantizer):
    """Quantizer supported by the Arm Ethos-U backend.

    Args:
        compile_spec (EthosUCompileSpec): Backend compile specification for
            Ethos-U targets.

    """

    def __init__(self, compile_spec: EthosUCompileSpec) -> None:
        super().__init__(compile_spec)


class VgfQuantizer(TOSAQuantizer):
    """Quantizer supported by the Arm Vgf backend.

    Args:
        compile_spec (VgfCompileSpec): Backend compile specification for Vgf
            targets.

    """

    def __init__(self, compile_spec: VgfCompileSpec) -> None:
        super().__init__(compile_spec)
