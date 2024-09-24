# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Quantizer for Arm backend
#

from __future__ import annotations

import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    convert_scalars_to_attrs,
    mark_nodes_as_annotated,
    propagate_annotation,
)
from executorch.backends.arm.quantizer.quantization_annotation import (
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import GraphModule, Node

__all__ = [
    "ArmQuantizer",
    "get_symmetric_quantization_config",
]


def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        # Both conv and linear should be able to handle relu + hardtanh fusion since
        # those are clamp ops
        "conv2d": [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
        ],
        "linear": [[torch.nn.Linear], [F.linear]],
        "add": [[torch.add]],
        "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
        "adaptive_avg_pool2d": [
            [torch.nn.AdaptiveAvgPool2d],
            [F.adaptive_avg_pool2d],
        ],
        "mul": [torch.mul],
        "sub": [[torch.sub]],
    }
    return copy.deepcopy(supported_operators)


def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_symmetric_quantization_config(),
        get_symmetric_quantization_config(is_per_channel=True),
    ]:
        ops = _supported_symmetric_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_symmetric_quantization_config(
    is_per_channel: bool = False,
    is_qat: bool = False,
    is_dynamic: bool = False,
    act_qmin: int = -128,
    act_qmax: int = 127,
    weight_qmin: int = -127,
    weight_qmax: int = 127,
):
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
    weight_qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )
    if is_qat:
        # TODO: qat + per channel?
        weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    elif is_per_channel:
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            extra_args["observer"] = MovingAverageMinMaxObserver
        else:
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
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


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_symmetric_config_and_operators()


NodeFilterType = Callable[[Node], bool]
"""Type for a Node Filter used by annotators. A Node filter is a function that takes
    a Node and returns whether the node should be annotated or not.
"""


def _get_module_name_filter(module_name: str) -> NodeFilterType:
    """Get the module_name_filter function for a given module name, the filter accepts
    a node and checks if the node comes from a module that has certain module name

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with name blocks.sub.linear1

    >> module_name_filter = _get_module_name_filter("blocks.sub")
    >> print(module_name_filter(node))
    True  # the node is from "blocks.sub" based on the fully qualified name "blocks.sub.linear1"
    """

    name_start = len("L['self'].")

    def module_name_filter(n: Node) -> bool:
        # node_stack example: {
        #    'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #    'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        # get_attr nodes doesn't have nn_module_stack?
        nn_module_stack = n.meta.get("nn_module_stack", {})
        names = [name[name_start:] for name, _ in nn_module_stack.values()]
        return module_name in names

    return module_name_filter


def _get_module_type_filter(tp: Callable) -> NodeFilterType:
    """Get the module_type_filter function for a given module type, the filter accepts
    a node and checks if the node comes from a module that has certain module type

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with type Block -> Sub -> Linear


    >> module_type_filter = _get_module_type_filter(Sub)  # submodule with type `Sub`, under the `Block` submodule
    >> print(module_type_filter(node))
    True  # the node is from the submodule `Sub` (same for `Block` and `Linear` as well)
    """

    def module_type_filter(n: Node) -> bool:
        # node_stack example: {
        #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = [t for _, t in nn_module_stack.values()]
        return tp in types

    return module_type_filter


def _get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> NodeFilterType:
    module_type_filters = [_get_module_type_filter(tp) for tp in tp_list]
    module_name_list_filters = [_get_module_name_filter(m) for m in module_name_list]

    def not_module_type_or_name_filter(n: Node) -> bool:
        return not any(f(n) for f in module_type_filters + module_name_list_filters)

    return not_module_type_or_name_filter


class ArmQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()

    # A list of supported static quantization annotators, in order of application.
    # For example, fusions come before singular ops.
    # The name must match the name used when registering the annotator.
    STATIC_ANNOTATION_ORDER = [
        "linear",
        "conv",
        "adaptive_avg_pool2d",
        "max_pool2d",
        "add",
        "sub",
        "mul",
        "sigmoid",
        "mm",
        "cat",
        "one_to_one",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.io_config: Optional[QuantizationConfig] = None
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(self, quantization_config: QuantizationConfig) -> ArmQuantizer:
        """Set quantization_config for submodules that are not already annotated by name or type filters."""
        self.global_config = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ) -> ArmQuantizer:
        """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ) -> ArmQuantizer:
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`
        """
        assert (
            quantization_config is not None
        ), " quantization_config == None is not supported yet"
        self.module_name_config[module_name] = quantization_config
        return self

    def set_io(self, quantization_config):
        """Set quantization_config for input and output nodes."""
        self.io_config = quantization_config
        return self

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        """An initial pass for transforming the graph to prepare it for annotation.
        Currently transforms scalar values to tensor attributes.
        """
        return convert_scalars_to_attrs(model)

    def annotate(self, model: GraphModule) -> GraphModule:
        """Performs the quantization annotation on the graph.
            Currently only does static quantization annotation.
        Args:
            model: The model to annotate statically.
        Returns:
            The annotated model.
        """
        model = self._annotate_for_static_quantization_config(model)
        propagate_annotation(model)
        return model

    def _annotate_all_static_patterns(
        self,
        model: GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> GraphModule:
        """Loops over all STATIC_OPS and runs the corresponding registred annotator.
        Args:
            model: The model to annotate statically.
            quantization_config: Specifices the QuantizationSpecs for the model's
                input activations, output activations, weights and biases.
            filter_fn: An optional filter function that takes a node and returns whether the node should be annotated.
        Returns:
            The annotated model.
        """
        # TODO: implement the support for None to be canceling out previous annotations
        if quantization_config is None:
            return model

        for op in self.STATIC_ANNOTATION_ORDER:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(
        self, model: GraphModule
    ) -> GraphModule:
        """Matches the correct QuantizationConfig with the correct module using a filter
        when running _annotate_all_static_patterns.
        """
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_name_filter(module_name)
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

        if self.io_config:
            self._annotate_io(model, self.io_config)

        return model

    def _annotate_io(
        self,
        model: GraphModule,
        quantization_config: QuantizationConfig,
    ):
        for node in model.graph.nodes:
            if arm_quantizer_utils.is_annotated(node):
                continue
            if node.op == "placeholder" and len(node.users) > 0:
                _annotate_output_qspec(
                    node,
                    quantization_config.get_output_act_qspec(),
                )
                mark_nodes_as_annotated([node])
            if node.op == "output":
                parent = node.all_input_nodes[0]
                _annotate_input_qspec_map(
                    node, parent, quantization_config.get_input_act_qspec()
                )
                mark_nodes_as_annotated([node])

    def validate(self, model: GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []
