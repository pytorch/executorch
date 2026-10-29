# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from functools import partial

import pytest
import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.quantization_annotator import annotate_graph
from executorch.backends.arm.quantizer.quantization_config import (
    get_int8_matmul_accumulator_qspec,
    QuantizationConfig,
    TOSAQuantizationConfig,
    VGFQuantizationConfig,
)
from executorch.backends.cortex_m.quantizer.node_finders import NodeTargetNodeFinder
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
)


@pytest.mark.parametrize(
    "config_type", [QuantizationConfig, TOSAQuantizationConfig, VGFQuantizationConfig]
)
def test_output_qspec_factory_is_node_specific_and_does_not_mutate_config(config_type):
    activation = replace(
        get_symmetric_quantization_config().input_activation,
        qscheme=torch.per_tensor_symmetric,
    )
    graph = torch.fx.Graph()
    lhs, rhs = graph.placeholder("lhs"), graph.placeholder("rhs")
    lhs.meta["val"] = torch.empty(1, 2, 3, device="meta")
    first = graph.call_function(torch.ops.aten.bmm.default, (lhs, rhs))
    second = graph.call_function(torch.ops.aten.bmm.default, (lhs, rhs))
    factory = partial(get_int8_matmul_accumulator_qspec, input_qspec=activation)
    config = config_type(activation, factory, None, None)

    for node in (first, second, first):
        qspec = config.get_output_act_qspec(node)
        assert isinstance(qspec, DerivedQuantizationSpec)
        assert qspec.derived_from == [(lhs, node), (rhs, node)]
        assert config.output_activation is factory
    with pytest.raises(ValueError, match="factory requires a node"):
        config.get_output_act_qspec()


@pytest.mark.parametrize(
    "target",
    [
        torch.ops.aten.grid_sampler.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.to.dtype,
    ],
)
def test_explicit_output_factory_overrides_tosa_output_rules(target):
    graph = torch.fx.Graph()
    operand = graph.placeholder("input")
    node = graph.call_function(target, (operand,))
    config = get_symmetric_quantization_config()
    output = config.output_activation
    visited = []

    def factory(current_node):
        visited.append(current_node)
        return output

    config = replace(config, output_activation=factory)
    assert config.get_output_act_qspec(node) is output
    assert visited == [node]


@pytest.mark.parametrize("use_factory", [False, True])
def test_output_spec_or_factory_can_disable_quantization(use_factory):
    config = replace(
        get_symmetric_quantization_config(),
        output_activation=(lambda node: None) if use_factory else None,
    )
    node = torch.fx.Graph().placeholder("input")
    assert config.get_output_act_qspec(node) is None
    if not use_factory:
        assert config.get_output_act_qspec() is None


def test_concrete_output_spec_remains_compatible_without_node():
    config = get_symmetric_quantization_config()
    assert config.get_output_act_qspec() is config.output_activation


def test_annotation_accepts_getter_only_config():
    config = get_symmetric_quantization_config()

    class GetterOnlyConfig:
        def get_input_act_qspec(self):
            return config.get_input_act_qspec()

        def get_output_act_qspec(self):
            return config.get_output_act_qspec()

        def get_weight_qspec(self):
            return config.get_weight_qspec()

        def get_bias_qspec(self, node):
            return config.get_bias_qspec(node)

    class Add(torch.nn.Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    model = torch.export.export(Add(), inputs, strict=True).graph_module
    add = next(
        node for node in model.graph.nodes if node.target == torch.ops.aten.add.Tensor
    )
    lhs, rhs = add.args

    annotate_graph(model, GetterOnlyConfig())

    annotation = add.meta["quantization_annotation"]
    assert annotation.input_qspec_map == {
        lhs: config.input_activation,
        rhs: config.input_activation,
    }
    assert annotation.output_qspec is config.output_activation


def test_concrete_output_spec_preserves_tosa_output_rules():
    graph = torch.fx.Graph()
    operand = graph.placeholder("input")
    sampler = graph.call_function(torch.ops.aten.grid_sampler.default, (operand,))
    mean = graph.call_function(torch.ops.aten.mean.dim, (operand,))
    config = get_symmetric_quantization_config()
    assert isinstance(
        config.get_output_act_qspec(sampler), FixedQParamsQuantizationSpec
    )
    shared = config.get_output_act_qspec(mean)
    assert isinstance(shared, SharedQuantizationSpec)
    assert shared.edge_or_node == (operand, mean)


@pytest.mark.parametrize("invalid", ["object", "per_channel"])
def test_factory_result_is_validated(invalid):
    config = get_symmetric_quantization_config()
    output = (
        object()
        if invalid == "object"
        else replace(config.output_activation, qscheme=torch.per_channel_symmetric)
    )
    config = replace(config, output_activation=lambda node: output)
    with pytest.raises(ValueError, match="Unsupported quantization_spec"):
        config.get_output_act_qspec(torch.fx.Graph().placeholder("input"))


@pytest.mark.parametrize("composable", [False, True])
def test_matmul_quantization_uses_output_factory(composable):
    class Matmul(torch.nn.Module):
        def forward(self, lhs, rhs):
            return torch.bmm(lhs, rhs)

    generator = torch.Generator().manual_seed(27)
    inputs = (
        torch.randn(1, 2, 3, generator=generator),
        torch.randn(1, 3, 4, generator=generator),
    )
    model = torch.export.export(Matmul(), inputs, strict=True).module()
    config = get_symmetric_quantization_config()
    activation = replace(config.input_activation, qscheme=torch.per_tensor_symmetric)
    config = replace(
        config,
        input_activation=activation,
        output_activation=partial(
            get_int8_matmul_accumulator_qspec, input_qspec=activation
        ),
    )
    quantizer = EthosUQuantizer(
        EthosUCompileSpec("ethos-u85-256"), use_composable_quantizer=composable
    )
    if composable:
        quantizer.set_node_finder(
            config, NodeTargetNodeFinder(torch.ops.aten.bmm.default)
        )
    else:
        quantizer.set_global(config)
    prepared = prepare_pt2e(model, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    matmul = next(
        node
        for node in converted.graph.nodes
        if node.target == torch.ops.aten.bmm.default
    )
    assert isinstance(
        matmul.meta["quantization_annotation"].output_qspec, DerivedQuantizationSpec
    )
    torch.testing.assert_close(
        converted(*inputs), Matmul()(*inputs), atol=0.05, rtol=0.05
    )


@pytest.mark.parametrize("composable", [False, True])
def test_io_output_factory_receives_placeholder_node(composable):
    inputs = (torch.randn(1, 3),)
    model = torch.export.export(torch.nn.ReLU(), inputs, strict=True).module()
    config = get_symmetric_quantization_config()
    visited = []

    def factory(node):
        visited.append(node)
        return config.output_activation

    quantizer = EthosUQuantizer(
        EthosUCompileSpec("ethos-u85-256"), use_composable_quantizer=composable
    )
    quantizer.set_global(config)
    quantizer.set_io(replace(config, output_activation=factory))
    prepared = prepare_pt2e(model, quantizer)
    prepared(*inputs)
    assert visited
    assert all(node.op == "placeholder" for node in visited)
