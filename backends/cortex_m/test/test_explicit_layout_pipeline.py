# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
    CortexMPassManager,
    LiftConstantTensorsPass,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import (
    CortexMQuantize,
    CortexMRunPasses,
    CortexMTester,
)
from executorch.backends.cortex_m.utils import lowered_output_dtype
from executorch.backends.test.harness.stages import Quantize, StageType
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

# Temporary opt-in coverage. Move these invariants into the standard Cortex-M
# tests when this pass manager becomes the default.


class Conv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConvPadConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.nn.functional.pad(self.conv1(x), (1, 1, 1, 1)))


class UnsupportedAvgPool(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x, kernel_size=2, stride=2, divisor_override=3
        )


class LutActivationThenConv1d(torch.nn.Module):
    """A LUT activation feeding a convolution.

    tanh lowers to ``cortex_m::quantized_activation``, but only in
    ``AtenToCortexMPass``, which runs after the layout validator. Until then
    it keeps an fp32 fake tensor and passes that dtype on to the convolution,
    so validating on ``val`` rejects this graph even though it lowers.
    An activation that ``ActivationFusionPass`` folds into the convolution
    clamp, such as relu, never exposes the gap.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, 1)

    def forward(self, x):
        return self.conv(torch.tanh(x))


class LutActivationViewThenConv1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, 1)

    def forward(self, x):
        return self.conv(torch.tanh(x).view(1, 8, 16))


class LutActivationReluThenConv1d(torch.nn.Module):
    """A relu between a LUT activation and a convolution.

    Nothing fuses this relu. ``ActivationFusionPass`` only folds an
    activation that follows a convolution, so it reaches
    ``QuantizedClampActivationPass``, which has to quantize its bound
    against the sigmoid's qparams. Left alone on int8 codes, ``aten.relu``
    clamps at 0 rather than at the zero point and erases everything below
    it -- and it is meant to be a no-op here, sigmoid being positive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, 1)

    def forward(self, x):
        return self.conv(torch.relu(torch.sigmoid(x)))


class LutActivationClampThenConv2d(torch.nn.Module):
    """A two-sided clamp between a LUT activation and a convolution.

    The same gap as ``LutActivationReluThenConv1d``, with bounds of its
    own: unquantized, ``aten.clamp`` holds fp32 limits against int8 codes
    and keeps an fp32 value on a tensor the convolution reads as int8.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x):
        return self.conv(torch.clamp(torch.tanh(x), -0.5, 0.5))


class LutActivationThenElementwise(torch.nn.Module):
    """An elementwise binary op fed by a LUT activation.

    ``minimum`` and ``maximum`` pick their kernel from the same fake
    tensor the input-side layout permute does, and there is no layout
    copy here to involve the validator -- the convolution is what drags
    one in. So this is the gap on its own: a graph that is int8 start to
    finish and still leaves one op on the portable kernel.
    """

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x, y):
        return self.fn(torch.tanh(x), y)


class NHWCPaddedConv(torch.nn.Module):
    def __init__(self, channels, shared_pad):
        super().__init__()
        out_channels = 64 if channels == 1 else 8
        self.conv = torch.nn.Conv2d(channels, out_channels, (10, 4), stride=2)
        self.shared_pad = shared_pad

    def forward(self, x):
        padded = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (1, 1, 4, 5))
        output = self.conv(padded).permute(0, 2, 3, 1)
        return (output, padded) if self.shared_pad else output


def _count(exported_program, target) -> int:
    return sum(node.target == target for node in exported_program.graph.nodes)


def _only(exported_program, target) -> Node:
    nodes = [node for node in exported_program.graph.nodes if node.target == target]
    assert len(nodes) == 1, f"expected one {target}, found {len(nodes)}"
    return nodes[0]


def _run_explicit_layout_pass_manager(tester: CortexMTester) -> CortexMTester:
    target_config = CortexMTargetConfig(cpu=CortexM.M55)
    tester.run_passes(
        CortexMRunPasses(
            target_config=target_config,
            use_explicit_layout=True,
        )
    )
    return tester


def _run_explicit_layout_passes(tester: CortexMTester) -> CortexMTester:
    tester.quantize(Quantize(CortexMQuantizer(use_explicit_layout=True)))
    tester.export().to_edge()
    return _run_explicit_layout_pass_manager(tester)


@pytest.mark.parametrize("use_explicit_layout", [False, True])
@pytest.mark.parametrize("test_method", ["test_dialect", "test_implementation"])
def test_pipeline_uses_configured_stages(monkeypatch, test_method, use_explicit_layout):
    class CustomQuantize(CortexMQuantize):
        pass

    class CustomRunPasses(CortexMRunPasses):
        pass

    inputs = (torch.randn(1, 3, 8, 8),)
    if not use_explicit_layout:
        inputs = (inputs[0].to(memory_format=torch.channels_last),)
    tester = CortexMTester(Conv2d().eval(), inputs)
    tester.stage_classes[StageType.QUANTIZE] = CustomQuantize
    if use_explicit_layout:
        tester.stage_classes[StageType.RUN_PASSES] = lambda use_explicit_layout: (
            CustomRunPasses(
                target_config=tester.target_config,
                use_explicit_layout=use_explicit_layout,
            )
        )
    else:
        tester.stage_classes[StageType.RUN_PASSES] = lambda: CustomRunPasses(
            target_config=tester.target_config
        )
    if test_method == "test_dialect":
        tester.test_dialect(
            {}, {}, use_explicit_layout=use_explicit_layout, compare_outputs=False
        )
    else:
        monkeypatch.setattr(tester, "to_executorch", lambda: tester)
        monkeypatch.setattr(tester, "serialize", lambda: tester)
        tester.test_implementation(
            use_explicit_layout=use_explicit_layout, compare_outputs=False
        )

    assert isinstance(tester.stages[StageType.QUANTIZE], CustomQuantize)
    assert isinstance(tester.stages[StageType.RUN_PASSES], CustomRunPasses)
    assert (
        _count(
            tester.get_artifact(StageType.RUN_PASSES).exported_program(),
            (
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default
                if use_explicit_layout
                else exir_ops.edge.cortex_m.quantized_conv2d.default
            ),
        )
        == 1
    )


def test_layout_pipelines_select_distinct_spatial_operators():
    legacy_input = torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
    legacy = CortexMTester(
        Conv2d().eval().to(memory_format=torch.channels_last),
        (legacy_input,),
    )
    legacy.quantize().export().to_edge().run_passes()
    legacy_program = legacy.get_artifact(StageType.RUN_PASSES).exported_program()

    explicit = _run_explicit_layout_passes(
        CortexMTester(Conv2d().eval(), (torch.randn(1, 3, 8, 8),))
    )
    explicit_program = explicit.get_artifact(StageType.RUN_PASSES).exported_program()

    for program in (legacy_program, explicit_program):
        assert all(
            node.users for node in program.graph.nodes if node.op == "placeholder"
        )

    assert _count(legacy_program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 1
    assert (
        _count(
            legacy_program,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        )
        == 0
    )
    assert (
        _count(explicit_program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 0
    )
    assert (
        _count(
            explicit_program,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        )
        == 1
    )
    assert _count(explicit_program, exir_ops.edge.cortex_m.transpose.default) == 2


@pytest.mark.parametrize("entry_point", ["legacy", "edge_passes", "edge_manager"])
@pytest.mark.parametrize(
    "passes,lifted,pruned",
    [
        pytest.param([], False, False, id="empty"),
        pytest.param([LiftConstantTensorsPass], True, False, id="lift"),
        pytest.param(
            [LiftConstantTensorsPass, RemoveUnusedConstantsPass],
            True,
            True,
            id="lift_and_prune",
        ),
    ],
)
def test_constant_cleanup_respects_pass_list_and_lift_order(
    passes, lifted, pruned, entry_point
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_lifted_tensor_constant0", torch.tensor([1.0]))
            self.register_buffer("_lifted_tensor_constant1", torch.tensor([2.0]))

        def forward(self, input):
            return input + self._lifted_tensor_constant1

    class OtherEntryPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_lifted_tensor_constant2", torch.tensor([7.0]))

        def forward(self, x):
            return x + self._lifted_tensor_constant2

    inputs = (torch.zeros(1),)
    edge = to_edge(torch.export.export(Model(), inputs))
    program = edge.exported_program()
    assert "_lifted_tensor_constant0" in program.graph_signature.buffers
    program.graph_module.register_buffer("new_tensor", torch.tensor([3.0]))
    output = next(node for node in program.graph.nodes if node.op == "output")
    original = output.args[0][0]
    with program.graph.inserting_before(original):
        constant = program.graph.get_attr("new_tensor")
        constant.meta = original.meta.copy()
        result = program.graph.call_function(
            exir_ops.edge.aten.add.Tensor, (original.args[1], constant)
        )
        result.meta = original.meta.copy()
    original.replace_input_with(original.args[1], result)
    program.graph_module.recompile()
    program.validate()

    other = to_edge(torch.export.export(OtherEntryPoint(), inputs)).exported_program()
    program.state_dict.update(other.state_dict)
    other._state_dict = program.state_dict
    other.validate()
    original_code = program.graph_module.code
    original_signature = copy.deepcopy(program.graph_signature)
    original_state = program.state_dict.copy()
    program.graph_module.meta["constant_lifting_test"] = "preserved"

    if entry_point == "edge_passes":
        transformed = edge.transform(
            [pass_cls() for pass_cls in passes]
        ).exported_program()
    elif entry_point == "edge_manager":
        transformed = edge.transform(
            CortexMPassManager(passes=passes)
        ).exported_program()
    else:
        transformed = CortexMPassManager(program, passes=passes).transform()

    program.validate()
    other.validate()
    torch.testing.assert_close(program.module()(*inputs), torch.tensor([5.0]))
    torch.testing.assert_close(other.module()(*inputs), torch.tensor([7.0]))
    assert program.graph_module.code == original_code
    assert program.graph_signature == original_signature
    assert program.state_dict.keys() == original_state.keys()
    for name, tensor in original_state.items():
        assert program.state_dict[name] is tensor
        assert other.state_dict[name] is tensor
    transformed.validate()
    assert transformed.graph_signature.user_inputs == original_signature.user_inputs
    assert transformed.graph_module.meta["constant_lifting_test"] == "preserved"
    assert ("_lifted_tensor_constant0" in transformed.graph_signature.buffers) == (
        not pruned
    )
    assert sum(node.op == "get_attr" for node in transformed.graph.nodes) == (
        not lifted
    )
    assert (
        transformed.state_dict["_lifted_tensor_constant1"]
        is original_state["_lifted_tensor_constant1"]
    )
    torch.testing.assert_close(transformed.module()(*inputs), torch.tensor([5.0]))


def test_conv1d_is_quantized_before_layout_conversion():
    tester = CortexMTester(Conv1d().eval(), (torch.randn(1, 2, 8),))
    tester.quantize(Quantize(CortexMQuantizer(use_explicit_layout=True)))
    quantized = tester.get_artifact(StageType.QUANTIZE)
    [conv1d] = [
        node
        for node in quantized.graph.nodes
        if node.target == torch.ops.aten.conv1d.default
    ]

    weight = conv1d.args[1]
    assert isinstance(weight, Node)
    assert (
        weight.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
    )

    tester.export().to_edge()
    _run_explicit_layout_pass_manager(tester)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.aten.convolution.default) == 0


def test_explicit_layout_reuses_pad():
    tester = _run_explicit_layout_passes(
        CortexMTester(ConvPadConv().eval(), (torch.randn(1, 3, 8, 8),))
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.pad.default) == 1


def _lower_nhwc_padded_conv(channels, shared_pad):
    torch.manual_seed(7)
    tester = _run_explicit_layout_passes(
        CortexMTester(
            NHWCPaddedConv(channels, shared_pad).eval(),
            (torch.randn(1, 49, 10, channels),),
        )
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.pad.default) == int(shared_pad)
    if not shared_pad:
        assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0
    return tester


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shared_pad", [False, True])
def test_explicit_layout_fuses_same_padding(channels, shared_pad):
    tester = _lower_nhwc_padded_conv(channels, shared_pad)
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shared_pad", [False, True])
def test_implementation_explicit_layout_fuses_same_padding(channels, shared_pad):
    tester = _lower_nhwc_padded_conv(channels, shared_pad)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@pytest.mark.parametrize("hardtanh", [False, True])
def test_implementation_transpose_conv2d_strided_pointwise(hardtanh):
    torch.manual_seed(0)
    inputs = (torch.randn(2, 4, 9).unsqueeze(2),)
    model = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(4, 4, (1, 1), stride=(1, 2)),
        torch.nn.Hardtanh(-0.5, 0.5) if hardtanh else torch.nn.Identity(),
    ).eval()
    tester = _run_explicit_layout_passes(CortexMTester(model, inputs))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default)
        == 1
    )
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


def test_explicit_layout_accepts_conv_fed_by_lut_activation():
    """A convolution behind a LUT activation lowers and keeps its numerics."""
    torch.manual_seed(0)
    inputs = (torch.randn(1, 8, 16),)
    tester = _run_explicit_layout_passes(
        CortexMTester(LutActivationThenConv1d().eval(), inputs)
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.aten.convolution.default) == 0
    assert _count(program, exir_ops.edge.cortex_m.quantized_activation.default) == 1
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


def test_layout_permute_behind_lut_activation_lowers_to_transpose():
    """Every layout move in an int8 chain reaches a CMSIS-NN transpose.

    The convolution has a layout move on each side. The one on its output
    is synthesized as ``channels_last.permute_copy``, whose substitution
    reads no dtype; the one on its input is an ``aten.permute_copy``, and
    its substitution read the same fake tensor the layout validator used
    to. Behind a LUT activation that tensor is fp32, so the input-side
    move was left on the portable kernel while the output-side move
    converted. Numerics are unaffected -- portable ``permute_copy``
    computes the right thing on int8 -- so what this costs is the
    invariant: a selective build registering only the kernels the graph
    is supposed to need fails to load.
    """
    torch.manual_seed(0)
    inputs = (torch.randn(1, 8, 16),)
    tester = _run_explicit_layout_passes(
        CortexMTester(LutActivationThenConv1d().eval(), inputs)
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert _count(program, exir_ops.edge.aten.permute_copy.default) == 0
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


@pytest.mark.parametrize(
    "fn,cortex_m_op,aten_op",
    [
        (torch.minimum, exir_ops.edge.cortex_m.minimum, exir_ops.edge.aten.minimum),
        (torch.maximum, exir_ops.edge.cortex_m.maximum, exir_ops.edge.aten.maximum),
    ],
)
def test_elementwise_behind_lut_activation_lowers_to_cortex_m(fn, cortex_m_op, aten_op):
    """An elementwise binary op behind a LUT activation reaches CMSIS-NN."""
    torch.manual_seed(0)
    inputs = (torch.randn(1, 4, 8), torch.randn(1, 4, 8))
    tester = _run_explicit_layout_passes(
        CortexMTester(LutActivationThenElementwise(fn).eval(), inputs)
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, cortex_m_op.default) == 1
    assert _count(program, aten_op.default) == 0
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


@pytest.mark.parametrize(
    "model,inputs,bounds",
    [
        (LutActivationReluThenConv1d, (torch.randn(1, 8, 16),), (int, type(None))),
        (LutActivationClampThenConv2d, (torch.randn(1, 4, 8, 8),), (int, int)),
    ],
)
def test_clamp_between_lut_activation_and_conv_is_quantized(model, inputs, bounds):
    """A clamp-like activation inside an int8 chain runs on int8 terms.

    The convolution reaches a CMSIS-NN kernel either way. What the fake
    tensor costs is the clamp between it and the LUT activation: left in
    float, its bounds mean nothing against int8 codes and the tensor the
    convolution reads is still declared fp32.
    """
    torch.manual_seed(0)
    tester = _run_explicit_layout_passes(CortexMTester(model().eval(), inputs))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.quantized_activation.default) == 1
    clamp = _only(program, exir_ops.edge.aten.clamp.default)
    assert tuple(type(bound) for bound in clamp.args[1:]) == bounds
    assert clamp.meta["val"].dtype == torch.int8
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


@pytest.mark.parametrize(
    "model,inputs,bounds",
    [
        (LutActivationReluThenConv1d, (torch.randn(1, 8, 16),), (int, type(None))),
        (LutActivationClampThenConv2d, (torch.randn(1, 4, 8, 8),), (int, int)),
    ],
)
def test_clamp_between_lut_activation_and_conv_is_quantized_on_legacy_passes(
    model, inputs, bounds
):
    """The same clamp on the default pass list, which has no validator.

    Here the clamp is the only thing standing between the LUT activation
    and the rest of the graph: nothing rejects an unquantized one, so it
    reaches the program. `aten.relu` left alone cuts at code 0 while the
    zero point is -128, and `aten.clamp` left on float bounds promotes
    the tensor back to fp32 in front of a dequantize that reads int8.
    """
    torch.manual_seed(0)
    tester = CortexMTester(model().eval(), inputs)
    tester.quantize().export().to_edge().run_passes()
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_activation.default) == 1
    assert _count(program, exir_ops.edge.aten.relu.default) == 0
    clamp = _only(program, exir_ops.edge.aten.clamp.default)
    assert tuple(type(bound) for bound in clamp.args[1:]) == bounds
    assert clamp.meta["val"].dtype == torch.int8
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


def test_explicit_layout_accepts_conv_fed_by_view_of_lut_activation():
    """A dtype-agnostic op between the LUT activation and the conv stays int8."""
    torch.manual_seed(0)
    inputs = (torch.randn(1, 128),)
    tester = _run_explicit_layout_passes(
        CortexMTester(LutActivationViewThenConv1d().eval(), inputs)
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.quantized_activation.default) == 1
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


def test_lowered_output_dtype_reads_first_output_of_multi_output_node():
    node = torch.fx.Graph().call_function(
        exir_ops.edge.aten.max_pool2d_with_indices.default
    )
    node.meta["val"] = (
        torch.empty(1, dtype=torch.int8),
        torch.empty(1, dtype=torch.int64),
    )
    assert lowered_output_dtype(node) == torch.int8


def test_lowered_output_dtype_declines_a_non_node_argument():
    """Substitutions read it straight off ``node.args``, which holds literals."""
    assert lowered_output_dtype(5.0) is None


def test_explicit_layout_rejects_unsupported_spatial_operator():
    tester = CortexMTester(UnsupportedAvgPool(), (torch.randn(1, 3, 8, 8),))

    with pytest.raises(Exception) as caught:
        _run_explicit_layout_passes(tester)

    error = caught.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert "NHWC-eligible" in str(error)
