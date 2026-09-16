# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    DynamicW8A8QParamsSupport,
)
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.linear.default"

input_t1 = Tuple[torch.Tensor]

test_data_rank1_FP = {
    # test_name: (test_data, out_features, has_bias)
    "model_linear_rank1_zeros": lambda: (
        torch.zeros(10),
        15,
        True,
    ),
    "model_linear_rank1_ones": lambda: (
        torch.ones(10),
        15,
        False,
    ),
    "model_linear_rank1_negative_ones": lambda: (
        torch.ones(10) * (-1),
        20,
        True,
    ),
    "model_linear_rank1_rand": lambda: (
        torch.rand(10),
        10,
        True,
    ),
    "model_linear_rank1_negative_large_rand": lambda: (
        torch.rand(10) * (-100),
        30,
        False,
    ),
    "model_linear_rank1_large_randn": lambda: (
        torch.randn(15) * 100,
        20,
        True,
    ),
}

test_data_rank2_FP = {
    # test_name: (test_data, out_features, has_bias)
    "model_linear_rank2_zeros": lambda: (
        torch.zeros(10, 20),
        15,
        True,
    ),
    "model_linear_rank2_ones": lambda: (
        torch.ones(2, 240),
        960,
        False,
    ),
    "model_linear_rank2_negative_ones": lambda: (
        torch.ones(10, 20) * (-1),
        20,
        True,
    ),
    "model_linear_rank2_rand": lambda: (
        torch.rand(2, 240),
        960,
        True,
    ),
    "model_linear_rank2_negative_large_rand": lambda: (
        torch.rand(10, 20) * (-100),
        30,
        False,
    ),
    "model_linear_rank2_large_randn": lambda: (
        torch.randn(15, 20) * 100,
        20,
        True,
    ),
}

test_data_rank4_FP = {
    # test_name: (test_data, out_features, has_bias)
    "model_linear_rank4_zeros": lambda: (
        torch.zeros(5, 10, 25, 20),
        30,
        True,
    ),
    "model_linear_rank4_ones": lambda: (
        torch.ones(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_ones": lambda: (
        torch.ones(5, 10, 25, 20) * (-1),
        30,
        True,
    ),
    "model_linear_rank4_rand": lambda: (
        torch.rand(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_large_rand": lambda: (
        torch.rand(5, 10, 25, 20) * (-100),
        30,
        True,
    ),
    "model_linear_rank4_large_randn": lambda: (
        torch.randn(5, 10, 25, 20) * 100,
        30,
        False,
    ),
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_rank1_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (*v(), q))
    for (k, v) in test_data_rank1_FP.items()
    for q in [True, False]
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_rank2_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (*v(), q))
    for (k, v) in test_data_rank2_FP.items()
    for q in [True, False]
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_rank4_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (*v(), q))
    for (k, v) in test_data_rank4_FP.items()
    for q in [True, False]
}


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.fc(x)


@common.parametrize("test_data", test_data_rank1_FP | test_data_rank4_FP)
def test_linear_tosa_FP(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineFP[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
def test_linear_tosa_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
def test_linear_tosa_INT_a8w4(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        tosa_extensions=["int4"],
        frobenius_threshold=0.4,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_dtype_count,
        {
            "CONST": {"INT4": 2},
            "CONV2D": {"INT32": 1},
            "RESCALE": {"INT8": 1},
        },
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_rank1_INT | test_data_rank2_INT | test_data_rank4_INT,
)
@common.XfailIfNoCorstone300
def test_linear_u55_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    EthosU55PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize(
    "test_data",
    test_data_rank1_INT | test_data_rank2_INT | test_data_rank4_INT,
)
@common.XfailIfNoCorstone320
def test_linear_u85_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    EthosU85PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize("test_data", test_data_rank1_FP | test_data_rank4_FP)
@common.SkipIfNoModelConverter
def test_linear_vgf_no_quant(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=in_features, out_features=out_features, bias=has_bias),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@common.SkipIfNoModelConverter
def test_linear_vgf_quant(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=in_features, out_features=out_features, bias=has_bias),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        quantize=True,
    )
    pipeline.run()


def _make_dynamic_w8a8_choose_node(
    shape: tuple[int, ...], epsilon: object
) -> torch.fx.Node:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.float32)
    return graph.call_function(
        DynamicW8A8QParamsSupport._choose_target(),
        args=(x, -127, 127, epsilon, torch.int8),
    )


def test_dynamic_w8a8_qparams_support_accepts_decomposable_choose() -> None:
    choose = _make_dynamic_w8a8_choose_node((2, 16), 2**-12)
    assert DynamicW8A8QParamsSupport._choose_signature_is_valid(choose)


def test_dynamic_w8a8_qparams_support_rejects_scalar_choose() -> None:
    choose = _make_dynamic_w8a8_choose_node((), 2**-12)
    assert not DynamicW8A8QParamsSupport._choose_signature_is_valid(choose)


def test_dynamic_w8a8_qparams_support_rejects_invalid_epsilon() -> None:
    invalid_epsilons = (0.0, -1.0, float("inf"), float("-inf"), float("nan"), True)
    for epsilon in invalid_epsilons:
        choose = _make_dynamic_w8a8_choose_node((2, 16), epsilon)
        assert not DynamicW8A8QParamsSupport._choose_signature_is_valid(choose)


def test_dynamic_w8a8_qparams_support_rejects_opaque_epsilon_node() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 16), dtype=torch.float32)
    epsilon = graph.placeholder("epsilon")
    choose = graph.call_function(
        DynamicW8A8QParamsSupport._choose_target(),
        args=(x, -127, 127, epsilon, torch.int8),
    )
    assert not DynamicW8A8QParamsSupport._choose_signature_is_valid(choose)


def _make_linear_node_for_bias_qspec() -> torch.fx.Node:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    bias = graph.placeholder("bias")
    return graph.call_function(
        torch.ops.aten.linear.default,
        args=(x, weight, bias),
    )


def test_dynamic_w8a8_bias_qspec_is_none() -> None:
    """Dynamic activation scale keeps Linear bias in FP32."""
    config = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=True,
        act_qmin=-127,
        act_qmax=127,
    )
    linear = _make_linear_node_for_bias_qspec()

    # The config still contains its normal bias configuration.  The dynamic
    # behavior is implemented by get_bias_qspec(), which deliberately returns
    # None because the activation scale is not known at AOT time.
    assert config.bias is not None
    assert config.get_bias_qspec(linear) is None


def test_static_w8a8_bias_qspec_is_derived() -> None:
    """Static W8A8 retains the existing derived INT32 Linear bias qspec."""
    config = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=False,
        act_qmin=-127,
        act_qmax=127,
    )
    linear = _make_linear_node_for_bias_qspec()

    assert config.bias is not None
    assert config.get_bias_qspec(linear) is not None


dynamic_w8a8_test_data = {
    "rank1_per_channel_bias_folded": lambda: (
        torch.randn(16),
        True,
        True,
        True,
    ),
    "rank2_per_channel_bias_folded": lambda: (
        torch.randn(2, 16),
        True,
        True,
        True,
    ),
    "rank3_per_tensor_no_bias_folded": lambda: (
        torch.randn(2, 3, 16),
        False,
        False,
        True,
    ),
    "rank2_zero_input_per_channel_unfolded": lambda: (
        torch.zeros(2, 16),
        True,
        True,
        False,
    ),
}


# Deterministic cases for numerical execution.  Keep these separate from the
# structural matrix above: numerical regressions should not depend on random
# input or randomly initialized model parameters.
dynamic_w8a8_numerical_test_data = {
    "rank1_per_channel_bias_folded": lambda: (
        torch.linspace(-2.0, 2.0, 16),
        True,
        True,
        True,
    ),
    "rank2_per_channel_bias_folded": lambda: (
        torch.linspace(-2.0, 2.0, 2 * 16).reshape(2, 16),
        True,
        True,
        True,
    ),
    "rank3_per_tensor_no_bias_folded": lambda: (
        torch.linspace(-3.0, 3.0, 2 * 3 * 16).reshape(2, 3, 16),
        False,
        False,
        True,
    ),
    "rank2_zero_per_channel_bias_unfolded": lambda: (
        torch.zeros(2, 16),
        True,
        True,
        False,
    ),
}


dynamic_w8a8_range_test_data = {
    "small_range": lambda: (torch.linspace(-0.01, 0.01, 2 * 16).reshape(2, 16),),
    "normal_range": lambda: (torch.linspace(-1.0, 1.0, 2 * 16).reshape(2, 16),),
    "large_range": lambda: (torch.linspace(-100.0, 100.0, 2 * 16).reshape(2, 16),),
}


def _make_dynamic_w8a8_numerical_linear(has_bias: bool) -> Linear:
    """Create a deterministic Linear so numerical tests are reproducible."""
    model = Linear(
        in_features=16,
        out_features=8,
        bias=has_bias,
    ).eval()

    with torch.no_grad():
        model.fc.weight.copy_(
            torch.linspace(
                -1.25,
                1.25,
                8 * 16,
                dtype=torch.float32,
            ).reshape(8, 16)
        )
        if model.fc.bias is not None:
            model.fc.bias.copy_(torch.linspace(-0.25, 0.25, 8, dtype=torch.float32))

    return model


def _configure_dynamic_w8a8_pipeline(pipeline, per_channel: bool) -> None:
    pipeline.quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=per_channel,
            is_dynamic=True,
            act_qmin=-127,
            act_qmax=127,
        )
    )

    dynamic_qdq = [
        "torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor",
        "torch.ops.quantized_decomposed.dequantize_per_tensor.tensor",
        "torch.ops.quantized_decomposed.quantize_per_tensor.tensor",
    ]
    pipeline.change_args("check.quant_nodes", dynamic_qdq)
    pipeline.change_args("check_not.quant_nodes", dynamic_qdq)

    if pipeline.has_stage("check_not.exir_quant_nodes"):
        edge_qdq_prefix = "executorch_exir_dialects_edge__ops_quantized_decomposed_"
        pipeline.change_args(
            "check_not.exir_quant_nodes",
            [
                edge_qdq_prefix + "choose_qparams_symmetric_tensor",
                edge_qdq_prefix + "quantize_per_tensor_tensor",
                edge_qdq_prefix + "dequantize_per_tensor_tensor",
                edge_qdq_prefix + "quantize_per_tensor_default",
                edge_qdq_prefix + "dequantize_per_tensor_default",
                edge_qdq_prefix + "quantize_per_channel_default",
                edge_qdq_prefix + "dequantize_per_channel_default",
            ],
        )


@common.parametrize("test_data", dynamic_w8a8_test_data)
def test_linear_tosa_INT_FP_dynamic_w8a8(test_data):
    """Verify that dynamic W8A8 is lowered to one TOSA MATMUL."""
    input_data, has_bias, per_channel, fold_quantize = test_data()
    pipeline = TosaPipelineINT[input_t1](
        Linear(in_features=16, out_features=8, bias=has_bias),
        (input_data,),
        aten_op=aten_op,
        exir_op=[],
        run_on_tosa_ref_model=False,
        per_channel_quantization=per_channel,
        fold_quantize=fold_quantize,
        tosa_extensions=["FP"],
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    _configure_dynamic_w8a8_pipeline(pipeline, per_channel)
    pipeline.count_tosa_ops({"MATMUL": 1})
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_data", dynamic_w8a8_test_data)
def test_linear_vgf_dynamic_w8a8(test_data):
    """Verify that the complete dynamic W8A8 graph compiles to VGF."""
    input_data, has_bias, per_channel, fold_quantize = test_data()
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=16, out_features=8, bias=has_bias),
        (input_data,),
        aten_op=aten_op,
        exir_op=[],
        quantize=True,
        per_channel_quantization=per_channel,
        fold_quantize=fold_quantize,
        run_on_vulkan_runtime=False,
        tosa_spec="TOSA-1.0+FP+INT",
    )
    _configure_dynamic_w8a8_pipeline(pipeline, per_channel)
    pipeline.run()


@common.parametrize("test_data", dynamic_w8a8_numerical_test_data)
def test_linear_tosa_INT_FP_dynamic_w8a8_numerics(test_data):
    """Execute lowered dynamic W8A8 and compare with the PT2E QDQ graph."""
    input_data, has_bias, per_channel, fold_quantize = test_data()
    model = _make_dynamic_w8a8_numerical_linear(has_bias)

    pipeline = TosaPipelineINT[input_t1](
        model,
        (input_data,),
        aten_op=aten_op,
        exir_op=[],
        run_on_tosa_ref_model=True,
        compare_tosa_ref_model_outputs=True,
        per_channel_quantization=per_channel,
        fold_quantize=fold_quantize,
        tosa_extensions=["FP"],
        atol=2e-3,
        rtol=2e-3,
        qtol=0,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    _configure_dynamic_w8a8_pipeline(pipeline, per_channel)
    pipeline.count_tosa_ops({"MATMUL": 1})

    assert pipeline.is_tosa_ref_model_available(), (
        "Dynamic W8A8 numerical tests require the TOSA reference model; "
        "otherwise the execution/comparison stage would be removed."
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_data", dynamic_w8a8_numerical_test_data)
def test_linear_vgf_dynamic_w8a8_runtime_numerics(test_data):
    """Execute serialized VGF on Vulkan and compare with PT2E dynamic QDQ."""
    input_data, has_bias, per_channel, fold_quantize = test_data()
    model = _make_dynamic_w8a8_numerical_linear(has_bias)

    pipeline = VgfPipeline[input_t1](
        model,
        (input_data,),
        aten_op=aten_op,
        exir_op=[],
        quantize=True,
        per_channel_quantization=per_channel,
        fold_quantize=fold_quantize,
        run_on_vulkan_runtime=True,
        tosa_spec="TOSA-1.0+FP+INT",
        atol=1e-2,
        rtol=1e-2,
        qtol=0,
    )
    _configure_dynamic_w8a8_pipeline(pipeline, per_channel)
    pipeline.run()


@common.parametrize("test_data", dynamic_w8a8_range_test_data)
def test_linear_tosa_dynamic_w8a8_runtime_activation_ranges(test_data):
    """Recompute activation qparams correctly for very different ranges."""
    inputs = test_data()
    model = _make_dynamic_w8a8_numerical_linear(has_bias=True)

    pipeline = TosaPipelineINT[input_t1](
        model,
        inputs,
        aten_op=aten_op,
        exir_op=[],
        run_on_tosa_ref_model=True,
        compare_tosa_ref_model_outputs=True,
        per_channel_quantization=True,
        fold_quantize=True,
        tosa_extensions=["FP"],
        atol=2e-3,
        rtol=2e-3,
        qtol=0,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    _configure_dynamic_w8a8_pipeline(pipeline, per_channel=True)
    pipeline.count_tosa_ops({"MATMUL": 1})

    assert pipeline.is_tosa_ref_model_available(), (
        "Dynamic W8A8 range tests require the TOSA reference model; "
        "otherwise the execution/comparison stage would be removed."
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@common.SkipIfNoModelConverter
def test_linear_vgf_quant_a8w4(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=in_features, out_features=out_features, bias=has_bias),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


test_data_all_16a8w = test_data_rank1_INT | test_data_rank2_INT | test_data_rank4_INT


@common.parametrize("test_data", test_data_all_16a8w)
def test_linear_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization (16-bit activations, 8-bit
    weights)
    """
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    # Create pipeline with custom 16A8W quantization config
    pipeline = TosaPipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    # Run the pipeline
    pipeline.run()


@common.parametrize("test_data", test_data_all_16a8w)
@common.XfailIfNoCorstone300
def test_linear_16a8w_u55_INT(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization on U55 (16-bit activations,
    8-bit weights)
    """
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    pipeline = EthosU55PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_all_16a8w)
@common.XfailIfNoCorstone320
def test_linear_16a8w_u85_INT(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization on U85 (16-bit activations,
    8-bit weights)
    """
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    pipeline = EthosU85PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
        a16w8_quantization=True,
    )

    pipeline.run()
