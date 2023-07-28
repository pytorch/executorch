# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.utils as utils
import torch
import torch.nn.functional as F
from executorch import exir
from executorch.backends.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir import CaptureConfig
from executorch.exir.dialects._ops import ops as exir_ops

from torch.ao.quantization import PlaceholderObserver, QConfig, QConfigMapping
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    weight_observer_range_neg_127_to_127,
)

from torch.ao.quantization.qconfig_mapping import _get_symmetric_qnnpack_qconfig_mapping

from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)

from torch.fx import symbolic_trace

"""
How to write a pattern:

1. Find the op and write a function to use this op
2. Get the graph like following
    add_pattern_graph = (
        exir.capture(
            add,
            model_inputs,
            config=CaptureConfig(pt2_mode=True, enable_dynamic_shape=True),
        )
        .to_edge(exir.EdgeCompileConfig(_use_edge_ops=True)).module
        .graph
    )
"""
# Shorter Target Names
T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
T_Addmm = exir_ops.edge.aten.addmm.default
T_VCopy = exir_ops.edge.aten.view_copy.default
T_TCopy = exir_ops.edge.aten.t_copy.default
T_Conv = exir_ops.edge.aten.convolution.default
T_Hardtanh = exir_ops.edge.aten.hardtanh.default
T_MeanDim = exir_ops.edge.aten.mean.dim
T_MaxDim = exir_ops.edge.aten.max.dim
T_Add = exir_ops.edge.aten.add.Tensor
T_Sub = exir_ops.edge.aten.sub.Tensor
T_MaxPool2d = exir_ops.edge.aten.max_pool2d_with_indices.default
T_ReLU = exir_ops.edge.aten.relu.default
T_Clamp = exir_ops.edge.aten.clamp.default
T_Floor = exir_ops.edge.aten.floor.default
T_Minimum = exir_ops.edge.aten.minimum.default
T_Mul = exir_ops.edge.aten.mul.Tensor


def _capture(module, example_inputs, pt_mode=True) -> torch.fx.GraphModule:
    capture_config = CaptureConfig(pt2_mode=pt_mode, enable_dynamic_shape=False)
    edge_config = exir.EdgeCompileConfig(
        _check_ir_validity=False,
        _use_edge_ops=True,
        passes=[DuplicateDequantNodePass()],
    )

    return (
        exir.capture(module.eval(), example_inputs, config=capture_config)
        .to_edge(config=edge_config)
        .exported_program.graph_module
    )


def get_pattern_graph(one_module, example_inputs, pt_mode=True):
    class WrappedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.one_module = one_module

        def forward(self, *args):
            return self.one_module(*args)

    return _capture(WrappedModule(), example_inputs, pt_mode=pt_mode).graph


def get_quantized_pattern_graph(module, example_inputs, qconfig_mapping=None):

    prepared = prepare_fx(
        module,
        _get_symmetric_qnnpack_qconfig_mapping()
        if not qconfig_mapping
        else qconfig_mapping,
        example_inputs,
        backend_config=get_executorch_backend_config(),
    )

    converted: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
        prepared,
        backend_config=get_executorch_backend_config(),
    )

    captured = _capture(converted, example_inputs)

    utils.remove_first_quant_and_last_dequant(captured)  # TODO Delete

    return captured.graph


def get_add_modules():
    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class M2(torch.nn.Module):
        def forward(self, x):
            return x + x

    return [(M(), (torch.ones(1), torch.ones(1))), (M2(), (torch.ones(1),))]


def get_add_graphs():
    add_graphs = []
    for add_module, model_inputs in get_add_modules():
        add_graphs.append(get_pattern_graph(add_module, model_inputs))
    return add_graphs


def get_quantized_add_graphs():
    add_graphs = []
    for add_module, model_inputs in get_add_modules():
        add_graphs.append(get_quantized_pattern_graph(add_module, model_inputs))
    return add_graphs


def get_div_graph():
    div_module = torch.div
    model_inputs = (torch.ones(1), torch.ones(1))
    return get_pattern_graph(div_module, model_inputs)


def get_clamp_graph():
    def clamp_pattern(x: torch.Tensor, min_val, max_val) -> torch.Tensor:
        return T_Clamp(x, min_val, max_val)

    return symbolic_trace(clamp_pattern).graph


def get_sub_graph():
    def sub_pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return T_Sub(x, y)

    return symbolic_trace(sub_pattern).graph


def get_floor_graph():
    def floor_pattern(x: torch.Tensor) -> torch.Tensor:
        return T_Floor(x)

    return symbolic_trace(floor_pattern).graph


def get_mean_dim_graphs():
    # Only support patterns for mean dim that are writable as AvgPooling2d
    def mean_dim_pattern(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return T_MeanDim(x, [-2, -1], True)

    def mean_dim_pattern2(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return T_MeanDim(x, [-1, -2], True)

    return [
        symbolic_trace(mean_dim_pattern).graph,
        symbolic_trace(mean_dim_pattern2).graph,
    ]


def get_minimum_graph():
    def minimum_pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return T_Minimum(x, y)

    return symbolic_trace(minimum_pattern).graph


def get_max_dim_graph():
    def max_pattern(x: torch.Tensor, dim, keepdim):
        return T_MaxDim(x, dim, keepdim)

    return symbolic_trace(max_pattern).graph


def get_multiply_graph():
    def mul_pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return T_Mul(x, y)

    return symbolic_trace(mul_pattern).graph


def get_max_pool2d_graph():
    class MaxPool2dModule(torch.nn.Module):
        def __init__(
            self,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
        ):
            super().__init__()
            self.max_pool2d_module = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        def forward(self, x):
            return self.max_pool2d_module(x)

    maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
    model_inputs = (torch.randn(4, 3, 12, 12),)

    return get_pattern_graph(maxpool2d_module, model_inputs)


def get_conv2d_modules():
    return_list = []
    for has_bias in (True, False):
        conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=[1, 1],
            padding=[1, 1],
            groups=1,
            dilation=[1, 1],
            bias=has_bias,
        ).eval()

        model_inputs = (torch.ones(1, 1, 1, 1),)
        return_list.append((conv, model_inputs))
    return return_list


def get_conv2d_graphs():
    conv_patterns = []
    for conv, model_inputs in get_conv2d_modules():
        conv_patterns.append(get_pattern_graph(conv, model_inputs))
    return conv_patterns


def get_batch_norm_graphs():
    class ConvBatchModule(torch.nn.Module):
        def __init__(self, has_bias):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, 1),
                stride=[1, 1],
                padding=[1, 1],
                groups=1,
                dilation=[1, 1],
                bias=has_bias,
            )
            self.bn = torch.nn.BatchNorm2d(1)
            self.eval()

        def forward(self, x):
            y = self.conv(x)
            return self.bn(y)

    batch_norm_graphs = []
    for has_bias in (True, False):
        model_inputs = (torch.ones(1, 1, 1, 1),)
        batch_norm_graphs.append(
            get_pattern_graph(ConvBatchModule(has_bias), model_inputs)
        )

    return batch_norm_graphs


def get_addmm_without_transpose_weights_graph():
    linear_module = torch.nn.Linear(3, 3)
    model_inputs = (torch.ones(3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_mm_without_transpose_weights_graph():
    linear_module = torch.nn.Linear(3, 3, bias=False)
    model_inputs = (torch.ones(3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_addmm_without_view_copy_graph():
    linear_module = torch.nn.Linear(3, 3)
    model_inputs = (torch.ones(3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_addmm_with_view_copy_graph():
    linear_module = torch.nn.Linear(3, 3)
    model_inputs = (torch.ones(1, 3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_mm_without_view_copy_graph():
    linear_module = torch.nn.Linear(3, 3, bias=False)
    model_inputs = (torch.ones(3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_mm_with_view_copy_graph():
    linear_module = torch.nn.Linear(3, 3, bias=False)
    model_inputs = (torch.ones(1, 3, 3),)
    return get_pattern_graph(linear_module, model_inputs)


def get_all_fp_linear_pattern():
    return [
        get_addmm_without_transpose_weights_graph(),
        get_mm_without_transpose_weights_graph(),
        get_addmm_with_view_copy_graph(),
        get_addmm_without_view_copy_graph(),
        get_mm_with_view_copy_graph(),
        get_mm_without_view_copy_graph(),
    ]


def get_all_quantized_linear_pattern():
    return [
        get_static_quant_per_tensor_addmm_with_view_copy_graph(),
        get_static_quant_per_tensor_addmm_without_view_copy_graph(),
    ]


def get_all_dynamically_quantized_linear_pattern():
    return [
        get_dynamic_quant_per_tensor_addmm_with_view_copy_graph(),
        get_dynamic_quant_per_tensor_addmm_without_view_copy_graph(),
        get_dynamic_quant_per_tensor_mm_with_view_copy_graph(),
        get_dynamic_quant_per_tensor_mm_without_view_copy_graph(),
        get_dynamic_quant_per_channel_addmm_with_view_copy_graph(),
        get_dynamic_quant_per_channel_addmm_without_view_copy_graph(),
        get_dynamic_quant_per_channel_mm_with_view_copy_graph(),
        get_dynamic_quant_per_channel_mm_without_view_copy_graph(),
    ]


def get_sigmoid_graph():
    sigmoid_module = torch.nn.Sigmoid()
    model_inputs = (torch.rand(7, 5, 3),)
    return get_pattern_graph(sigmoid_module, model_inputs)


def get_activation_module(activation_fn):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fn = activation_fn()

        def forward(self, x):
            return self.fn(x)

    example_input = (torch.randn(7),)
    return (M().eval(), example_input)


def get_hardtanh_graph():
    return get_pattern_graph(*get_activation_module(torch.nn.Hardtanh))


def get_quantized_hardtanh_graphs():
    return [get_quantized_pattern_graph(*get_activation_module(torch.nn.Hardtanh))]


def get_relu_graph():
    relu_module = torch.nn.ReLU()
    model_inputs = (torch.randn(2, 3, 4),)
    return get_pattern_graph(relu_module, model_inputs)


def get_softmax_graph():
    sigmoid_module = torch.nn.Softmax(1)
    model_inputs = (torch.randn(2, 3),)
    return get_pattern_graph(sigmoid_module, model_inputs)


def get_quantized_conv_graphs():
    qconv_patterns = []
    for conv, model_inputs in get_conv2d_modules():
        qconv_patterns.append(get_quantized_pattern_graph(conv, model_inputs))
    return qconv_patterns


def fuse_relu_module(module, num_inputs=1):
    if num_inputs == 1:
        return torch.nn.Sequential(module, torch.nn.ReLU())
    elif num_inputs == 2:

        class ReLUOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()
                self.mod = module

            def forward(self, x, y):
                return self.relu(self.mod(x, y))

        return ReLUOutput()
    else:
        raise NotImplementedError("No case for more than two inputs")


def get_quantized_conv_relu_graphs():
    qconv_relu_graphs = []
    for conv, model_inputs in get_conv2d_modules():
        qconv_relu_graphs.append(
            get_quantized_pattern_graph(fuse_relu_module(conv), model_inputs)
        )
    return qconv_relu_graphs


def get_quantized_add_relu_graphs():
    qadd_relu_graphs = []
    for add, model_inputs in get_add_modules():
        qadd_relu_graphs.append(
            get_quantized_pattern_graph(
                fuse_relu_module(add, len(model_inputs)), model_inputs
            )
        )

    return qadd_relu_graphs


def get_mean_module():
    supported_dims = [[-1, -2], [-2, -1]]

    class M(torch.nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return torch.mean(x, self.dims, keepdim=True)

    example_input = (torch.randn(3, 2),)
    return [
        (M(supported_dim).eval(), example_input) for supported_dim in supported_dims
    ]


def get_quantized_mean_dim_graphs():
    graphs = []
    for mean_module, example_inputs in get_mean_module():
        graphs.append(get_quantized_pattern_graph(mean_module, example_inputs))
    return graphs


def get_max_pool2d_module(args):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_pool2d = torch.nn.MaxPool2d(*args)

        def forward(self, x):
            return self.max_pool2d(x)

    example_input = (torch.randn(4, 4, 4, 4),)
    return (M().eval(), example_input)


def get_quantized_max_pool_2d_graphs():
    return [
        get_quantized_pattern_graph(*get_max_pool2d_module((2, 1))),
        get_quantized_pattern_graph(*get_max_pool2d_module((2, 1, 1))),
    ]


def get_dynamic_quantized_linear_graph(linear_module, example_inputs, weight_qconfig):
    assert weight_qconfig in [
        weight_observer_range_neg_127_to_127,
        per_channel_weight_observer_range_neg_127_to_127,
    ]

    act_affine_quant_obs = PlaceholderObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-128,
        quant_max=127,
        eps=2**-12,
        is_dynamic=True,
    )
    qconfig_mapping = QConfigMapping().set_object_type(
        F.linear,
        QConfig(
            activation=act_affine_quant_obs,
            weight=weight_qconfig,
        ),
    )
    return get_quantized_pattern_graph(linear_module, example_inputs, qconfig_mapping)


# Per tensor Static Variants
def get_static_quant_per_tensor_addmm_with_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
    return get_quantized_pattern_graph(linear, example_inputs)


def get_static_quant_per_tensor_addmm_without_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 3, dtype=torch.float),)
    return get_quantized_pattern_graph(linear, example_inputs)


# TODO: Per channel Static Variants

# Per tensor Dynamic Variants
def get_dynamic_quant_per_tensor_addmm_with_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
    weight_qconfig = weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_tensor_addmm_without_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 3, dtype=torch.float),)
    weight_qconfig = weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_tensor_mm_with_view_copy_graph():
    linear = torch.nn.Linear(3, 4, bias=False).eval()
    example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
    weight_qconfig = weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_tensor_mm_without_view_copy_graph():
    linear = torch.nn.Linear(3, 4, bias=False).eval()
    example_inputs = (torch.ones(1, 3, dtype=torch.float),)
    weight_qconfig = weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


# Per channel Dynamic Variants
def get_dynamic_quant_per_channel_addmm_with_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
    weight_qconfig = per_channel_weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_channel_addmm_without_view_copy_graph():
    linear = torch.nn.Linear(3, 4).eval()
    example_inputs = (torch.ones(1, 3, dtype=torch.float),)
    weight_qconfig = per_channel_weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_channel_mm_with_view_copy_graph():
    linear = torch.nn.Linear(3, 4, bias=False).eval()
    example_inputs = (torch.ones(1, 1, 3, dtype=torch.float),)
    weight_qconfig = per_channel_weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


def get_dynamic_quant_per_channel_mm_without_view_copy_graph():
    linear = torch.nn.Linear(3, 4, bias=False).eval()
    example_inputs = (torch.ones(1, 3, dtype=torch.float),)
    weight_qconfig = per_channel_weight_observer_range_neg_127_to_127
    return get_dynamic_quantized_linear_graph(linear, example_inputs, weight_qconfig)


# TODO(T148779166) Recompose bilinear
def get_static_resize_bilinear_2d_graphs():
    class StaticResizeBilinear2DSizeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            a = torch.nn.functional.interpolate(
                x,
                size=(x.shape[2] * 2, x.shape[3] * 3),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            return a

    class StaticResizeBilinear2DScaleFactorModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            a = torch.nn.functional.interpolate(
                x,
                scale_factor=3.0,
                mode="bilinear",
                align_corners=True,
                antialias=False,
            )
            return a

    static_resize_bilinear_2d_size_module = StaticResizeBilinear2DSizeModule().eval()
    static_resize_bilinear_2d_scale_factor_module = (
        StaticResizeBilinear2DScaleFactorModule().eval()
    )
    model_inputs = (torch.randn(2, 3, 4, 5),)
    return [
        get_pattern_graph(
            static_resize_bilinear_2d_size_module, model_inputs, pt_mode=True
        ),
        get_pattern_graph(
            static_resize_bilinear_2d_scale_factor_module, model_inputs, pt_mode=True
        ),
    ]


def get_static_constant_pad_graph():
    class StaticConstantPadModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            a = torch.nn.functional.pad(
                input=x,
                pad=(1, 2, 3, 4, 5, 6),
                mode="constant",
                value=2.3,
            )
            return a

    static_constant_pad_module = StaticConstantPadModule()
    model_inputs = (torch.randn(size=(5, 4, 3, 2)),)
    return get_pattern_graph(static_constant_pad_module, model_inputs)
