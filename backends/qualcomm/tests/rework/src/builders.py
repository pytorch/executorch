# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import pytest
import torch

from executorch.backends.qualcomm import _passes

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import (
    OpCast,
    OpConv2d,
    OpConv3d,
    OpTranspose,
)
from executorch.backends.qualcomm.builders.utils import (
    is_parameter,
    is_runtime_input,
    mark_runtime_inputs,
)
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    Tolerance,
    verify_output_emulator,
    verify_output_remote,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS,
    QCOM_AXIS_ORDER,
    QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS,
    QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS,
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_FILTER_AXIS_ORDER_HWIO,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_RUNTIME_INPUTS,
    QCOM_SCALE,
    QCOM_SCALES,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)
from executorch.backends.qualcomm.utils.utils import to_edge_transform_and_lower_to_qnn
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.utils import get_delegates
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class _AnnotateCustomOpLayouts(PassBase):
    def call(self, graph_module):
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target.__name__ == "qnn_builder_test.sin.default"
            ):
                axis_order = (
                    QCOM_FILTER_AXIS_ORDER_HWIO
                    if node.meta["val"].dim() == 4
                    else (0, 1)
                )
                node.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: axis_order}
                node.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: axis_order}
        return PassResult(graph_module, False)


class CustomOpAxisOrders:
    class _Single(torch.nn.Module):
        def __init__(self, static=False, consumer=None, dtype=torch.int8):
            super().__init__()
            self.static = static
            self.consumer = consumer
            if static:
                self.register_buffer("weight", torch.ones(2, 3, dtype=dtype))

        def forward(self, x):
            value = self.weight if self.static else x
            output = torch.ops.qnn_builder_test.foo.default(value)
            return output if self.consumer is None else (output, self.consumer(value))

    class _TupleChain(torch.nn.Module):
        def forward(self, x):
            first, second = torch.ops.qnn_builder_test.pair.default(x)
            return torch.ops.qnn_builder_test.bar.default(first), second

    @staticmethod
    def test(builder_pipeline, subtests):
        CustomOpAxisOrders._test_input_metadata(builder_pipeline, subtests)
        for static in (False, True):
            for normal_first in (False, True):
                with subtests.test(
                    msg="shared_normal_consumer",
                    static=static,
                    normal_first=normal_first,
                ):
                    exported = builder_pipeline.export(
                        CustomOpAxisOrders._Single(static, torch.cos, torch.float32),
                        (torch.randn(2, 3),),
                    )
                    nodes = [
                        node
                        for node in exported.graph.nodes
                        if node.op == "call_function"
                    ]
                    custom, normal = nodes
                    custom.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: (1, 0)}
                    custom.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: (1, 0)}
                    visitors = builder_pipeline.visitors(exported)
                    wrappers = defaultdict(dict)
                    if normal_first:
                        visitors[normal.target.__name__].define_node(normal, wrappers)
                    if static:
                        with pytest.raises(ValueError, match="incompatible consumer"):
                            visitors[custom.target.__name__].define_node(
                                custom, wrappers
                            )
                    else:
                        operations = visitors[custom.target.__name__].define_node(
                            custom, wrappers
                        )
                        if not normal_first:
                            visitors[normal.target.__name__].define_node(
                                normal, wrappers
                            )
                        builder_pipeline.assert_transpose(operations[0], (1, 0))
                        builder_pipeline.assert_tensor(
                            wrappers[custom.args[0].name][0], (2, 3)
                        )
                        builder_pipeline.assert_tensor(
                            wrappers[f"{custom.args[0].name}_to_1_0"][0], (3, 2)
                        )
                        builder_pipeline.assert_tensor(wrappers[custom.name][0], (3, 2))

        with subtests.test(msg="reuses_shared_graph_input_transform"):
            edge_program, nodes = builder_pipeline.export_custom(
                CustomOpAxisOrders._Single(
                    consumer=torch.ops.qnn_builder_test.bar.default
                ),
                (torch.randn(2, 3),),
            )
            for node in nodes:
                node.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: (1, 0)}
            _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)
            operations, wrappers = builder_pipeline.build(edge_program)
            assert len(operations) == 3
            builder_pipeline.assert_transpose(operations[0], (1, 0))
            assert nodes[0].args[0] is nodes[1].args[0]
            transforms = [name for name in wrappers if name.endswith("_to_1_0")]
            assert len(transforms) == 1
            builder_pipeline.assert_tensor(wrappers[transforms[0]][0], (3, 2))

        for matching in (False, True):
            with subtests.test(msg="tuple_chain", matching=matching):
                edge_program, producer, consumer = builder_pipeline.export_chain(
                    CustomOpAxisOrders._TupleChain(),
                    (torch.randn(2, 3),),
                    (1, 0),
                    (1, 0) if matching else (0, 1),
                )
                if not matching:
                    producer.meta.pop(QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS)
                    with pytest.raises(ValueError, match="same axis order"):
                        _passes.ValidateRuntimeConvFilterLayouts()(
                            edge_program.graph_module
                        )
                    producer.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: (1, 0)}
                    visitors = builder_pipeline.visitors(edge_program)
                    builder = visitors[consumer.target.__name__]
                    assert builder.define_node(consumer, defaultdict(dict)) is not None
                    wrappers = defaultdict(dict)
                    visitors[producer.target.__name__].define_node(producer, wrappers)
                    accessor = consumer.args[0]
                    visitors[accessor.target.__name__].define_node(accessor, wrappers)
                    with pytest.raises(ValueError, match="conflicting axis orders"):
                        builder.define_node(consumer, wrappers)
                    continue
                _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)
                operations, wrappers = builder_pipeline.build(edge_program)
                assert len(operations) == 2
                assert wrappers[consumer.args[0].name][0] is wrappers[producer.name][0]
                builder_pipeline.assert_tensor(wrappers[producer.name][0], (3, 2))

    @staticmethod
    def test_basic(builder_pipeline):
        edge_program, (custom,) = builder_pipeline.export_custom(
            CustomOpAxisOrders._Single(), (torch.randn(2, 3),)
        )
        assert QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS not in custom.meta
        assert QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS not in custom.meta
        operations, wrappers = builder_pipeline.build(edge_program)
        assert len(operations) == 1
        assert not any("_to_" in name for name in wrappers)
        for name in (custom.args[0].name, custom.name):
            builder_pipeline.assert_tensor(
                wrappers[name][0],
                (2, 3),
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            )

    @staticmethod
    def _test_input_metadata(builder_pipeline, subtests):
        with subtests.test(msg="missing_input_metadata"):
            edge_program, (node,) = builder_pipeline.export_custom(
                CustomOpAxisOrders._Single(), (torch.randn(2, 3),)
            )
            node.args[0].meta["val"] = None
            with pytest.raises(ValueError, match="input 0.*no tensor metadata"):
                builder_pipeline.build(edge_program)
        with subtests.test(msg="scalar_identity_axis_order"):
            edge_program, (node,) = builder_pipeline.export_custom(
                CustomOpAxisOrders._Single(), (torch.ones(()),)
            )
            node.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: ()}
            operations, wrappers = builder_pipeline.build(edge_program)
            assert len(operations) == 1
            assert not any("_to_" in name for name in wrappers)
            builder_pipeline.assert_tensor(wrappers[node.args[0].name][0], (1,))


class CustomOpCompatibleFanout:
    class _Model(torch.nn.Module):
        def __init__(self, custom_first, static=False, dtype=torch.float32):
            super().__init__()
            self.custom_first = custom_first
            self.static = static
            if static:
                self.register_buffer("weight", torch.ones(4, 3, 3, 3, dtype=dtype))

        def forward(self, x, weight=None):
            weight = (
                self.weight
                if self.static
                else torch.ops.qnn_builder_test.foo.default(weight)
            )
            if self.custom_first:
                other = torch.ops.qnn_builder_test.bar.default(weight)
            output = torch.nn.functional.conv2d(x, weight)
            if not self.custom_first:
                other = torch.ops.qnn_builder_test.bar.default(weight)
            return output, (
                other if self.static else torch.nn.functional.conv2d(x, other)
            )

    @staticmethod
    def test(builder_pipeline, subtests):
        for custom_first in (False, True):
            with subtests.test(custom_first=custom_first):
                edge_program, producer, consumer = builder_pipeline.export_chain(
                    CustomOpCompatibleFanout._Model(custom_first),
                    (torch.randn(1, 3, 5, 5), torch.randn(4, 3, 3, 3)),
                    QCOM_FILTER_AXIS_ORDER_HWIO,
                    QCOM_FILTER_AXIS_ORDER_HWIO,
                )
                consumer.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {
                    0: QCOM_FILTER_AXIS_ORDER_HWIO
                }
                _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)
                _passes.LayoutTransform(edge_program, insert_permute=True)(
                    edge_program.graph_module
                )
                operations, wrappers = builder_pipeline.build(edge_program)
                convolutions = [
                    node
                    for node in edge_program.graph.nodes
                    if node.target == exir_ops.edge.aten.convolution.default
                ]
                assert len(convolutions) == 2
                assert (
                    wrappers[convolutions[0].args[1].name][0]
                    is wrappers[producer.name][0]
                )
                assert wrappers[consumer.args[0].name][0] is wrappers[producer.name][0]
                builder_pipeline.assert_tensor(wrappers[producer.name][0], (3, 3, 3, 4))
                transposes = [
                    op.GetOpWrapper()
                    for op in operations
                    if op.GetOpWrapper().GetOpType() == OpTranspose.op_name
                ]
                assert all(producer.name not in op.GetName() for op in transposes)


class CustomOpQuantizedInputLayouts:
    class _QuantizedParameter(torch.nn.Module):
        def __init__(self, per_channel):
            super().__init__()
            self.per_channel = per_channel
            self.register_buffer("scales", torch.tensor([0.125, 0.25]))
            self.register_buffer("zero_points", torch.zeros(2, dtype=torch.int64))
            self.weight = torch.nn.Parameter(
                torch.randint(-8, 8, (2, 3), dtype=torch.int8), requires_grad=False
            )

        def forward(self, x):
            if self.per_channel:
                weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    self.weight, self.scales, self.zero_points, 0, -128, 127, torch.int8
                )
            else:
                weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    self.weight, 0.125, 0, -128, 127, torch.int8
                )
            x = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            return torch.ops.qnn_builder_test.foo.default(weight), x

    @staticmethod
    def test_qdq_parameter(builder_pipeline, qnn_compile_spec, pass_pipeline, subtests):
        for per_channel in (False, True):
            with subtests.test(per_channel=per_channel):
                edge_program = pass_pipeline.lower_edge_ep(
                    module=CustomOpQuantizedInputLayouts._QuantizedParameter(
                        per_channel
                    ),
                    sample_input=(torch.randn(2, 3),),
                    backend_type=QnnExecuTorchBackendType.kHtpBackend,
                    compile_spec=qnn_compile_spec,
                    target_pass=_passes.ValidateRuntimeConvFilterLayouts,
                    quantizer=None,
                )
                custom = next(
                    node
                    for node in edge_program.graph.nodes
                    if node.op == "call_function"
                    and node.target.__name__ == "qnn_builder_test.foo.default"
                )
                dequantize = custom.args[0]
                assert dequantize.target in dq_ops
                assert is_parameter(dequantize.args[0], edge_program)
                assert QCOM_QUANT_ATTRS in dequantize.args[0].meta
                custom.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: (1, 0)}
                _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)

    @staticmethod
    def test_static_lpbq_conv3d(builder_pipeline, qnn_compile_spec, pass_pipeline):
        quantizer = make_quantizer(quant_dtype=QuantDtype.use_16a4w_block)
        quantizer.set_block_size_map({"conv3d": (1, 32, 1, 1, 1)})
        edge_program = pass_pipeline.lower_edge_ep(
            module=torch.nn.Conv3d(32, 4, (2, 3, 3)),
            sample_input=(torch.randn(1, 32, 4, 6, 6),),
            backend_type=QnnExecuTorchBackendType.kHtpBackend,
            compile_spec=qnn_compile_spec,
            target_pass=_passes.LayoutTransform,
            quantizer=quantizer,
        )
        conv = next(
            node
            for node in edge_program.graph.nodes
            if node.target == exir_ops.edge.aten.convolution.default
        )
        operation = builder_pipeline.visitors(edge_program)[
            conv.target.__name__
        ].define_node(conv, defaultdict(dict))
        assert operation.GetOpWrapper().GetOpType() == OpConv3d.op_name
        weight = PyQnnManager.PyQnnTensorWrapper(
            operation.GetOpWrapper().GetInputTensors()[1]
        )
        assert tuple(weight.GetDims()) == (2, 3, 3, 32, 4)
        weight_config = operation.GetOpWrapper().GetOpConfig()["inputTensors"][1].v2
        assert (
            weight_config.type == PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC
        )
        assert (
            weight_config.quantizeParams.quantizationEncoding
            == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
        )

    @staticmethod
    def _assert_encoding(weight, dims, dtype, axis_order, attrs, original):
        transformed = PyQnnManager.PyQnnTensorWrapper(weight)
        assert transformed.GetDataType() == dtype
        assert tuple(transformed.GetDims()) == dims
        encoding = transformed.GetEncodings()
        axis = attrs.get(QCOM_AXIS)
        assert encoding.axis == (
            -1 if axis is None else axis_order.index(axis % len(axis_order))
        )
        if original is not None:
            assert original.GetDataType() == dtype
            np.testing.assert_array_equal(encoding.data, original.GetEncodings().data)
        if axis is not None:
            np.testing.assert_array_equal(encoding.data["scale"], attrs[QCOM_SCALES])
            np.testing.assert_array_equal(
                encoding.data["offset"], -attrs[QCOM_ZERO_POINTS]
            )

    @staticmethod
    def test(builder_pipeline, subtests):
        cases = (
            ("graph_input", None, torch.uint8, 0, 255),
            ("graph_input", -1, torch.int8, -128, 127),
            ("graph_input", 0, torch.int8, -8, 7),
            ("static_input", 0, torch.int8, -128, 127),
            ("custom_output", 0, torch.int8, -128, 127),
            ("shared_conv_first", 0, torch.int8, -128, 127),
            ("shared_custom_first", 0, torch.int8, -128, 127),
        )
        for path, axis, dtype, quant_min, quant_max in cases:
            with subtests.test(path=path, axis=axis, dtype=dtype, quant_min=quant_min):
                shared = path.startswith("shared_")
                isolated_weights = []
                if shared:
                    edge_program, (node,) = builder_pipeline.export_custom(
                        CustomOpCompatibleFanout._Model(
                            path == "shared_custom_first", static=True, dtype=dtype
                        ),
                        (torch.ones(1, 3, 5, 5, dtype=dtype),),
                    )
                    axis_order = QCOM_FILTER_AXIS_ORDER_HWIO
                    expected_dims = (3, 3, 3, 4)
                else:
                    edge_program, (node,) = builder_pipeline.export_custom(
                        CustomOpAxisOrders._Single(static=path == "static_input"),
                        (torch.randn(2, 3),),
                    )
                    axis_order, expected_dims = (1, 0), (3, 2)
                input_node = node.args[0]
                output = path == "custom_output"
                quant_node = node if output else input_node
                node.meta[
                    (
                        QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS
                        if output
                        else QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS
                    )
                ] = {0: axis_order}
                attrs = {
                    QCOM_DTYPE: dtype,
                    QCOM_QUANT_MIN: quant_min,
                    QCOM_QUANT_MAX: quant_max,
                }
                if axis is None:
                    attrs.update(
                        {
                            QCOM_ENCODING: exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                            QCOM_SCALE: 0.125,
                            QCOM_ZERO_POINT: 3,
                        }
                    )
                else:
                    channels = quant_node.meta["val"].shape[axis]
                    attrs.update(
                        {
                            QCOM_ENCODING: exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
                            QCOM_AXIS: axis,
                            QCOM_SCALES: torch.arange(1, channels + 1) * 0.125,
                            QCOM_ZERO_POINTS: torch.arange(channels),
                        }
                    )
                quant_node.meta[QCOM_QUANT_ATTRS] = attrs
                if shared:
                    _passes.ValidateRuntimeConvFilterLayouts()(
                        edge_program.graph_module
                    )
                    conv = next(
                        n
                        for n in edge_program.graph.nodes
                        if n.target == exir_ops.edge.aten.convolution.default
                    )
                    for isolated_node in (node, conv):
                        isolated_wrappers = defaultdict(dict)
                        builder_pipeline.visitors(edge_program)[
                            isolated_node.target.__name__
                        ].define_node(isolated_node, isolated_wrappers)
                        isolated_weights.append(isolated_wrappers[input_node.name][0])
                operations, wrappers = builder_pipeline.build(edge_program)
                if shared:
                    ops = [op.GetOpWrapper() for op in operations]
                    conv_op = next(
                        op for op in ops if op.GetOpType() == OpConv2d.op_name
                    )
                    custom_op = next(op for op in ops if op.GetName() == node.name)
                    assert (
                        conv_op.GetInputTensors()[1] is custom_op.GetInputTensors()[0]
                    )
                original = None
                if path == "graph_input":
                    builder_pipeline.assert_transpose(operations[0], (1, 0))
                    original = PyQnnManager.PyQnnTensorWrapper(
                        wrappers[input_node.name][0]
                    )
                elif not shared:
                    assert len(operations) == 1
                wrapper_name = (
                    f"{input_node.name}_to_1_0"
                    if path == "graph_input"
                    else quant_node.name
                )
                expected_dtype = (
                    PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8
                    if dtype == torch.uint8
                    else PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8
                )
                for weight in (*isolated_weights, wrappers[wrapper_name][0]):
                    CustomOpQuantizedInputLayouts._assert_encoding(
                        weight,
                        expected_dims,
                        expected_dtype,
                        axis_order,
                        attrs,
                        original,
                    )
                assert quant_node.meta[QCOM_QUANT_ATTRS].get(QCOM_AXIS) == axis


class RuntimeConvBackwardFilters:
    class _Single(torch.nn.Module):
        def __init__(self, mixed_dtypes=False):
            super().__init__()
            self.mixed_dtypes = mixed_dtypes

        def forward(self, grad_output, x, weight):
            dtypes = (
                (torch.float16, torch.float32)
                if self.mixed_dtypes
                else (grad_output.dtype,)
            )
            outputs = tuple(
                torch.ops.aten.convolution_backward.default(
                    grad_output.to(dtype),
                    x.to(dtype),
                    weight,
                    None,
                    [2, 2],
                    [1, 2],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    [True, False, False],
                )[0]
                for dtype in dtypes
            )
            return outputs if self.mixed_dtypes else outputs[0]

    @staticmethod
    def test(builder_pipeline, subtests):
        for axis_order, mixed_dtypes in (
            ((2, 3, 0, 1), False),
            ((2, 3, 1, 0), True),
        ):
            with subtests.test(axis_order=axis_order, mixed_dtypes=mixed_dtypes):
                weight_dtype = torch.float32 if mixed_dtypes else torch.float16
                edge_program = builder_pipeline.export(
                    RuntimeConvBackwardFilters._Single(mixed_dtypes),
                    (
                        torch.randn(1, 4, 4, 5),
                        torch.randn(1, 3, 7, 9),
                        torch.randn(4, 3, 3, 5, dtype=weight_dtype),
                    ),
                )
                conv_backwards = [
                    node
                    for node in edge_program.graph.nodes
                    if node.target == exir_ops.edge.aten.convolution_backward.default
                ]
                assert len(conv_backwards) == (2 if mixed_dtypes else 1)
                filter_node = conv_backwards[0].args[2]
                assert all(node.args[2] is filter_node for node in conv_backwards)
                filter_node.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: axis_order}
                _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)
                wrappers = defaultdict(dict)
                visitor = get_node_visitors(edge_program)[
                    "aten.convolution_backward.default"
                ]
                for conv_backward in conv_backwards:
                    ops = visitor.define_node(conv_backward, wrappers)
                    dtype = conv_backward.args[0].meta["val"].dtype
                    dtype_name = str(dtype).split(".")[-1]
                    hwoi = axis_order == (2, 3, 0, 1)
                    filter_name = (
                        f"{filter_node.name}_to_{dtype_name}"
                        if hwoi
                        else f"{filter_node.name}_{dtype_name}_hwoi"
                    )
                    builder_pipeline.assert_tensor(
                        wrappers[filter_name][0],
                        (3, 5, 4, 3),
                        (
                            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16
                            if dtype == torch.float16
                            else PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32
                        ),
                    )
                    needs_cast = filter_node.meta["val"].dtype != dtype
                    assert len(ops) == 1 + int(needs_cast) + int(not hwoi)
                    if needs_cast:
                        cast = ops[0].GetOpWrapper()
                        assert cast.GetOpType() == OpCast.op_name
                        cast_output = PyQnnManager.PyQnnTensorWrapper(
                            cast.GetOutputTensors()[0]
                        )
                        assert (
                            cast_output.GetName()
                            == f"{filter_node.name}_to_{dtype_name}"
                        )
                        assert cast.GetName() == f"{cast_output.GetName()}_cast"
                        builder_pipeline.assert_tensor(
                            cast.GetOutputTensors()[0],
                            (3, 5, 4, 3) if hwoi else (3, 5, 3, 4),
                            PyQnnManager.PyQnnTensorWrapper(
                                wrappers[filter_name][0]
                            ).GetDataType(),
                        )
                    if not hwoi:
                        builder_pipeline.assert_transpose(
                            ops[int(needs_cast)], (0, 1, 3, 2)
                        )
                if mixed_dtypes:
                    assert (
                        wrappers[f"{filter_node.name}_float16_hwoi"][0]
                        is not wrappers[f"{filter_node.name}_float32_hwoi"][0]
                    )


class CustomDequantOpWrapper:
    @staticmethod
    def test(builder_pipeline):
        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randint(-128, 128, (4, 3, 3, 3), dtype=torch.int8)
                )

            def forward(self, scale):
                return (
                    torch.ops.qnn_builder_test.dequant.default(self.weight, scale),
                    torch.ops.qnn_builder_test.dequant.default(self.weight, scale + 1),
                )

        edge_program, dequant_nodes = builder_pipeline.export_custom(
            _Model(), (torch.ones(4),)
        )
        assert len(dequant_nodes) == 2
        weight_node = dequant_nodes[0].args[0]
        assert dequant_nodes[1].args[0] is weight_node
        for dequant_node in dequant_nodes:
            dequant_node.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: (2, 3, 1, 0)}
            dequant_node.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: (2, 3, 1, 0)}
        operations, wrappers = builder_pipeline.build(edge_program)
        assert not any(
            op.GetOpWrapper().GetOpType() == OpTranspose.op_name for op in operations
        )
        assert f"{weight_node.name}_to_2_3_1_0" not in wrappers
        for node in dequant_nodes:
            builder_pipeline.assert_tensor(
                wrappers[node.name][0],
                (3, 3, 3, 4),
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
            )
        builder_pipeline.assert_tensor(
            wrappers[weight_node.name][0],
            (3, 3, 3, 4),
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_8,
        )


class CustomOpMixedOutputLayouts:
    class _Model(torch.nn.Module):
        def forward(self, x, weight):
            activation, filter = torch.ops.qnn_builder_test.pair_inputs.default(
                x, weight
            )
            return torch.nn.functional.conv2d(activation, filter), activation

    @staticmethod
    def test(builder_pipeline, subtests):
        inputs = (
            torch.randn(4, 3, 5, 7, dtype=torch.float16),
            torch.randn(6, 3, 3, 3, dtype=torch.float16),
        )
        for annotate_first in (False, True):
            with subtests.test(annotate_first=annotate_first):
                edge_program, (producer,) = builder_pipeline.export_custom(
                    CustomOpMixedOutputLayouts._Model(), inputs
                )
                producer.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {
                    1: QCOM_FILTER_AXIS_ORDER_HWIO
                }
                producer.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {
                    1: QCOM_FILTER_AXIS_ORDER_HWIO
                }
                _passes.ValidateRuntimeConvFilterLayouts()(edge_program.graph_module)
                if annotate_first:
                    _passes.LayoutTransform(edge_program)(edge_program.graph_module)
                _passes.LayoutTransform(edge_program, insert_permute=True)(
                    edge_program.graph_module
                )
                conv = next(
                    node
                    for node in edge_program.graph.nodes
                    if node.target == exir_ops.edge.aten.convolution.default
                )
                activation_permute = conv.args[0]
                assert (
                    activation_permute.target == exir_ops.edge.aten.permute_copy.default
                )
                activation = activation_permute.args[0]
                assert activation.args == (producer, 0)
                assert conv.args[1].args == (producer, 1)
                assert QCOM_AXIS_ORDER not in producer.meta
                assert QCOM_AXIS_ORDER not in activation.meta
                output = edge_program.graph.output_node()
                assert output.args[0][1] is activation
                operations, wrappers = builder_pipeline.build(edge_program)
                transpose = next(
                    op
                    for op in operations
                    if op.GetOpWrapper().GetName() == activation_permute.name
                )
                builder_pipeline.assert_transpose(transpose, (0, 2, 3, 1))
                builder_pipeline.assert_tensor(wrappers[producer.name][0], (4, 3, 5, 7))
                builder_pipeline.assert_tensor(
                    wrappers[activation_permute.name][0], (4, 5, 7, 3)
                )
                builder_pipeline.assert_tensor(wrappers[producer.name][1], (3, 3, 3, 6))


class RuntimeLayoutPartitioning:
    class _Conv(torch.nn.Module):
        def __init__(self, mode="backward"):
            super().__init__()
            self.mode = mode
            self.backward = RuntimeConvBackwardFilters._Single()

        def forward(self, grad_output, x, weight):
            weight = torch.ops.qnn_builder_test.sin.default(weight)
            output = torch.ops.aten.convolution.default(
                x, weight, None, [2, 2], [1, 2], [1, 1], False, [0, 0], 1
            )
            if self.mode == "fanout":
                return output, weight.sum(dim=(-2, -1), keepdim=True)
            return output, self.backward(grad_output, x, weight)

    class _Chain(torch.nn.Module):
        def forward(self, x):
            first = torch.ops.qnn_builder_test.sin.default(x)
            second = torch.ops.qnn_builder_test.sin.default(first)
            return second.sum(dim=1, keepdim=True)

    @staticmethod
    def test(builder_pipeline, runtime_layout_compile_spec, subtests):
        soc_model, specs = runtime_layout_compile_spec
        cases = (
            ("backward", "aten.convolution_backward.default", False),
            ("backward", "aten.convolution_backward.default", True),
            ("fanout", "aten.sum.dim_IntList", True),
            ("chain", "aten.sum.dim_IntList", True),
        )
        if soc_model == QcomChipset.SM8350:
            cases = cases[:1]
        for mode, skipped_op, skip in cases:
            with subtests.test(mode=mode, skip_consumer=skip):
                RuntimeLayoutPartitioning._test_partition(
                    builder_pipeline, soc_model, specs, mode, skipped_op, skip
                )
        if soc_model != QcomChipset.SM8350:
            RuntimeLayoutPartitioning._test_constant_ownership(
                builder_pipeline, specs, subtests
            )

    @staticmethod
    def _test_partition(builder_pipeline, soc_model, specs, mode, skipped_op, skip):
        module = (
            RuntimeLayoutPartitioning._Chain()
            if mode == "chain"
            else RuntimeLayoutPartitioning._Conv(mode)
        )
        inputs = (
            (torch.randn(2, 4),)
            if mode == "chain"
            else (
                torch.randn(1, 4, 4, 5),
                torch.randn(1, 3, 7, 9),
                torch.randn(4, 3, 3, 5),
            )
        )
        edge_program = builder_pipeline.export(module, inputs)
        nodes = [
            node for node in edge_program.graph.nodes if node.op == "call_function"
        ]
        _AnnotateCustomOpLayouts()(edge_program.graph_module)
        _passes.LayoutTransform(edge_program)(edge_program.graph_module)
        partitioner = QnnPartitioner(
            specs, skip_node_op_set={skipped_op} if skip else set()
        )
        partitioner.partition(edge_program)
        if skip or (mode == "backward" and soc_model == QcomChipset.SM8350):
            assert all("delegation_tag" not in node.meta for node in nodes)
            lowered = to_backend(edge_program, partitioner)
            assert not get_delegates(lowered.graph)
            torch.testing.assert_close(lowered.module()(*inputs), module(*inputs))
        else:
            tags = {node.meta.get("delegation_tag") for node in nodes}
            assert None not in tags
            assert len(tags) == 1

    class _StaticFilter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.randn(4, 3, 3, 3, dtype=torch.float16)
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.register_buffer("scale", torch.ones_like(weight))

        def forward(self, x):
            weight = torch.ops.qnn_builder_test.multiply.default(
                self.weight, self.scale
            )
            return torch.ops.aten.convolution.default(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
            )

    @staticmethod
    def _test_constant_ownership(builder_pipeline, custom_op_compile_spec, subtests):
        with subtests.test(msg="no_copy_constant_fallback"):
            module = RuntimeLayoutPartitioning._StaticFilter()
            inputs = (torch.randn(1, 3, 8, 8, dtype=torch.float16),)
            edge_program, (producer,) = builder_pipeline.export_custom(module, inputs)
            weight, scale = producer.args
            weight.meta["no_copy"] = True
            producer.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {
                0: QCOM_FILTER_AXIS_ORDER_HWIO,
                1: QCOM_FILTER_AXIS_ORDER_HWIO,
            }
            producer.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {
                0: QCOM_FILTER_AXIS_ORDER_HWIO
            }
            producer.meta[QCOM_RUNTIME_INPUTS] = (1,)
            _passes.LayoutTransform(edge_program)(edge_program.graph_module)
            partitioner = QnnPartitioner(
                custom_op_compile_spec,
                skip_node_op_set={"aten.convolution.default"},
            )
            partitioner.partition(edge_program)
            assert is_runtime_input(scale)
            assert all(
                "delegation_tag" not in node.meta for node in (weight, scale, producer)
            )
            lowered = to_backend(edge_program, partitioner)
            assert not get_delegates(lowered.graph)
            torch.testing.assert_close(lowered.module()(*inputs), module(*inputs))


class CustomOpRuntimeInputs:
    class _Scale(torch.nn.Module):
        def __init__(self, use_parameter=False, dtype=torch.float32, mutation=None):
            super().__init__()
            self.mutation = mutation
            scale = torch.ones(4, dtype=dtype)
            if use_parameter:
                self.scale = torch.nn.Parameter(scale)
            else:
                self.register_buffer("scale", scale)

        def forward(self, x=None):
            out = (self.scale if x is None else x) * self.scale
            if self.mutation == "add":
                self.scale.add_(1)
            elif self.mutation == "copy":
                self.scale.copy_(x)
            return out

    @staticmethod
    def _export_scale(
        use_parameter=False,
        dtype=torch.float32,
        mutation=None,
        repeated=False,
        runtime_indices=None,
    ):
        module = CustomOpRuntimeInputs._Scale(use_parameter, dtype, mutation)
        inputs = () if repeated else (torch.ones(4, dtype=dtype),)
        edge = to_edge(
            torch.export.export(module, inputs),
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        )
        mul = next(
            node
            for node in edge.exported_program().graph.nodes
            if node.target == exir_ops.edge.aten.mul.Tensor
        )
        scale = mul.args[1]
        mul.meta[QCOM_RUNTIME_INPUTS] = (
            tuple(i for i, arg in enumerate(mul.args) if arg is scale)
            if runtime_indices is None
            else runtime_indices
        )
        return edge, scale, mul

    @staticmethod
    def test(builder_pipeline, qnn_compile_spec, subtests):
        for use_parameter in (False, True):
            with subtests.test(msg="externalized", use_parameter=use_parameter):
                edge, scale, mul = CustomOpRuntimeInputs._export_scale(use_parameter)
                exported = edge.exported_program()
                mark_runtime_inputs(exported)
                assert is_runtime_input(scale)
                assert not is_parameter(scale, exported)
                partitioner = QnnPartitioner(qnn_compile_spec)
                partitioner.partition(exported)
                assert "delegation_tag" in mul.meta
                assert "delegation_tag" not in scale.meta

                delegated = edge.to_backend(partitioner).exported_program()
                delegates = get_delegates(delegated.graph)
                assert len(delegates) == 1
                lowered_node = delegates[0]
                lowered_module = getattr(delegated.graph_module, lowered_node.target)
                delegate_program = lowered_module.original_module
                call_delegate = next(iter(lowered_node.users))
                delegate_arg_names = tuple(
                    arg.name
                    for arg in call_delegate.args[1:]
                    if isinstance(arg, torch.fx.Node)
                )
                assert scale.name not in delegated.graph_signature.user_inputs
                assert scale.name in delegate_program.graph_signature.user_inputs
                assert delegate_arg_names == tuple(
                    delegate_program.graph_signature.user_inputs
                )

        with subtests.test(msg="runtime_input_survives_i64_cast"):
            edge, scale, _ = CustomOpRuntimeInputs._export_scale(dtype=torch.int64)
            mark_runtime_inputs(edge.exported_program())
            exported = edge.transform(
                [_passes.I64toI32(edge.exported_program())]
            ).exported_program()
            scale = next(
                node for node in exported.graph.nodes if node.name == scale.name
            )
            assert is_runtime_input(scale)
            assert not is_parameter(scale, exported)
            cast = next(iter(scale.users))
            assert cast.target in (
                exir_ops.edge.aten._to_copy.default,
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            )
            assert cast.kwargs["dtype"] == torch.int32
            QnnPartitioner(qnn_compile_spec).partition(exported)
            assert is_runtime_input(scale)
            assert not is_parameter(scale, exported)
            assert "delegation_tag" not in scale.meta
            assert is_runtime_input(cast)
            assert "delegation_tag" in cast.meta

        with subtests.test(msg="runtime_parameter_keeps_htp_partition"):
            edge, scale, mul = CustomOpRuntimeInputs._export_scale(repeated=True)
            QnnPartitioner(qnn_compile_spec).partition(edge.exported_program())
            assert "delegation_tag" in mul.meta
            assert is_runtime_input(scale)

        for mutation in ("add", "copy"):
            with subtests.test(msg="reject_mutable_buffer", mutation=mutation):
                edge, _, _ = CustomOpRuntimeInputs._export_scale(mutation=mutation)
                with pytest.raises(ValueError, match="must not be a mutable buffer"):
                    mark_runtime_inputs(edge.exported_program())
        with subtests.test(msg="reject_partially_marked_repeated_input"):
            edge, _, _ = CustomOpRuntimeInputs._export_scale(
                repeated=True, runtime_indices=(0,)
            )
            with pytest.raises(ValueError, match="must not have non-runtime users"):
                mark_runtime_inputs(edge.exported_program())

        CustomOpRuntimeInputs._test_intermediates(builder_pipeline, qnn_compile_spec)

    class _Intermediate(torch.nn.Module):
        def forward(self, x):
            y = torch.sin(x)
            return torch.cos(y), y

    @staticmethod
    def _test_intermediates(builder_pipeline, qnn_compile_spec):
        edge_program = builder_pipeline.export(
            CustomOpRuntimeInputs._Intermediate(),
            (torch.randn(2, 4, dtype=torch.float16),),
        )
        sin = next(
            node
            for node in edge_program.graph.nodes
            if node.target == exir_ops.edge.aten.sin.default
        )
        cos = next(iter(sin.users))
        cos.meta[QCOM_RUNTIME_INPUTS] = (0,)
        partitioner = QnnPartitioner(qnn_compile_spec)
        partitioner.partition(edge_program)
        assert is_runtime_input(sin)
        assert sin.meta.get("delegation_tag") is not None
        assert sin.meta["delegation_tag"] == cos.meta["delegation_tag"]
        lowered = to_backend(edge_program, partitioner)
        delegates = get_delegates(lowered.graph)
        assert len(delegates) == 1
        delegate = getattr(lowered.graph_module, delegates[0].target)
        targets = {node.target for node in delegate.original_module.graph.nodes}
        assert exir_ops.edge.aten.sin.default in targets
        assert exir_ops.edge.aten.cos.default in targets


class CustomOpLayoutEndToEnd:
    class _Model(RuntimeLayoutPartitioning._Conv):
        def forward(self, grad_output, x, weight):
            output, grad_input = super().forward(grad_output, x, weight)
            return output, grad_input, torch.cos(weight)

    @staticmethod
    def test(custom_op_compile_spec, qnn_config):
        module = CustomOpLayoutEndToEnd._Model()
        inputs = (
            torch.randn(1, 4, 4, 5, dtype=torch.float16) * 0.2,
            torch.randn(1, 3, 7, 9, dtype=torch.float16) * 0.2,
            torch.randn(4, 3, 3, 5, dtype=torch.float16) * 0.1,
        )
        pass_manager = get_qnn_pass_manager_cls(QnnExecuTorchBackendType.kHtpBackend)
        passes = pass_manager.get_capture_program_passes()
        passes[_AnnotateCustomOpLayouts] = {
            QCOM_PASS_ACTIVATE_KEY: True,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
        }
        dependencies = pass_manager.get_passes_dependency_for_capture_program()
        dependencies[_passes.LayoutTransform] = [
            *dependencies.get(_passes.LayoutTransform, []),
            _AnnotateCustomOpLayouts,
        ]
        lowered = to_edge_transform_and_lower_to_qnn(
            module,
            inputs,
            custom_op_compile_spec,
            passes_job=passes,
            dep_table=dependencies,
        )
        delegates = get_delegates(lowered.exported_program().graph)
        assert len(delegates) == 1
        executable = lowered.to_executorch()
        if qnn_config.compile_only:
            return
        verify = (
            verify_output_emulator if qnn_config.enable_x86_64 else verify_output_remote
        )
        verify(
            module=module,
            inputs=inputs,
            executorch_prog=executable,
            metrics=Tolerance(rtol=1e-2, atol=1e-3),
            qnn_config=qnn_config,
        )
