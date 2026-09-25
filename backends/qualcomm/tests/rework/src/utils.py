# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
import os
import tempfile
from collections import defaultdict
from functools import reduce
from typing import Tuple

import pytest

import torch

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.debugger.utils import DrawGraph as _DrawGraphTool
from executorch.backends.qualcomm.export_utils import (
    convert_pt2e,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    make_quantizer,
    prepare_pt2e,
    prepare_qat_pt2e,
    QcomChipset,
    QuantDtype,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.backends.qualcomm.partition.utils import (
    warn_if_dilated_conv_may_not_fit_vtcm,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    calibrate,
    check_exception,
    default_property,
    EXCEPTION_FROM_PREPROCESS,
)
from executorch.backends.qualcomm.tests.utils import validate_context_binary
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    dump_context_from_pte,
    rewrite_prepared_observer,
    skip_annotation,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


# ---------------------------------------------------------------------------
# Shared model — multi-input / multi-output
# forward(x, y) -> (Tensor, Tensor)
#   x -> conv1 -> relu1 -> a ─┐
#                              add -> conv3 -> c   (outputs: c, a)
#   y -> conv2 -> relu2 -> b ─┘
# Provides: two inputs, two outputs, two conv2d ops (SeqMSE), one add (skip target)
# ---------------------------------------------------------------------------


class _UtilsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x, y):
        a = self.relu1(self.conv1(x))
        b = self.relu2(self.conv2(y))
        c = self.conv3(a + b)
        return c, a


class _CompositeDelegateModule(torch.nn.Module):
    def __init__(
        self,
        compiler_specs,
        to_edge_transform_and_lower_method,
        quantize_method=None,
    ) -> None:
        super().__init__()
        self.modules = [
            _UtilsModel(),
            _UtilsModel(),
        ]
        self.sample_inputs = [
            (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)),
            (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)),
        ]
        self.lowered_modules = []
        for module, sample_input in zip(self.modules, self.sample_inputs):
            if quantize_method:
                module = quantize_method(module, sample_input)
            edge_prog = to_edge_transform_and_lower_method(
                module, sample_input, compiler_specs
            )
            self.lowered_modules.append(
                edge_prog.exported_program().graph_module._modules.get(
                    "lowered_module_0"
                )
            )

    def forward(self, x1, y1, x2, y2):
        z11, z12 = self.lowered_modules[0](x1, y1)
        z21, z22 = self.lowered_modules[1](x2, y2)
        return z11 + z21, z12 + z22

    def get_random_input(self):
        return tuple(e for tup in self.sample_inputs for e in tup)

    def get_reference_module(self):
        class CompositeReferenceModule(torch.nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.modules = modules

            def forward(self, x1, y1, x2, y2):
                z11, z12 = self.modules[0](x1, y1)
                z21, z22 = self.modules[1](x2, y2)
                return z11 + z21, z12 + z22

        return CompositeReferenceModule(self.modules)


# ---------------------------------------------------------------------------
# Test Bodies
# ---------------------------------------------------------------------------


class DumpContextFromPte:
    @staticmethod
    def test(quantizer, compile_spec):
        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        with calibrate(module, [inputs], quantizer) as quantized:
            exec_prog = to_edge_transform_and_lower_to_qnn(
                quantized, inputs, compile_spec
            ).to_executorch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as f:
                exec_prog.write_to_file(f)
            dump_context_from_pte(pte_path)
            binary_path = f"{tmp_dir}/forward_0.bin"
            assert os.path.isfile(binary_path)
            with open(binary_path, "rb") as f:
                validate_context_binary(f.read())


class DrawGraph:
    _golden = """digraph test {
        rankdir=TB
        "input_0_x@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightgreen">name: input_0_x@0</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "quantized_decomposed_quantize_per_tensor_default@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: quantized_decomposed_quantize_per_tensor_default@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "input_1_y@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightgreen">name: input_1_y@0</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "quantized_decomposed_quantize_per_tensor_default_1@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: quantized_decomposed_quantize_per_tensor_default_1@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param0@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param0@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [3, 3, 4, 4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param1@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param1@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_convolution_default@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_convolution_default@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_relu_default@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_relu_default@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "output_quantized_decomposed_dequantize_per_tensor_default@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightgreen">name: output_quantized_decomposed_dequantize_per_tensor_default@0</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param2@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param2@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [3, 3, 4, 4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param3@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param3@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_convolution_default_1@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_convolution_default_1@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_relu_default_1@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_relu_default_1@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_add_tensor@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_add_tensor@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param4@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param4@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [1, 1, 4, 4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "b__frozen_param5@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightpink">name: b__frozen_param5@0</TD></TR>
                    <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32</TD></TR>
                    <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                    <TR><TD BGCOLOR="lightpink">dims: [4]</TD></TR>
                    <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "aten_convolution_default_2@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="white">name: aten_convolution_default_2@0</TD></TR>
                    <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                    <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                    <TR><TD BGCOLOR="white">dims: [1, 8, 8, 4]</TD></TR>
                    <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "output_quantized_decomposed_dequantize_per_tensor_default_1@0" [label=<
                    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                    <TR><TD BGCOLOR="lightgreen">name: output_quantized_decomposed_dequantize_per_tensor_default_1@0</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">dims: [1, 4, 8, 8]</TD></TR>
                    <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
        "input_0_x@0" -> "quantized_decomposed_quantize_per_tensor_default@0"
        "input_1_y@0" -> "quantized_decomposed_quantize_per_tensor_default_1@0"
        "quantized_decomposed_quantize_per_tensor_default@0" -> "aten_convolution_default@0"
        "b__frozen_param0@0" -> "aten_convolution_default@0"
        "b__frozen_param1@0" -> "aten_convolution_default@0"
        "aten_convolution_default@0" -> "aten_relu_default@0"
        "aten_relu_default@0" -> "output_quantized_decomposed_dequantize_per_tensor_default@0"
        "quantized_decomposed_quantize_per_tensor_default_1@0" -> "aten_convolution_default_1@0"
        "b__frozen_param2@0" -> "aten_convolution_default_1@0"
        "b__frozen_param3@0" -> "aten_convolution_default_1@0"
        "aten_convolution_default_1@0" -> "aten_relu_default_1@0"
        "aten_relu_default@0" -> "aten_add_tensor@0"
        "aten_relu_default_1@0" -> "aten_add_tensor@0"
        "aten_add_tensor@0" -> "aten_convolution_default_2@0"
        "b__frozen_param4@0" -> "aten_convolution_default_2@0"
        "b__frozen_param5@0" -> "aten_convolution_default_2@0"
        "aten_convolution_default_2@0" -> "output_quantized_decomposed_dequantize_per_tensor_default_1@0"
    }
    """

    @staticmethod
    def _build_op_wrapper_list(module, inputs):
        delegated_program = capture_program(module, inputs)
        graph_module = get_qnn_pass_manager_cls()().transform_for_preprocess_pipeline(
            delegated_program.exported_program
        )
        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            delegated_program.exported_program, enable_tensor_dump=False
        )
        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target.__name__ in node_visitors:
                wrapper = node_visitors[node.target.__name__].define_node(
                    node, nodes_to_wrappers
                )
                if wrapper is not None:
                    if isinstance(wrapper, list):
                        py_op_wrapper_list.extend(wrapper)
                    else:
                        py_op_wrapper_list.append(wrapper)
        return py_op_wrapper_list

    @staticmethod
    def test(quantizer):
        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        with calibrate(module, [inputs], quantizer) as quantized_module:
            op_wrappers = __class__._build_op_wrapper_list(quantized_module, inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            _DrawGraphTool("test", tmp_dir, op_wrappers, dot_string=True)
            with open(os.path.join(tmp_dir, "test.dot")) as f:
                result = f.read()
            assert sorted(__class__._golden.split()) == sorted(
                result.split()
            ), "Generated .dot file does not match the golden file."


class DilatedConvVtcmWarning:
    """The partitioner's advisory warning for dilated convs that may not tile.

    fp16 HTP does not reject a dilated convolution it failed to tile into VTCM;
    it emits a context binary that stops the DSP at execute. The partitioner
    therefore logs a warning naming the operator, without changing whether it is
    delegated. See https://github.com/pytorch/executorch/issues/23096.
    """

    _VTCM_8MB = 8

    @staticmethod
    def _conv_node(cin, cout, kernel, dilation, hw, groups=1, padding=None):
        """One `aten.convolution` node with the metadata the check reads.

        Traced rather than hand-built so the argument order and `val` metadata
        come from the real decomposition. Meta device: only shapes are needed,
        and these convs reach several MB of weights.
        """
        if padding is None:
            padding = dilation * (kernel - 1) // 2

        class _Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                )

            def forward(self, x):
                return self.conv(x)

        with torch.device("meta"):
            module = _Conv().eval()
            inputs = (torch.randn(1, cin, hw, hw),)

        edge = torch.export.export(module, inputs, strict=True).run_decompositions()
        for node in edge.graph_module.graph.nodes:
            if node.target is torch.ops.aten.convolution.default:
                # The partitioner sees the edge dialect's overload.
                node.target = exir_ops.edge.aten.convolution.default
                return node
        raise AssertionError("no convolution node in the exported graph")

    @staticmethod
    def test(subtests, caplog):
        vtcm = __class__._VTCM_8MB
        # The reported convolution: 960 -> 256, 3x3, dilation 36, 65x65 input.
        reported = {"cin": 960, "cout": 256, "kernel": 3, "dilation": 36, "hw": 65}

        with subtests.test(msg="warns_and_names_the_operator"):
            node = __class__._conv_node(**reported)
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                estimate = warn_if_dilated_conv_may_not_fit_vtcm(node, vtcm)
            assert estimate is not None
            # Must be actionable: the node, both byte counts, and the issue.
            for expected in (
                node.name,
                str(estimate),
                str(vtcm * 1024 * 1024),
                "23096",
            ):
                assert expected in caplog.text

        # Anything that is not the pathological shape must stay silent, or the
        # warning becomes noise in every export log.
        silent_cases = [
            # The Conv2d shape the op tests already sweep.
            (
                "small_dilated",
                {"cin": 1, "cout": 6, "kernel": 3, "dilation": 2, "hw": 8},
                vtcm,
            ),
            # Wide and heavy, but dilation 1 tiles down to the kernel.
            ("undilated", {**reported, "dilation": 1, "padding": 1}, vtcm),
            # Depthwise: the channel volume driving the failure is not present.
            (
                "grouped",
                {
                    "cin": 960,
                    "cout": 960,
                    "kernel": 3,
                    "dilation": 36,
                    "hw": 65,
                    "groups": 960,
                },
                vtcm,
            ),
            # SocInfo defaults vtcm_size_in_mb to 0, which means "unknown" and
            # must not be read as "no space available".
            ("unknown_vtcm_size", reported, 0),
        ]
        for label, kwargs, vtcm_mb in silent_cases:
            with subtests.test(msg=label):
                node = __class__._conv_node(**kwargs)
                with caplog.at_level(logging.WARNING):
                    caplog.clear()
                    assert (
                        warn_if_dilated_conv_may_not_fit_vtcm(node, vtcm_mb) is None
                    ), f"{label} should not warn"
                assert caplog.text == ""

        with subtests.test(msg="threshold_follows_the_soc"):
            # 4 MB parts (e.g. SAR2230P, SW6100) hit this at lower cin, so the
            # check must read VTCM from the SoC rather than hardcode 8 MB.
            node = __class__._conv_node(cin=256, cout=64, kernel=3, dilation=36, hw=65)
            assert warn_if_dilated_conv_may_not_fit_vtcm(node, 8) is None
            assert warn_if_dilated_conv_may_not_fit_vtcm(node, 1) is not None


class DilatedConvTcmFitSentinel:
    """Alarm for when HTP learns to tile the convolution from #23096.

    This asserts the *desired* end state -- that the reported convolution
    compiles -- so it fails today and is marked xfail at the call site. When the
    backend fix lands it starts passing, and `xfail(strict=True)` turns that
    XPASS into a failure, which is the notification to delete this, the
    partitioner warning in `warn_if_dilated_conv_may_not_fit_vtcm`, and the
    troubleshooting entry in `docs/source/backends-qualcomm.md`.

    a16w8 rather than fp16 because the quantized path is where the TCM-fit check
    actually runs: it fails on the x86 host at finalize with `not sufficiently
    tiled to fit in TCM`, so no device is involved. The fp16 path skips the check
    and only fails on-device, which would mean deliberately hanging a DSP on
    every CI run.

    Two things this does not cover:

    * It only fires when the test runs against a QNN SDK carrying the fix. With
      a pinned SDK version it stays xfail however long the backend takes, so the
      real trigger is an SDK upgrade rather than the fix itself.
    * It watches the tiling defect. If the backend instead only makes fp16
      perform the check it currently skips -- the cheaper of the two fixes --
      a16w8 keeps failing and this stays xfail, even though the warning has by
      then become redundant, since users would get a compile error.

    It is also a sentinel rather than a correctness test: it flips for any change
    that stops this shape raising, including one in the quantizer path or in how
    the SDK reports the failure.
    """

    @staticmethod
    def test():
        # The reported shape. padding == dilation keeps the output spatially the
        # same size as the input, which is what ASPP does.
        module = torch.nn.Sequential(
            torch.nn.Conv2d(
                960, 256, kernel_size=3, padding=36, dilation=36, bias=False
            )
        ).eval()
        inputs = (torch.randn(1, 960, 65, 65),)

        quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            soc_model=default_property().soc_model,
        )
        compile_spec = generate_qnn_executorch_compiler_spec(
            soc_model=getattr(QcomChipset, default_property().soc_model),
            backend_options=generate_htp_compiler_spec(use_fp16=False),
        )

        with calibrate(module, [inputs], quantizer) as quantized:
            # Raises today: graph prepare cannot tile this into VTCM and reports
            # `Requires 8638464 bytes` against 8388608 available on an 8 MB part.
            to_edge_transform_and_lower_to_qnn(quantized, inputs, compile_spec)


class FixedPointFloatingPointMixedPrecision:
    # test specifically for graph which owns floating point / fixed point
    # operators simultaneously
    @staticmethod
    def test(subtests, quantizer, compile_spec):
        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        def calibrator(gm):
            gm(*inputs)

        for fallback_to_cpu, expected_partitions in [
            (True, 2),
            (False, 3),
        ]:
            with subtests.test(msg=f"fallback_to_cpu={fallback_to_cpu}"):
                _, edge_prog_mgrs = skip_annotation(
                    nn_module=module,
                    quantizer=quantizer,
                    compiler_specs=compile_spec,
                    sample_input=inputs,
                    calibration_cb=calibrator,
                    fp_node_id_set={"add"},
                    fallback_to_cpu=fallback_to_cpu,
                )
                assert len(edge_prog_mgrs) == expected_partitions


class MultiContextsComposite:
    @staticmethod
    def test(compile_spec):
        module = _CompositeDelegateModule(
            compiler_specs=compile_spec,
            to_edge_transform_and_lower_method=to_edge_transform_and_lower_to_qnn,
        )
        sample_input = module.get_random_input()
        edge_prog = to_edge(
            torch.export.export(module, sample_input, strict=True),
        )
        # should complete without error
        edge_prog.to_executorch()


class RewritePreparedObserver:
    @staticmethod
    def test(quantizer):
        import math

        from torchao.quantization.pt2e import FixedQParamsObserver

        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
        exported = torch.export.export(module, inputs, strict=True).module()
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*inputs)

        new_obs = FixedQParamsObserver(
            scale=0.004,
            zero_point=0,
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
        )
        rewrite_prepared_observer(prepared, {"activation_post_process_3": new_obs})
        assert (
            prepared.activation_post_process_3 is new_obs
        ), "observer is not overridden correctly"
        # should complete without error
        converted = convert_pt2e(prepared)
        q_node = [
            n
            for n in converted.graph.nodes
            if n.name == "quantize_per_tensor_default_2"
        ][0]
        assert (
            math.isclose(q_node.args[1], 0.004, abs_tol=1e-8) and q_node.args[2] == 0
        ), "scale / offset do not match the overridden values"


class SkipNodePartitioner:
    @staticmethod
    def _count_lowered_modules(edge_prog_mgr):
        gm = edge_prog_mgr.exported_program().graph_module
        return len([k for k in gm._modules if k.startswith("lowered_module")])

    @staticmethod
    def test(subtests, quantizer, compile_spec):
        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        cases = [
            # skip add by node id: [conv1+relu1 / conv2+relu2], [conv3] → 2 partitions
            ("node_id", {"skip_node_id_set": {"aten_add_tensor"}}, 2),
            # skip add by op: same split → 2 partitions
            ("node_op", {"skip_node_op_set": {"aten.add.Tensor"}}, 2),
        ]

        for label, skip_kwargs, expected in cases:
            with subtests.test(msg=label):
                with calibrate(module, [inputs], quantizer) as quantized:
                    edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                        quantized, inputs, compile_spec, **skip_kwargs
                    )
                assert __class__._count_lowered_modules(edge_prog_mgr) == expected

        # expected failure due to fallback per-channel weight required
        # a per-chennel dequantized op which is unavailable in QNN
        cases = [
            ("node_id", {"skip_node_id_set": {"aten_convolution_default"}}, 2),
        ]
        for label, skip_kwargs, _ in cases:
            with subtests.test(msg=label):
                with pytest.raises(  # noqa: B017
                    Exception, check=check_exception(EXCEPTION_FROM_PREPROCESS)
                ):
                    with calibrate(module, [inputs], quantizer) as quantized:
                        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                            quantized, inputs, compile_spec, **skip_kwargs
                        )


class SkipNodeQuantizer:
    @staticmethod
    def test(subtests, quantizer, compile_spec):
        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        def calibrator(gm):
            gm(*inputs)

        cases = [
            ("node_id", {"fp_node_id_set": {"add"}}, 2),
            ("node_id", {"fp_node_id_set": {"conv2d"}}, 1),
            ("node_op", {"fp_node_op_set": {torch.ops.aten.add.Tensor}}, 2),
        ]

        for label, skip_kwargs, expected_partitions in cases:
            with subtests.test(msg=label):
                graph_module, edge_prog_mgrs = skip_annotation(
                    nn_module=module,
                    quantizer=quantizer,
                    compiler_specs=compile_spec,
                    sample_input=inputs,
                    calibration_cb=calibrator,
                    **skip_kwargs,
                )
                assert len(edge_prog_mgrs) == expected_partitions

                # skipped nodes must not be annotated
                skipped_targets = set(skip_kwargs.get("fp_node_id_set", set())) | {
                    op.__name__ if hasattr(op, "__name__") else str(op)
                    for op in skip_kwargs.get("fp_node_op_set", set())
                }
                for node in graph_module.graph.nodes:
                    name_matches = node.name in skipped_targets
                    op_matches = (
                        hasattr(node.target, "__name__")
                        and node.target.__name__ in skipped_targets
                    ) or str(node.target) in skipped_targets
                    if name_matches or op_matches:
                        annotated = (
                            Q_ANNOTATION_KEY in node.meta
                            and node.meta[Q_ANNOTATION_KEY]._annotated
                        )
                        assert (
                            not annotated
                        ), f"node {node.name} should not be annotated"


class QAT:
    @staticmethod
    def _make_qat_quantizer(quant_dtype, block_size_map=None):
        q = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            is_qat=True,
            soc_model="SM8650",
        )
        if block_size_map:
            q.set_block_size_map(block_size_map)
        return q

    @staticmethod
    def _get_converted_module(
        ori_module: torch.nn.Module,
        prepared: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
    ) -> torch.fx.GraphModule:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        output = prepared(*inputs)
        loss = reduce(
            operator.add,
            [criterion(qdq, ref) for qdq, ref in zip(output, ori_module(*inputs))],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return convert_pt2e(prepared)

    @staticmethod
    def test(subtests):
        from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
            PerBlockParamFakeQuantize,
        )
        from torchao.quantization.pt2e import FusedMovingAvgObsFakeQuantize

        module = _UtilsModel()
        inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))

        with subtests.test(msg="16a4w_pcq"):
            # activation: FusedMovingAvgObsFakeQuantize (uint16 range, per-tensor)
            # weight: FusedMovingAvgObsFakeQuantize (int4 range [-7,7], per-channel)
            q = __class__._make_qat_quantizer(QuantDtype.use_16a4w)
            exported = torch.export.export(module, inputs, strict=True).module()
            prepared = prepare_qat_pt2e(exported, q)
            fq_modules = [
                m
                for m in prepared.modules()
                if isinstance(m, FusedMovingAvgObsFakeQuantize)
            ]
            assert len(fq_modules) > 0, "no FusedMovingAvgObsFakeQuantize found"
            # activation FQs have uint16 range (quant_max=65535)
            act_fqs = [m for m in fq_modules if m.quant_max == 65535]
            # weight FQs have int4 range (quant_max=7)
            weight_fqs = [m for m in fq_modules if m.quant_max == 7]
            assert len(act_fqs) > 0, "no uint16-range activation FQ found for 16a4w"
            assert len(weight_fqs) > 0, "no int4-range weight FQ found for 16a4w"
            # should complete without error
            __class__._get_converted_module(module, prepared, inputs)

        with subtests.test(msg="16a4w_block"):
            # activation: FusedMovingAvgObsFakeQuantize (uint16 range)
            # weight: PerBlockParamFakeQuantize (int4 range)
            q = __class__._make_qat_quantizer(
                QuantDtype.use_16a4w_block,
                block_size_map={"conv2d": (1, 4, 1, 1)},
            )
            exported = torch.export.export(module, inputs, strict=True).module()
            prepared = prepare_qat_pt2e(exported, q)
            act_fqs = [
                m
                for m in prepared.modules()
                if isinstance(m, FusedMovingAvgObsFakeQuantize) and m.quant_max == 65535
            ]
            block_fqs = [
                m
                for m in prepared.modules()
                if isinstance(m, PerBlockParamFakeQuantize)
            ]
            assert (
                len(act_fqs) > 0
            ), "no uint16-range activation FQ found for 16a4w_block"
            assert (
                len(block_fqs) > 0
            ), "no PerBlockParamFakeQuantize found for 16a4w_block"
            # should complete without error
            __class__._get_converted_module(module, prepared, inputs)

        with subtests.test(msg="fp16a8w"):
            # activation: None (FP16 — no activation quantization)
            # weight: FusedMovingAvgObsFakeQuantize (int8 range [-127,127], per-tensor or per-channel)
            q = __class__._make_qat_quantizer(QuantDtype.use_fp16a8w)
            exported = torch.export.export(module, inputs, strict=True).module()
            prepared = prepare_qat_pt2e(exported, q)
            fq_modules = [
                m
                for m in prepared.modules()
                if isinstance(m, FusedMovingAvgObsFakeQuantize)
            ]
            # no activation FQ: quant_max==65535 or quant_max==255 nodes must be absent
            act_fqs = [m for m in fq_modules if m.quant_max in (255, 65535)]
            # weight FQ: int8 range (quant_max=127)
            weight_fqs = [m for m in fq_modules if m.quant_max == 127]
            assert len(act_fqs) == 0, "fp16a8w should have no activation FQ"
            assert len(weight_fqs) > 0, "no int8-range weight FQ found for fp16a8w"
            # should complete without error
            __class__._get_converted_module(module, prepared, inputs)
