# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.tests.rework.src.builders import *  # noqa: F403


def test_custom_op_without_axis_orders(builder_pipeline):
    CustomOpAxisOrders.test_basic(builder_pipeline)  # noqa: F405


def test_custom_op_axis_orders(builder_pipeline, subtests):
    CustomOpAxisOrders.test(builder_pipeline, subtests)  # noqa: F405


def test_custom_op_compatible_fanout(builder_pipeline, subtests):
    CustomOpCompatibleFanout.test(builder_pipeline, subtests)  # noqa: F405


def test_custom_op_quantized_input_layouts(builder_pipeline, subtests):
    CustomOpQuantizedInputLayouts.test(builder_pipeline, subtests)  # noqa: F405


def test_custom_op_qdq_parameter_layouts(
    builder_pipeline, qnn_compile_spec, pass_pipeline, subtests
):
    CustomOpQuantizedInputLayouts.test_qdq_parameter(  # noqa: F405
        builder_pipeline, qnn_compile_spec, pass_pipeline, subtests
    )


def test_static_lpbq_conv3d(builder_pipeline, qnn_compile_spec, pass_pipeline):
    CustomOpQuantizedInputLayouts.test_static_lpbq_conv3d(  # noqa: F405
        builder_pipeline, qnn_compile_spec, pass_pipeline
    )


def test_custom_op_runtime_inputs(builder_pipeline, qnn_compile_spec, subtests):
    CustomOpRuntimeInputs.test(  # noqa: F405
        builder_pipeline, qnn_compile_spec, subtests
    )


def test_runtime_conv_backward_filters(builder_pipeline, subtests):
    RuntimeConvBackwardFilters.test(builder_pipeline, subtests)  # noqa: F405


def test_custom_dequant_op_wrapper(builder_pipeline):
    CustomDequantOpWrapper.test(builder_pipeline)  # noqa: F405


def test_runtime_layout_partitioning(
    builder_pipeline, runtime_layout_compile_spec, subtests
):
    RuntimeLayoutPartitioning.test(  # noqa: F405
        builder_pipeline, runtime_layout_compile_spec, subtests
    )


def test_custom_op_layout_end_to_end(custom_op_compile_spec, qnn_config):
    CustomOpLayoutEndToEnd.test(custom_op_compile_spec, qnn_config)  # noqa: F405


def test_custom_op_mixed_output_layouts(builder_pipeline, subtests):
    CustomOpMixedOutputLayouts.test(builder_pipeline, subtests)  # noqa: F405
