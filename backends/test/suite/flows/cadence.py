# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cadence generic flows for the backend test suite.

Every case is lowered with the Cadence AOT compiler and runs on the x86
cadence_runner, which registers the Cadence generic kernels and the portable
kernels that backends/cadence/aot/functions.yaml selects. Build the runner with
backends/cadence/build_cadence_runner.sh and run pytest from the repository root.
flow.py leaves the flows unregistered when the runner is not built.
"""

from executorch.backends.cadence.aot.pass_utils import CompileMode
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
from executorch.backends.cadence.test.tester import CadenceTester
from executorch.backends.test.suite.flow import TestFlow

# Known failures, as test function names. Each test function runs one
# parametrization per flow; for the models, that is the static-shape one, because
# param_skip_reasons skips the dynamic-shape one first. xfail_patterns is strict,
# so a case that starts passing fails the run until its name is removed.

# Failures in every flow.
_COMMON_FAILURES = [
    # The compiler lowers 3-D convolutions to cadence::conv3d, which has no generic
    # kernel, so the runner fails to load the program.
    "test_conv3d_basic",
    "test_conv3d_channels",
    "test_conv3d_depthwise",
    "test_conv3d_different_spatial_dims",
    "test_conv3d_dilation",
    "test_conv3d_dtype",
    "test_conv3d_groups",
    "test_conv3d_kernel_size",
    "test_conv3d_no_bias",
    "test_conv3d_padding",
    "test_conv3d_padding_circular",
    "test_conv3d_padding_reflect",
    "test_conv3d_padding_replicate",
    "test_conv3d_stride",
    # No 3-D transposed convolution kernel, and the transposed_convolution meta
    # rejects 5-D inputs.
    "test_convtranspose3d_basic",
    "test_convtranspose3d_channels",
    "test_convtranspose3d_depthwise",
    "test_convtranspose3d_different_spatial_dims",
    "test_convtranspose3d_dilation",
    "test_convtranspose3d_dtype",
    "test_convtranspose3d_groups",
    "test_convtranspose3d_kernel_size",
    "test_convtranspose3d_no_bias",
    "test_convtranspose3d_output_padding",
    "test_convtranspose3d_padding",
    "test_convtranspose3d_stride",
    # adaptive_max_pool2d/3d, median and round.decimals are not core ATen and have
    # no portable kernel, so the compiler rejects them.
    "test_adaptive_maxpool1d_batch_sizes",
    "test_adaptive_maxpool1d_dtype",
    "test_adaptive_maxpool1d_input_sizes",
    "test_adaptive_maxpool1d_output_size",
    "test_adaptive_maxpool1d_return_indices",
    "test_adaptive_maxpool2d_batch_sizes",
    "test_adaptive_maxpool2d_dtype",
    "test_adaptive_maxpool2d_input_sizes",
    "test_adaptive_maxpool2d_output_size",
    "test_adaptive_maxpool2d_return_indices",
    "test_adaptive_maxpool3d_batch_sizes",
    "test_adaptive_maxpool3d_dtype",
    "test_adaptive_maxpool3d_input_sizes",
    "test_adaptive_maxpool3d_output_size",
    "test_adaptive_maxpool3d_return_indices",
    "test_median_basic",
    "test_median_dtype",
    "test_median_scalar",
    "test_median_shapes",
    "test_median_dim",
    "test_median_keepdim",
    "test_median_keepdim_with_indices",
    "test_median_with_indices",
    "test_round_decimals",
    # The 3-D pools have no portable kernel, so the runner cannot load them.
    "test_adaptive_avgpool3d_batch_sizes",
    "test_adaptive_avgpool3d_dtype",
    "test_adaptive_avgpool3d_input_sizes",
    "test_adaptive_avgpool3d_output_size",
    "test_avgpool3d_batch_sizes",
    "test_avgpool3d_ceil_mode",
    "test_avgpool3d_combinations",
    "test_avgpool3d_count_include_pad",
    "test_avgpool3d_dtype",
    "test_avgpool3d_input_sizes",
    "test_avgpool3d_kernel_size",
    "test_avgpool3d_padding",
    "test_avgpool3d_stride",
    "test_maxpool3d_batch_sizes",
    "test_maxpool3d_ceil_mode",
    "test_maxpool3d_combinations",
    "test_maxpool3d_dilation",
    "test_maxpool3d_dtype",
    "test_maxpool3d_input_sizes",
    "test_maxpool3d_kernel_size",
    "test_maxpool3d_padding",
    "test_maxpool3d_return_indices",
    "test_maxpool3d_stride",
]

# Failures in both int8 flows.
_INT8_FAILURES = [
    # With random weights, the activations grow to about 1e12, and the int8 error,
    # about 3% of the output, exceeds the suite's tolerance.
    "test_inception_v3",
    # One output quantization step off, above the suite's atol of 0.1.
    "test_resnet50",
    "test_wide_resnet50_2",
]


def _xfail_patterns(names: list[str]) -> list[str]:
    # A test's name continues with "[" and its parameters, so the pattern matches
    # only the test function it names.
    return [f"{name}[" for name in names]


# The int8 LSTM unrolls into a subgraph per time step, which the Cadence passes
# take about a minute to compile for this case's 50-step subtest. With its other
# subtests, the case takes about the suite's 120 s operator timeout.
_INT8_SKIPPED_TESTS = ["test_lstm_seq_lengths"]


def _create_cadence_flow(
    name: str,
    xfail_patterns: list[str],
    quantize: bool = False,
    mode: CompileMode = CompileMode.DEFAULT,
) -> TestFlow:
    def create_tester(module, inputs) -> CadenceTester:
        quantizer = CadenceDefaultQuantizer() if quantize else None
        return CadenceTester(module, inputs, quantizer=quantizer, mode=mode)

    return TestFlow(
        name,
        backend="cadence",
        tester_factory=create_tester,
        quantize=quantize,
        is_delegated=False,
        skip_patterns=_INT8_SKIPPED_TESTS if quantize else [],
        xfail_patterns=xfail_patterns,
        param_skip_reasons={
            "use_dynamic_shapes": {
                True: "The Cadence passes and memory planning need static shapes."
            }
        },
    )


CADENCE_GENERIC_TEST_FLOW = _create_cadence_flow(
    "cadence_generic", _xfail_patterns(_COMMON_FAILURES)
)
CADENCE_GENERIC_INT8_TEST_FLOW = _create_cadence_flow(
    "cadence_generic_int8",
    _xfail_patterns(_COMMON_FAILURES + _INT8_FAILURES),
    quantize=True,
)
CADENCE_GENERIC_INT8_SIZE_TEST_FLOW = _create_cadence_flow(
    "cadence_generic_int8_size",
    _xfail_patterns(_COMMON_FAILURES + _INT8_FAILURES),
    quantize=True,
    mode=CompileMode.SIZE,
)
