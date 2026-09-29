# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.passes.decompose_mean_pass import DecomposeMeanPass
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


# DecomposeMeanPass rewrites the mean before annotation, so the edge graph
# already holds an average pool; what these pin down is that it then quantizes
# like any other pool, that no zero padding is inserted around it, and that the
# reduced dimensions are dropped by a view exactly when keepdim is false.
class CortexMSpatialMean(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mean_dim": 0,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_pad_default": 0,
        "executorch_exir_dialects_edge__ops_aten_mean_dim": 0,
    }

    def forward(self, x):
        return x.mean([2, 3])


class CortexMSpatialMeanKeepdim(CortexMSpatialMean):
    ops_before_transforms = {
        **CortexMSpatialMean.ops_before_transforms,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 0,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }
    ops_after_transforms = {
        **CortexMSpatialMean.ops_after_transforms,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 0,
    }

    def forward(self, x):
        return x.mean([2, 3], keepdim=True)


class CortexMNegativeDimMean(CortexMSpatialMean):
    def forward(self, x):
        return x.mean([-2, -1])


# The declining cases share one pair of counts, which pins only the decline
# itself: their boundary quant/dequant counts differ from each other.
fallback_ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
}
fallback_ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 0,
}


class CortexMChannelMean(torch.nn.Module):
    def forward(self, x):
        return x.mean([1])


class CortexMAllButBatchMean(torch.nn.Module):
    def forward(self, x):
        return x.mean([1, 2, 3])


class CortexMOneSpatialDimMean(torch.nn.Module):
    def forward(self, x):
        return x.mean([2])


class CortexMDtypeMean(torch.nn.Module):
    def forward(self, x):
        return x.mean([2, 3], dtype=torch.float32)


class CortexMConstantMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("stats", ramp_tensor(-1, 1, (4, 8)))

    def forward(self, x):
        return x + self.stats.mean([0])


def nhwc(shape):
    """The pool kernel requires channels_last, and nothing before the runtime
    rejects a contiguous tensor, so the layout has to arrive with the program
    input -- which is how the model tests and the suite flow drive it."""
    return ramp_tensor(-5, 5, shape).to(memory_format=torch.channels_last)


test_cases = {
    "spatial": McuTestCase(
        model=CortexMSpatialMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    "spatial_keepdim": McuTestCase(
        model=CortexMSpatialMeanKeepdim(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    "spatial_negative_dims": McuTestCase(
        model=CortexMNegativeDimMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    "spatial_non_square": McuTestCase(
        model=CortexMSpatialMean(),
        example_inputs=(nhwc((1, 8, 3, 5)),),
    ),
    # The only case that reaches the kernel's batch loop, and the only one
    # whose view moves more than one row.
    "spatial_batched": McuTestCase(
        model=CortexMSpatialMean(),
        example_inputs=(nhwc((3, 8, 4, 4)),),
    ),
}

# Reductions with no average-pool equivalent. They stay in fp32, which the test
# runner carries no kernel for, so these only go as far as the dialect.
fallback_test_cases = {
    "channel": McuTestCase(
        model=CortexMChannelMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    "all_but_batch": McuTestCase(
        model=CortexMAllButBatchMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    "one_spatial_dim": McuTestCase(
        model=CortexMOneSpatialDimMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    # The trailing dims of a 3-D tensor normalize to the same [2, 3] a spatial
    # reduction does, and pooling them would be wrong.
    "rank_3_trailing_dims": McuTestCase(
        model=CortexMNegativeDimMean(),
        example_inputs=(ramp_tensor(-5, 5, (2, 8, 4)),),
    ),
    # dtype= changes the accumulation type, which avg_pool2d has no argument
    # for.
    "accumulate_dtype": McuTestCase(
        model=CortexMDtypeMean(),
        example_inputs=(nhwc((1, 8, 4, 4)),),
    ),
    # A mean over a buffer reaches the pass as a bare tensor rather than a
    # proxy, so the guard for it has to come before anything reads a shape.
    "constant_operand": McuTestCase(
        model=CortexMConstantMean(),
        example_inputs=(ramp_tensor(-5, 5, (2, 8)),),
    ),
}


def test_dynamic_batch_is_refused():
    """A symbolic dimension has no literal to put in the kernel size or the
    view, and the suite's flow does not exercise dynamic shapes."""
    exported = torch.export.export(
        CortexMSpatialMean(),
        (torch.randn(2, 8, 4, 4),),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},),
    )
    rewritten = DecomposeMeanPass()(exported.module()).graph_module
    assert any(node.target is torch.ops.aten.mean.dim for node in rewritten.graph.nodes)


@parametrize("test_case", test_cases)
def test_lowered_shape_matches_eager(test_case, cortex_m_target):
    """The harness takes its reference from a program exported after the pass
    has run, so it compares the rewrite against itself. The reduced shape is
    the one thing that comparison cannot see."""
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize().export().to_edge().run_passes()
    graph = tester.get_artifact().exported_program().graph
    (output,) = graph.output_node().args[0]
    assert (
        output.meta["val"].shape == test_case.model(*test_case.example_inputs).shape
    ), output.meta["val"].shape


@parametrize("test_case", test_cases)
def test_dialect_mean(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_mean(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)


@parametrize("test_case", fallback_test_cases)
def test_dialect_mean_fallback(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        fallback_ops_before_transforms, fallback_ops_after_transforms, qtol=1
    )
