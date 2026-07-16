# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.flip.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_flip_default"

input_t1 = Tuple[torch.Tensor]  # Input x

# (input tensor, dims) — single-axis, negative-axis, and multi-axis cases.
test_data_suite = {
    "rank1": lambda: (torch.rand(10), [0]),
    "rank2_dim0": lambda: (torch.rand(4, 5), [0]),
    "rank2_dim1_neg": lambda: (torch.rand(4, 5), [-1]),
    "rank3_multi": lambda: (torch.rand(2, 3, 4), [0, 2]),
    "rank4_dim2": lambda: (torch.rand(2, 3, 4, 5), [2]),
    "rank4_multi": lambda: (torch.rand(2, 3, 4, 5), [1, 3]),
    "rank4_all_neg": lambda: (torch.rand(2, 3, 4, 5), [-4, -3, -2, -1]),
}

test_data_suite_bf16 = {
    "rank4_multi_bf16": lambda: (
        torch.rand(2, 3, 4, 5, dtype=torch.bfloat16),
        [1, 3],
    ),
}

# The fp8 cases compare the lowered output against eager torch.flip. Eager
# torch.flip on a float8 CPU tensor raises NotImplementedError ("flip_cpu not
# implemented for Float8_*") when the flipped dims include the last axis, so
# these cases flip outer axes only to keep that comparison runnable.
test_data_suite_fp8 = {
    "rank4_multi_fp8e4m3": lambda: (
        torch.rand(2, 3, 4, 5).to(torch.float8_e4m3fn),
        [0, 2],
        "fp8e4m3",
    ),
    "rank4_multi_fp8e5m2": lambda: (
        torch.rand(2, 3, 4, 5).to(torch.float8_e5m2),
        [0, 2],
        "fp8e5m2",
    ),
}

# U55 has no REVERSE, so flip must not be delegated there.
test_data_suite_u55_reject = {
    "rank4_dim2": lambda: (torch.rand(2, 3, 4, 5), [2]),
}

# flip over no dims is the identity;
# the partitioner must prune the trivial partition instead of delegating it.
test_data_suite_empty = {
    "empty_dims": lambda: (torch.rand(4, 5), []),
}


class Flip(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return torch.flip(x, self.dims)


class FlipBool(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.flip(x > 0, [1])


@common.parametrize("test_data", test_data_suite)
def test_flip_tosa_FP(test_data):
    data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](Flip(dims), (data,), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_flip_tosa_FP_reverse_count(test_data):
    """Each flip dim must lower to exactly one TOSA REVERSE."""
    data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](Flip(dims), (data,), aten_op, exir_op)
    pipeline.count_tosa_ops({"REVERSE": len(dims)})
    pipeline.run()


@common.parametrize("test_data", test_data_suite_bf16)
def test_flip_tosa_FP_bf16(test_data):
    data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Flip(dims), (data,), aten_op, exir_op, tosa_extensions=["bf16"]
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_fp8)
def test_flip_tosa_FP_fp8(test_data):
    data, dims, tosa_extension = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Flip(dims),
        (data,),
        aten_op,
        exir_op,
        tosa_extensions=[tosa_extension],
    )
    pipeline.count_tosa_ops({"REVERSE": len(dims)})
    pipeline.run()


def test_flip_bool_tosa_FP():
    """Flip on a bool mask (REVERSE supports bool)."""
    pipeline = TosaPipelineFP[input_t1](
        FlipBool(), (torch.randn(2, 4),), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_flip_tosa_INT(test_data):
    data, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](Flip(dims), (data,), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_flip_16a8w_tosa_INT(test_data):
    """16A8W quantization (16-bit activations, 8-bit weights)."""
    data, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](
        Flip(dims),
        (data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=False)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_u55_reject)
def test_flip_u55_INT_not_delegated(test_data):
    data, dims = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        Flip(dims),
        (data,),
        non_delegated_ops={exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_empty)
def test_flip_empty_dims_not_delegated(test_data):
    """Flip over no dims is the identity."""
    data, dims = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        Flip(dims),
        (data,),
        non_delegated_ops={exir_op: 1},
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_flip_u85_INT(test_data):
    data, dims = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        Flip(dims),
        (data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_flip_vgf_no_quant(test_data):
    data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        Flip(dims),
        (data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_flip_vgf_quant(test_data):
    data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        Flip(dims),
        (data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
