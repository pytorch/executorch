# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.fuse_concat_pass import FuseConcatPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    PassPipeline,
)


aten_cat_op = "torch.ops.aten.cat.default"
cat_op = "executorch_exir_dialects_edge__ops_aten_cat_default"
slice_op = "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"


class SingleInputCat(torch.nn.Module):
    """Pattern 1: cat with a single input is a no-op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x], dim=0)

    data = (torch.randn(2, 3, 4),)
    ops_before_pass = {cat_op: 1}
    ops_after_pass: dict = {}
    ops_not_after_pass = [cat_op]


class CatThenSlice(torch.nn.Module):
    """Pattern 2: cat followed by slices that extract exactly the inputs."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        combined = torch.cat([a, b], dim=1)
        # Extract exactly a and b back out
        part_a = combined[:, :3, :]
        part_b = combined[:, 3:, :]
        return part_a + 1, part_b + 1

    data = (torch.randn(1, 3, 4), torch.randn(1, 5, 4))
    ops_before_pass = {cat_op: 1, slice_op: 2}
    ops_after_pass: dict = {}
    ops_not_after_pass = [cat_op, slice_op]


class SliceThenCat(torch.nn.Module):
    """Pattern 3: contiguous slices of the same tensor concatenated back."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :3, :]
        b = x[:, 3:, :]
        return torch.cat([a, b], dim=1)

    data = (torch.randn(1, 8, 4),)
    ops_before_pass = {cat_op: 1, slice_op: 2}
    ops_after_pass: dict = {}
    ops_not_after_pass = [cat_op, slice_op]


class CatNotEliminated(torch.nn.Module):
    """Negative test: cat of different tensors should NOT be eliminated."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)

    data = (torch.randn(1, 3, 4), torch.randn(1, 5, 4))
    ops_before_pass = {cat_op: 1}
    ops_after_pass = {cat_op: 1}


class SliceThenCatPartial(torch.nn.Module):
    """Negative test: non-contiguous slices should NOT be eliminated."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :3, :]
        b = x[:, 4:, :]  # Gap at index 3
        return torch.cat([a, b], dim=1)

    data = (torch.randn(1, 8, 4),)
    ops_before_pass = {cat_op: 1}
    ops_after_pass = {cat_op: 1}


class CatThenSliceMismatch(torch.nn.Module):
    """Negative test: slices that don't match original inputs."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([a, b], dim=1)
        return combined[:, 1:5, :]  # Crosses the boundary

    data = (torch.randn(1, 3, 4), torch.randn(1, 5, 4))
    ops_before_pass = {cat_op: 1}
    ops_after_pass = {cat_op: 1}


class CatThenSliceWithStep(torch.nn.Module):
    """Negative test: slices with step != 1 should NOT be eliminated."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        combined = torch.cat([a, b], dim=1)
        part_a = combined[:, :3:2, :]  # step=2, output shape differs from a
        part_b = combined[:, 3::1, :]
        return part_a + 1, part_b + 1

    data = (torch.randn(1, 3, 4), torch.randn(1, 5, 4))
    ops_before_pass = {cat_op: 1}
    ops_after_pass = {cat_op: 1}


class CatThenSubSlice(torch.nn.Module):
    """Pattern 4: slice extracts a sub-range within one concat input."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([a, b], dim=1)  # a dim1=6, b dim1=4
        # Range [1,5) falls entirely within a's range [0,6)
        return combined[:, 1:5, :] + 1

    data = (torch.randn(1, 6, 4), torch.randn(1, 4, 4))
    ops_before_pass = {cat_op: 1, slice_op: 1}
    ops_after_pass = {slice_op: 1}
    ops_not_after_pass = [cat_op]


class CatThenSubSliceSecondInput(torch.nn.Module):
    """Pattern 4: sub-slice within second concat input (tests offset adjust)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([a, b], dim=1)  # a dim1=3, b dim1=8
        # Range [5,9) falls within b's range [3,11), adjusted to [2,6) on b
        return combined[:, 5:9, :] + 1

    data = (torch.randn(1, 3, 4), torch.randn(1, 8, 4))
    ops_before_pass = {cat_op: 1, slice_op: 1}
    ops_after_pass = {slice_op: 1}
    ops_not_after_pass = [cat_op]


class SliceThenCatPartialContiguous(torch.nn.Module):
    """Pattern 5: contiguous slices covering a sub-range of the dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, 2:5, :]
        b = x[:, 5:8, :]
        return torch.cat([a, b], dim=1)  # Equivalent to x[:, 2:8, :]

    data = (torch.randn(1, 10, 4),)
    ops_before_pass = {cat_op: 1, slice_op: 2}
    ops_after_pass = {slice_op: 1}
    ops_not_after_pass = [cat_op]


positive_tests = {
    "single_input_cat": SingleInputCat(),
    "cat_then_slice": CatThenSlice(),
    "slice_then_cat": SliceThenCat(),
    "cat_then_sub_slice": CatThenSubSlice(),
    "cat_then_sub_slice_second_input": CatThenSubSliceSecondInput(),
    "slice_then_cat_partial_contiguous": SliceThenCatPartialContiguous(),
}

negative_tests = {
    "cat_not_eliminated": CatNotEliminated(),
    "slice_then_cat_partial": SliceThenCatPartial(),
    "cat_then_slice_mismatch": CatThenSliceMismatch(),
    "cat_then_slice_with_step": CatThenSliceWithStep(),
}


@common.parametrize("model", positive_tests)
def test_fuse_concat_eliminates(model):
    pipeline = PassPipeline(
        model,
        model.data,
        quantize=False,
        ops_before_pass=model.ops_before_pass,
        ops_after_pass=model.ops_after_pass,
        ops_not_after_pass=getattr(model, "ops_not_after_pass", []),
        pass_list=[FuseConcatPass],
    )
    pipeline.run()


@common.parametrize("model", negative_tests)
def test_fuse_concat_preserves(model):
    pipeline = PassPipeline(
        model,
        model.data,
        quantize=False,
        ops_before_pass=model.ops_before_pass,
        ops_after_pass=model.ops_after_pass,
        pass_list=[FuseConcatPass],
    )
    pipeline.run()


# End-to-end Ethos-U pipeline test models. The PassPipeline modules above wire
# the pattern directly to the model's forward args, which is fine for unit-level
# pass validation but causes I/O signature drift through Vela whenever the pass
# eliminates the only use of an input placeholder. These wrappers feed each
# pattern through a single input and surrounding compute so that lowering sees
# realistic graphs.


class SingleInputCatEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x + 1], dim=0) + 2

    data = (torch.randn(2, 3, 4),)


class CatThenSliceEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        a = x[:, :3, :] + 0.5
        b = x[:, 3:, :] + 0.5
        combined = torch.cat([a, b], dim=1)
        return combined[:, :3, :] + 1, combined[:, 3:, :] + 1

    data = (torch.randn(1, 8, 4),)


class SliceThenCatEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        a = y[:, :3, :]
        b = y[:, 3:, :]
        return torch.cat([a, b], dim=1) + 2

    data = (torch.randn(1, 8, 4),)


class CatThenSubSliceEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :6, :] + 0.5
        b = x[:, 6:, :] + 0.5
        combined = torch.cat([a, b], dim=1)
        return combined[:, 1:5, :] + 1

    data = (torch.randn(1, 10, 4),)


class CatThenSubSliceSecondInputEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :3, :] + 0.5
        b = x[:, 3:, :] + 0.5
        combined = torch.cat([a, b], dim=1)
        return combined[:, 5:9, :] + 1

    data = (torch.randn(1, 11, 4),)


class SliceThenCatPartialContiguousEthos(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, 2:5, :]
        b = x[:, 5:8, :]
        return torch.cat([a, b], dim=1) + 1

    data = (torch.randn(1, 10, 4),)


ethos_tests = {
    "single_input_cat": SingleInputCatEthos(),
    "cat_then_slice": CatThenSliceEthos(),
    "slice_then_cat": SliceThenCatEthos(),
    "cat_then_sub_slice": CatThenSubSliceEthos(),
    "cat_then_sub_slice_second_input": CatThenSubSliceSecondInputEthos(),
    "slice_then_cat_partial_contiguous": SliceThenCatPartialContiguousEthos(),
}


@common.parametrize("model", ethos_tests)
@common.XfailIfNoCorstone300
def test_fuse_concat_u55_INT(model):
    pipeline = EthosU55PipelineINT(
        model,
        model.data,
        aten_cat_op,
        cat_op,
    )
    pipeline.run()


@common.parametrize("model", ethos_tests)
@common.XfailIfNoCorstone320
def test_fuse_concat_u85_INT(model):
    pipeline = EthosU85PipelineINT(
        model,
        model.data,
        aten_cat_op,
        cat_op,
    )
    pipeline.run()
