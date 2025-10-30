# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import IntEnum
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)


class IndexTensorTestCommon:
    """Class containing constants common between the tests"""

    aten_op = "torch.ops.aten.index.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_Tensor"

    # Gathers and reshapes should result in no inaccuracies
    rtol = 0.0
    atol = 0.0

    class OpPlacement(IntEnum):
        """
        Simple enum used to indicate where slices or ellipsis should be placed
        in tests.
        IntEnum so that Dynamo does not complain about unsupported types.
        """

        BEFORE = 1
        MIDDLE = 2
        AFTER = 3


input_params_slice = Tuple[
    torch.Tensor, int, int, IndexTensorTestCommon.OpPlacement, Tuple[torch.Tensor]
]
input_params = Tuple[torch.Tensor, Tuple[torch.Tensor]]


class IndexTensor_Ellipsis(torch.nn.Module):
    """
    There are technical limitations with torch/export as it does not support
    the ellipsis class and as such the forward function has been crafted
    to circumvent that limitation.
    """

    # xfail - ellipsis unsupported
    test_data_ellipsis: dict[input_params] = {
        "test_4d_ellipsis_before": (
            torch.rand(size=(25, 5, 13, 7)),
            IndexTensorTestCommon.OpPlacement.BEFORE,
            (torch.arange(2, dtype=torch.int32),),
        ),
        "test_4d_ellipsis_middle": (
            torch.rand(size=(25, 5, 13, 7)),
            IndexTensorTestCommon.OpPlacement.MIDDLE,
            (
                torch.arange(2, dtype=torch.int32),
                torch.arange(2, dtype=torch.int32),
            ),
        ),
        "test_4d_ellipsis_after": (
            # Due to the information passed to the NodeVisitor and
            # preceding passes, detecting this and rejecting it for
            # partitioning is difficult and unreliable, as such
            # it is not xfail as the existing logic can handle it.
            torch.rand(size=(25, 5, 13, 7)),
            IndexTensorTestCommon.OpPlacement.AFTER,
            (torch.arange(2, dtype=torch.int32),),
        ),
    }

    def forward(
        self,
        input_: torch.Tensor,
        position: IndexTensorTestCommon.OpPlacement,
        indices: Tuple[None | torch.Tensor],
    ):
        match position:
            case IndexTensorTestCommon.OpPlacement.BEFORE:
                return input_[..., indices[0]]
            case IndexTensorTestCommon.OpPlacement.MIDDLE:
                return input_[indices[0], ..., indices[1]]
            case IndexTensorTestCommon.OpPlacement.AFTER:
                return input_[indices[0], ...]

        return input_[indices]


@common.parametrize(
    "test_data",
    IndexTensor_Ellipsis.test_data_ellipsis,
    xfails={
        # More info in index_tensor_support.py
        "test_4d_ellipsis_before": "Ellipsis before index unsupported",
        "test_4d_ellipsis_middle": "Ellipsis before index unsupported",
    },
)
def test_index_tensor_tosa_FP_ellipsis(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineFP[input_params](
                IndexTensor_Ellipsis(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
                atol=IndexTensorTestCommon.atol,
                rtol=IndexTensorTestCommon.rtol,
            ).run()
        )


@common.parametrize(
    "test_data",
    IndexTensor_Ellipsis.test_data_ellipsis,
    xfails={
        # More info in index_tensor_support.py
        "test_4d_ellipsis_before": "Ellipsis before index unsupported",
        "test_4d_ellipsis_middle": "Ellipsis before index unsupported",
    },
)
def test_index_tensor_tosa_INT_ellipsis(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineINT[input_params](
                IndexTensor_Ellipsis(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
            ).run()
        )


class IndexTensor_Slice(torch.nn.Module):
    """
    There are technical limitations with Dynamo as it does not support the
    slice class and as such the forward function has been crafted
    to circumvent that limitation.
    """

    # xfail - None unsupported
    test_data: dict[input_params_slice] = {
        "test_4d_slice_before_1d_idx": (
            # Value tens is 3D because with the
            torch.rand(size=(5, 3, 4, 5)),
            0,
            2,
            IndexTensorTestCommon.OpPlacement.BEFORE,
            (torch.arange(2, dtype=torch.int32),),
        ),
        "test_3d_slice_before_2d_idx": (
            # TODO: MLETORCH-859 - Testing framework does not support output rank > 4
            # With the bellow configuration a 4D value tensor and 2D index tensor
            # results in a 5D output.
            torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4),
            0,
            2,
            IndexTensorTestCommon.OpPlacement.BEFORE,
            (torch.arange(2, dtype=torch.int32).unsqueeze(0).tile(2, 1),),
        ),
        "test_4d_slice_middle": (
            torch.arange(5 * 3 * 2, dtype=torch.int32).reshape(5, 3, 2),
            0,
            2,
            IndexTensorTestCommon.OpPlacement.MIDDLE,
            (
                torch.arange(2, dtype=torch.int32),
                torch.arange(2, dtype=torch.int32),
            ),
        ),
        "test_4d_slice_after": (
            # Due to the information passed to the NodeVisitor and
            # preceding passes, detecting this and rejecting it for
            # partitioning is difficult and unreliable, as such
            # it is not xfail as the existing logic can handle it.
            torch.rand(size=(25, 5, 13, 7)),
            0,
            2,
            IndexTensorTestCommon.OpPlacement.AFTER,
            (torch.arange(2, dtype=torch.int32),),
        ),
    }

    def forward(
        self,
        input_: torch.Tensor,
        slice_start: int,
        slice_end: int,
        position: IndexTensorTestCommon.OpPlacement,
        indices: Tuple[None | torch.Tensor],
    ):
        match position:
            case IndexTensorTestCommon.OpPlacement.BEFORE:
                return input_[slice_start:slice_end, indices[0]]
            case IndexTensorTestCommon.OpPlacement.MIDDLE:
                return input_[indices[0], slice_start:slice_end, indices[1]]
            case IndexTensorTestCommon.OpPlacement.AFTER:
                return input_[indices[0], slice_start:slice_end]


@common.parametrize(
    "test_data",
    IndexTensor_Slice.test_data,
    xfails={
        # More info in index_tensor_support.py
        "test_4d_slice_before_1d_idx": "Slice before index unsupported",
        "test_3d_slice_before_2d_idx": "Slice before index unsupported",
        "test_4d_slice_middle": "Slice before index unsupported",
    },
)
def test_index_tensor_tosa_FP_slice(test_data: input_params_slice):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineFP[input_params_slice](
                IndexTensor_Slice(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
                atol=IndexTensorTestCommon.atol,
                rtol=IndexTensorTestCommon.rtol,
            ).run()
        )


@common.parametrize(
    "test_data",
    IndexTensor_Slice.test_data,
    xfails={
        # More info in index_tensor_support.py
        "test_4d_slice_before_1d_idx": "Slice before index unsupported",
        "test_3d_slice_before_2d_idx": "Slice before index unsupported",
        "test_4d_slice_middle": "Slice before index unsupported",
    },
)
def test_index_tensor_tosa_INT_slice(test_data: input_params_slice):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineINT[input_params_slice](
                IndexTensor_Slice(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
            ).run()
        )


class IndexTensor(torch.nn.Module):
    test_data: dict[input_params] = {
        "test_2d_1_idx": (torch.rand(5, 2), (torch.arange(5, dtype=torch.int32),)),
        "test_2d_1_less_than_max_idx": (
            torch.rand(5, 2),
            (torch.arange(3, dtype=torch.int32),),
        ),
        "test_2d_1_2d_idx": (
            torch.rand(5, 2),
            (torch.randint(5, size=(4, 3), dtype=torch.int32)),
        ),
        "test_2d_2_idx": (
            torch.rand(5, 2),
            (
                torch.randint(5, size=(5,), dtype=torch.int32),
                torch.randint(2, size=(5,), dtype=torch.int32),
            ),
        ),
        "test_2d_2_2d_idx_broadcastable": (
            torch.rand(5, 2),
            (
                torch.randint(5, size=(5, 3), dtype=torch.int32),
                torch.randint(2, size=(1, 3), dtype=torch.int32),
            ),
        ),
        "test_2d_2_2d_idx_broadcastable_2": (
            torch.rand(5, 2),
            (
                torch.randint(5, size=(5, 1), dtype=torch.int32),
                torch.randint(2, size=(3,), dtype=torch.int32),
            ),
        ),
        "test_3d_1_idx": (torch.rand(12, 3, 7), (torch.arange(12, dtype=torch.int32),)),
        "test_3d_2_idx": (
            torch.rand(12, 3, 7),
            (
                torch.arange(12, dtype=torch.int32),
                torch.randint(3, size=(12,), dtype=torch.int32),
            ),
        ),
        "test_3d_3_idx": (
            torch.rand(12, 3, 7),
            (
                torch.arange(12, dtype=torch.int32),
                torch.randint(3, size=(12,), dtype=torch.int32),
                torch.randint(7, size=(12,), dtype=torch.int32),
            ),
        ),
        "test_4d_1_idx": (
            torch.rand(15, 3, 7, 2),
            (torch.arange(15, dtype=torch.int32),),
        ),
        "test_4d_2_idx": (
            torch.rand(15, 3, 7, 2),
            (
                torch.randint(15, size=(15,), dtype=torch.int32),
                torch.randint(3, size=(1,), dtype=torch.int32),
            ),
        ),
        "test_4d_3_idx": (
            torch.rand(15, 3, 7, 2),
            (
                torch.arange(15, dtype=torch.int32),
                torch.randint(3, size=(15,), dtype=torch.int32),
                torch.randint(7, size=(15,), dtype=torch.int32),
            ),
        ),
        "test_4d_4_id_broadcastable": (
            torch.rand(15, 3, 7, 2),
            (
                torch.arange(15, dtype=torch.int32),
                torch.randint(3, size=(3, 1), dtype=torch.int32),
                torch.randint(6, size=(6, 1, 1), dtype=torch.int32),
                torch.randint(2, size=(15,), dtype=torch.int32),
            ),
        ),
    }

    # xfail - None (unsqueeze) unsupported
    test_data_none: dict[input_params] = {
        "test_3d_3_idx_with_none_before": (
            torch.rand(12, 3, 7),
            (
                None,
                torch.randint(3, size=(12,), dtype=torch.int32),
            ),
        ),
        "test_3d_3_idx_with_2_none_before": (
            torch.rand(12, 3, 7),
            (
                None,
                None,
                torch.randint(3, size=(12,), dtype=torch.int32),
            ),
        ),
        "test_3d_3_idx_with_none_around": (
            torch.rand(12, 3, 7),
            (
                None,
                torch.randint(3, size=(12,), dtype=torch.int32),
                None,
            ),
        ),
        "test_3d_3_idx_with_none_after": (
            # Due to the information passed to the NodeVisitor and
            # preceding passes, detecting this and rejecting it for
            # partitioning is difficult and unreliable, as such
            # it is not xfail as the existing logic can handle it.
            torch.rand(12, 3, 7),
            (
                torch.randint(3, size=(12,), dtype=torch.int32),
                None,
            ),
        ),
        "test_3d_3_idx_with_none_middle": (
            torch.rand(12, 3, 7),
            (
                torch.randint(3, size=(12,), dtype=torch.int32),
                None,
                torch.randint(3, size=(12,), dtype=torch.int32),
            ),
        ),
    }

    def forward(self, input_: torch.Tensor, indices: Tuple[None | torch.Tensor]):
        return input_[indices]


@common.parametrize("test_data", IndexTensor.test_data)
def test_index_tensor_tosa_FP(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineFP[input_params](
                IndexTensor(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
                atol=IndexTensorTestCommon.atol,
                rtol=IndexTensorTestCommon.rtol,
            ).run()
        )


@common.parametrize("test_data", IndexTensor.test_data)
def test_index_tensor_tosa_INT(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineINT[input_params](
                IndexTensor(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
            ).run()
        )


@common.parametrize(
    "test_data",
    IndexTensor.test_data_none,
    xfails={
        # More info in index_tensor_support.py
        "test_3d_3_idx_with_none_before": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_2_none_before": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_none_around": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_none_middle": "None (Unsqueeze) unsupported",
    },
)
def test_index_tensor_tosa_FP_none(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineFP[input_params](
                IndexTensor(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
                atol=IndexTensorTestCommon.atol,
                rtol=IndexTensorTestCommon.rtol,
            ).run()
        )


@common.parametrize(
    "test_data",
    IndexTensor.test_data_none,
    xfails={
        # More info in index_tensor_support.py
        "test_3d_3_idx_with_none_before": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_2_none_before": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_none_around": "None (Unsqueeze) unsupported",
        "test_3d_3_idx_with_none_middle": "None (Unsqueeze) unsupported",
    },
)
def test_index_tensor_tosa_INT_none(test_data: input_params):
    test_input = test_data
    with torch.no_grad():
        (
            TosaPipelineINT[input_params](
                IndexTensor(),
                test_input,
                IndexTensorTestCommon.aten_op,
                IndexTensorTestCommon.exir_op,
            ).run()
        )


@common.parametrize("test_data", IndexTensor.test_data)
@common.XfailIfNoCorstone300
def test_index_tensor_u55_INT_not_delegated(test_data: input_params):
    """Ethos-U55 backend BI pipeline test for index.Tensor"""
    test_input = test_data
    with torch.no_grad():
        OpNotSupportedPipeline[input_params](
            IndexTensor(),
            test_input,
            {IndexTensorTestCommon.exir_op: 1},
            quantize=True,
            u55_subset=True,
        ).run()
