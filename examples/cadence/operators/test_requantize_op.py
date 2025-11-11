# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Example script for exporting simple models to flatbuffer

import logging
import unittest

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from executorch.backends.cadence.aot.ref_implementations import *  # noqa

import itertools

from executorch.backends.cadence.aot.export_example import export_and_run_model
from parameterized import parameterized


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def create_tensor_with_dtype(
    shape: tuple[int], dtype: torch.dtype = torch.float32, _max: float = 1
) -> torch.Tensor:
    """
    Create a tensor with the given shape and dtype. '_max' indicates the maximum
    value in the tensor.
    """
    new_tensor: torch.Tensor = torch.rand(shape) * _max
    return new_tensor.to(dtype=dtype)


class CadenceRequantizeOpCases(unittest.TestCase):
    @parameterized.expand(
        # Check cross-product of in and out dtypes.
        [
            [(5, 2), 0.01, 0, 0.02, 1, in_dtype, out_dtype]
            for in_dtype, out_dtype in itertools.product(
                [torch.int8, torch.uint8, torch.int16, torch.uint16],
                repeat=2,
            )
        ]
    )
    def test_cadence_requantize_out(
        self,
        shape: tuple[int],
        in_scale: float,
        in_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ) -> None:
        class QuantModel(torch.nn.Module):
            def __init__(
                self,
                in_scale: float,
                in_zero_point: int,
                out_scale: float,
                out_zero_point: float,
                dtype: torch.dtype,
            ) -> None:
                super().__init__()
                self.in_scale = torch.tensor(in_scale)
                self.in_zero_point = torch.tensor(in_zero_point, dtype=torch.int32)
                self.out_scale = torch.tensor(out_scale)
                self.out_zero_point = torch.tensor(out_zero_point, dtype=torch.int32)
                self.dtype = dtype

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ops.cadence.requantize.default(
                    x,
                    self.in_scale,
                    self.in_zero_point,
                    self.out_scale,
                    self.out_zero_point,
                    self.dtype,
                )

        model = QuantModel(
            in_scale, in_zero_point, out_scale, out_zero_point, out_dtype
        )
        dtype_info = torch.iinfo(in_dtype)
        inputs = (
            create_tensor_with_dtype(shape, in_dtype, _max=float(dtype_info.max)),
        )

        # Run and verify correctness
        # Since this test is handling integers, its inputs and outputs might have
        # a larger MSE loss, and that's alright.
        # For example, if the ref output is [33, 50] and the real output is [33, 49],
        # the MSE loss is around 0.5, but the relative error is < 2%. So we set
        # the epsilon to a higher value.
        export_and_run_model(model, inputs, eps_error=1.0)


if __name__ == "__main__":
    unittest.main()
