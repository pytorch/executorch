# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestUpsampleBilinear2d(unittest.TestCase):
    class StaticResizeBilinear2dModule(torch.nn.Module):
        def forward(self, x):
            a = torch.nn.functional.interpolate(
                x,
                size=(x.shape[2] * 2, x.shape[3] * 3),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            a = torch.nn.functional.interpolate(
                a,
                scale_factor=3.0,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            return a

    class StaticResizeBilinear2dModuleWithAlignCorners(torch.nn.Module):
        def forward(self, x):
            a = torch.nn.functional.interpolate(
                x,
                size=(x.shape[2] * 2, x.shape[3] * 3),
                mode="bilinear",
                align_corners=True,
                antialias=False,
            )
            a = torch.nn.functional.interpolate(
                a,
                scale_factor=3.0,
                mode="bilinear",
                align_corners=True,
                antialias=False,
            )
            return a

    class Bilinear2dAntiAlias(torch.nn.Module):
        def forward(self, x):
            a = torch.nn.functional.interpolate(
                x,
                size=(x.shape[2] * 2, x.shape[3] * 3),
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )
            a = torch.nn.functional.interpolate(
                a,
                scale_factor=3.0,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return a

    # Since we may or may not enable dim order, use these ops only for
    # check_not since we have `to_copy` and `to_dim_order_copy` in the list.
    ops = {
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_index_Tensor",
        "executorch_exir_dialects_edge__ops_aten_arange_start_step",
        "executorch_exir_dialects_edge__ops_aten__to_copy_default",
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_clamp_default",
    }

    def test_fp32_static_resize_bilinear2d(self):
        example_inputs = (torch.randn(2, 3, 4, 5),)
        (
            Tester(self.StaticResizeBilinear2dModule(), example_inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_not(self.ops)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_static_resize_bilinear2d_with_align_corners(self):
        example_inputs = (torch.randn(2, 3, 4, 5),)
        (
            Tester(self.StaticResizeBilinear2dModuleWithAlignCorners(), example_inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_not(self.ops)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_static_resize_bilinear2d_antialiased(self):
        # Check bilinear2d_aa is not partitioned
        example_inputs = (torch.randn(2, 3, 4, 5),)
        (
            Tester(self.Bilinear2dAntiAlias(), example_inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten__upsample_bilinear2d_aa_default": 2
                }
            )
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )
