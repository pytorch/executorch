# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import namedtuple

from executorch.backends.vulkan.test.op_tests.utils.codegen import VkTestSuite


# Prime numbers dim sizes for testing
XL = 113
L = 89
M2 = 41
M1 = 37
M = 29
S2 = 11
S1 = 7
S = 5
XS = 3


def get_binary_elementwise_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, M2), (M1, M2)),
            ((M1, M2), (M1, 1), 2.0),
            ((M1, M2), (1, M2)),
            ((S, S1, S2), (S, S1, S2)),
            ((S, S1, S2), (S, S1, 1), 2.0),
            ((S, S1, S2), (S, 1, S2), 2.0),
        ]
    )
    test_suite.layouts = [
        "api::kWidthPacked",
        "api::kChannelsPacked",
    ]
    return test_suite


def get_mm_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, L), (L, M2)),
            ((S1, S2), (S2, M)),
        ],
    )
    test_suite.prepacked_args = ["mat2"]
    # ATen matmul doesn't support half
    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = [
        "api::kWidthPacked",
        "api::kChannelsPacked",
    ]
    return test_suite


def get_pool2d_inputs():
    test_suite = VkTestSuite(
        [
            ((S, M1, M2), [2, 2], [1, 1], [0, 0], [1, 1]),
        ]
    )
    return test_suite


def get_conv_inputs():
    test_suite = VkTestSuite(
        [
            (
                (1, 6, 40, 50),
                (8, 6, 3, 3),
                (8,),
                [1, 2],
                [2, 3],
                [1, 1],
                False,
                [0, 0],
                1,
            ),
            (
                (1, 6, 40, 50),
                (6, 8, 3, 3),
                (8,),
                [1, 2],
                [2, 3],
                [1, 1],
                True,
                [0, 1],
                1,
            ),
            (
                (1, 8, 72, 96),
                (8, 1, 3, 3),
                (8,),
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            (
                (1, 8, 72, 96),
                (8, 8, 1, 1),
                (8,),
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            ),
            (
                (1, 6, 40, 50),
                (8, 6, 3, 3),
                None,
                [1, 2],
                [2, 3],
                [1, 1],
                False,
                [0, 0],
                1,
            ),
            (
                (1, 6, 7),
                (6, 1, 3),
                (6,),
                [1],
                [0],
                [1],
                False,
                [0],
                6,
            ),
            (
                (2, 20, 30),
                (10, 4, 6),
                (10,),
                [5],
                [5],
                [3],
                False,
                [0],
                5,
            ),
            (
                (1, 9, 11),
                (9, 1, 3),
                None,
                [1],
                [0],
                [1],
                False,
                [0],
                9,
            ),
            (
                (5, 15, 30),
                (20, 3, 3),
                None,
                [3],
                [5],
                [7],
                False,
                [0],
                5,
            ),
        ]
    )
    return test_suite


def get_native_layer_norm_inputs():
    test_suite = VkTestSuite(
        [
            ((S1, S2), [S2], (S2), (S2), 0.001),
            ((M, M1, M2), [M2], (M2), (M2), 0.001),
            ((S, XL, M1, M2), [M2], (M2), (M2), 0.001),
        ]
    )
    return test_suite


def get_full_inputs():
    test_suite = VkTestSuite(
        [
            ([S1, S2], 42.0),
            ([M, M1, M2], 3.14),
            ([L, M, M1, M2], 2.72),
        ]
    )
    return test_suite


def get_select_int_inputs():
    test_suite = VkTestSuite(
        [
            ((6, 2, 7), 0, 3),
            ((6, 2, 7), 1, 0),
            ((6, 2, 7), 2, 3),
            ((6, 10, 7), 0, 3),
            ((6, 10, 7), 1, 0),
            ((6, 10, 7), 1, 9),
            ((6, 10, 7), 2, 6),
            ((9, 2, 9, 4), 0, 8),
            ((9, 2, 9, 4), 1, 1),
            ((9, 2, 9, 4), 2, 0),
            ((9, 2, 9, 4), 2, 8),
            ((9, 2, 9, 4), 3, 3),
            ((8, 6, 1, 1), 0, 4),
            ((8, 6, 1, 1), 1, 4),
        ]
    )
    return test_suite


def get_permute_inputs():
    test_suite = VkTestSuite(
        [
            ((9, 2, 9, 4), [0, 1, 2, 3]),
            ((9, 2, 9, 4), [0, 1, 3, 2]),
            ((9, 2, 9, 4), [0, 2, 1, 3]),
            ((9, 2, 9, 4), [0, 2, 3, 1]),
            ((9, 2, 9, 4), [0, 3, 1, 2]),
            ((9, 2, 9, 4), [0, 3, 2, 1]),
            ((9, 2, 9, 4), [3, 0, 1, 2]),
            ((9, 2, 9, 4), [3, 2, 0, 1]),
            ((9, 2, 9, 4), [2, 3, 0, 1]),
            ((9, 2, 9, 4), [2, 0, 3, 1]),
            ((9, 2, 9), [2, 0, 1]),
            ((9, 2, 9), [1, 2, 0]),
            ((9, 2), [0, 1]),
            ((9, 2), [1, 0]),
        ]
    )

    test_suite.layouts = ["api::kChannelsPacked"]
    return test_suite


def get_view_inputs():
    test_suite = VkTestSuite(
        [
            ((3, 4, 5), [1, 1, -1]),
            ((3, 4, 5), [1, -1, 1]),
            ((3, 4, 5), [-1, 1, 1]),
            ((8, 7, 2, 3), [4, 3, 7, 4]),
            ((8, 7, 2, 3), [7, -1, 2, 1]),
            ((8, 7, 2, 3), [1, 1, 1, -1]),
            ((8, 7, 2, 3), [-1]),
            ((2, 3, 3, 7), [2, -1, 1, 1]),
            ((3, 5, 2, 7), [7, -1, 2, 1]),
            ((2, 2, 8, 6), [2, 6, -1, 1]),
            ((2, 2, 8, 6), [6, -1, 1]),
            ((S1, S2, S1, S2), [S2, -1, 1, S1]),
            ((S1, S2, S1, S2), [S1, 1, -1, S2]),
            ((S1, S2, S1, S2), [-1, 1, S1, S2]),
        ]
    )
    test_suite.layouts = [
        "api::kWidthPacked",
        "api::kHeightPacked",
        "api::kChannelsPacked",
    ]
    return test_suite


def get_slice_inputs():
    Test = namedtuple("VkSliceTest", ["self", "dim", "start", "end", "step"])
    Test.__new__.__defaults__ = (None, 0, None, None, 1)

    # Slice by width and height
    test_cases = [
        Test(self=[1, 1, 4, 10], dim=3, start=3),
        Test(self=[1, 1, 4, 10], dim=3, start=3, step=2),
        Test(self=[1, 1, 4, 10], dim=3, start=3, end=4, step=2),
        Test(self=[1, 1, 4, 10], dim=2, start=3),
        Test(self=[9, 9, 9, 9], dim=2, start=0, end=9, step=1),
        Test(self=[9, 9, 9, 9], dim=2, start=1, end=8, step=1),
        Test(self=[9, 9, 9, 9], dim=2, start=1, end=2, step=1),
        Test(self=[9, 9, 9, 9], dim=3, start=1, end=5, step=1),
        Test(self=[9, 9, 9, 9], dim=3, start=1, end=5, step=2),
        Test(self=[9, 9, 9, 9], dim=-1, start=1, end=5, step=2),
        Test(self=[9, 9, 9, 9], dim=-2, start=1, end=5, step=2),
        Test(self=[9, 9, 9], dim=1, start=2, step=1),
        Test(self=[9, 9, 9], dim=1, start=2, step=2),
        Test(self=[9, 9, 9], dim=2, start=2, step=1),
        Test(self=[9, 9, 9], dim=2, start=2, step=2),
        Test(self=[9, 9], dim=0, start=2, step=1),
        Test(self=[9, 9], dim=0, start=2, step=2),
        Test(self=[9, 9], dim=1, start=2, step=1),
        Test(self=[9, 9], dim=1, start=2, step=2),
    ]

    # Slice by batch
    test_cases += [
        Test(self=[6, 5, 3, 2], dim=0),
        Test(self=[6, 5, 3, 2], dim=0, step=2),
        Test(self=[13, 13, 3, 2], dim=0, step=2),
        Test(self=[13, 13, 3, 2], dim=0, start=1, step=2),
        Test(self=[13, 13, 3, 2], dim=0, start=1, step=5),
        Test(self=[13, 13, 3, 2], dim=0, start=1, step=20),
        Test(self=[13, 2, 3, 2], dim=0, start=1, step=2),
        Test(self=[13, 2, 3, 2], dim=0, start=1, step=5),
        Test(self=[13, 2, 3, 2], dim=0, start=1, step=20),
    ]

    # Slice by channel
    test_cases += [
        Test(self=[2, 5, 1, 10], dim=1),
        Test(self=[2, 5, 1, 10], dim=1, start=1),
        Test(self=[2, 5, 1, 10], dim=1, start=1, step=2),
        Test(self=[5, 13, 1, 10], dim=1),
        Test(self=[5, 13, 1, 10], dim=1, start=1),
        Test(self=[5, 13, 1, 10], dim=1, start=1, step=2),
        Test(self=[5, 13, 1, 10], dim=1, start=1, step=5),
        Test(self=[5, 13, 1, 10], dim=1, start=1, step=20),
        Test(self=[13, 1, 10], dim=0),
        Test(self=[13, 1, 10], dim=0, start=1),
        Test(self=[13, 1, 10], dim=0, start=1, step=2),
        Test(self=[13, 1, 10], dim=0, start=1, step=5),
        Test(self=[13, 1, 10], dim=0, start=1, step=20),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = [
        "api::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


def get_unsqueeze_inputs():
    test_suite = VkTestSuite(
        [
            ((2, 3, 4), 0),
            ((1, 1, 1), 0),
            ((1, 1, 1), 1),
            ((1, 1, 1), 2),
            ((1, 1, 1), 3),
            ((9, 9, 9), 0),
            ((9, 9, 9), 1),
            ((9, 9, 9), 2),
            ((9, 9, 9), 3),
            ((9, 9), 0),
            ((9, 9), 1),
            ((9, 9), 2),
            ((9,), 0),
            ((9,), 1),
        ]
    )
    test_suite.layouts = [
        "api::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


def get_clone_inputs():
    test_suite = VkTestSuite(
        [
            ((S2, S1, S2, S1),),
            ((S2, S1, S2),),
            ((S2, S1),),
            ((S2,),),
            ((XS, S1, XS, S1),),
            ((XS, S1, XS),),
            ((S1, XS, S1),),
            ((XS, S1),),
            ((S1, XS),),
            ((S1,),),
            ((XS,),),
        ]
    )
    test_suite.layouts = [
        "api::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


def get_repeat_inputs():
    test_suite = VkTestSuite(
        [
            # Repeat channels only (most challenging case)
            ((3, XS, S), [2, 1, 1]),
            ((7, XS, S), [4, 1, 1]),
            ((1, 7, XS, S), [1, 4, 1, 1]),
            ((3, 7, XS, S), [1, 4, 1, 1]),
            # Repat channels with other dims
            ((1, 7, XS, S), [1, 4, 1, 3]),
            ((3, 7, XS, S), [1, 4, 1, 3]),
            ((3, 7, XS, S), [1, 4, 3, 1]),
            ((3, 7, XS, S), [1, 4, 3, 3]),
            # Repeat Batch
            ((3, 7, XS, S), [3, 4, 3, 3]),
            ((3, 7, XS, S), [3, 1, 3, 3]),
            # More other cases
            ((3, 7, 1, 1), [1, 4, 1, 1]),
            ((2, 3), [1, 4]),
            ((2, 3), [4, 1]),
            ((2, 3), [4, 4]),
            ((S1, S2, S2), [1, 3, 1]),
            ((S1, S2, S2), [1, 3, 3]),
            ((S1, S2, S2), [3, 3, 1]),
            ((S1, S2, S2), [3, 3, 3]),
            ((S1, S2, S2, S2), [1, 1, 3, 1]),
            ((S1, S2, S2, S2), [1, 1, 1, 3]),
            ((S1, S2, S2, S2), [1, 1, 3, 3]),
            ((S1, S2, S2, S2), [1, 3, 1, 3]),
            ((S1, S2, S2, S2), [3, 3, 3, 3]),
            ((S1, S2, S2, S2), [3, 3, 1, 1]),
        ]
    )
    test_suite.layouts = [
        "api::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    test_suite.dtypes = ["at::kFloat"]
    return test_suite


test_suites = {
    "aten.add.Tensor": get_binary_elementwise_inputs(),
    "aten.sub.Tensor": get_binary_elementwise_inputs(),
    "aten.div.Tensor": get_binary_elementwise_inputs(),
    "aten.mul.Tensor": get_binary_elementwise_inputs(),
    "aten.mm.default": get_mm_inputs(),
    "aten.max_pool2d_with_indices.default": get_pool2d_inputs(),
    "aten.convolution.default": get_conv_inputs(),
    "aten.native_layer_norm.default": get_native_layer_norm_inputs(),
    "aten.full.default": get_full_inputs(),
    "aten.select.int": get_select_int_inputs(),
    "aten.select_copy.int": get_select_int_inputs(),
    "aten.permute.default": get_permute_inputs(),
    "aten.permute_copy.default": get_permute_inputs(),
    "aten.view_copy.default": get_view_inputs(),
    "aten.slice_copy.Tensor": get_slice_inputs(),
    "aten.unsqueeze_copy.default": get_unsqueeze_inputs(),
    "aten.clone.default": get_clone_inputs(),
    "aten.repeat.default": get_repeat_inputs(),
}
