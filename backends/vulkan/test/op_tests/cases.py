# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from collections import namedtuple
from typing import Callable

from executorch.backends.vulkan.test.op_tests.utils.test_suite import VkTestSuite


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

test_suites = {}


def register_test_suite(aten_op):
    def test_suite_decorator(fn: Callable) -> Callable:
        if isinstance(aten_op, str):
            test_suites[aten_op] = fn()
        elif isinstance(aten_op, list):
            for op in aten_op:
                test_suites[op] = fn()
        return fn

    return test_suite_decorator


@register_test_suite(
    ["aten.add.Tensor", "aten.sub.Tensor", "aten.div.Tensor", "aten.mul.Tensor"]
)
def get_binary_elementwise_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, M2), (M1, M2)),
            ((M1, M2), (M1, 1), 2.0),
            ((M1, M2), (1, M2)),
            ((S, S1, S2), (S, S1, S2)),
            ((S, S1, S2), (S, S1, 1), 2.0),
            ((S, S1, S2), (S, 1, S2), 2.0),
            ((XS, S, S1, S2), (XS, S, 1, 1), 2.0),
            ((3, 64, 1), (1, 64, 1)),
        ]
    )
    test_suite.storage_types = [
        "utils::kBuffer",
        "utils::kTexture3D",
    ]

    highdim_test_suite = VkTestSuite(
        [
            ((4, 5, 8, 1, 2, 1), (4, 5, 8, 1, 1, 1)),
        ]
    )
    highdim_test_suite.storage_types = [
        "utils::kBuffer",
    ]
    highdim_test_suite.test_name_suffix = "highdim"

    for suite in [test_suite, highdim_test_suite]:
        suite.layouts = [
            "utils::kWidthPacked",
            "utils::kChannelsPacked",
        ]

    return [test_suite, highdim_test_suite]


# Eq requires a different test generator so it was split from the other test case.
@register_test_suite(
    [
        "aten.eq.Tensor",
        "aten.gt.Tensor",
        "aten.lt.Tensor",
        "aten.ge.Tensor",
        "aten.le.Tensor",
    ]
)
def get_binary_elementwise_compare_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, M2), (M1, M2)),
            ((M1, M2), (M1, 1), 2.0),
            ((M1, M2), (1, M2)),
            ((S, S1, S2), (S, S1, S2)),
            ((S, S1, S2), (S, S1, 1), 2.0),
            ((S, S1, S2), (S, 1, S2), 2.0),
            ((XS, S, S1, S2), (XS, S, 1, 1), 2.0),
            ((3, 64, 1), (1, 64, 1)),
        ]
    )
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = [
        "utils::kBuffer",
        "utils::kTexture3D",
    ]
    test_suite.data_gen = "make_casted_randint_tensor"
    return test_suite


@register_test_suite("aten.mm.default")
def get_mm_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, L), (L, M2)),
            ((S1, S2), (S2, M)),
            ((6, 32), (32, 64)),
        ],
    )
    test_suite.prepacked_args = ["mat2"]
    # ATen matmul doesn't support half
    test_suite.dtypes = ["at::kFloat"]
    test_suite.storage_types = ["utils::kTexture3D", "utils::kBuffer"]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    return test_suite


@register_test_suite("aten.bmm.default")
def get_bmm_inputs():
    test_suite = VkTestSuite(
        [
            ((S, M1, L), (S, L, M2)),
            ((M, S1, S2), (M, S2, M)),
            ((4, 6, 32), (4, 32, 16)),
        ],
    )
    test_suite.prepacked_args = ["mat2"]
    # ATen matmul doesn't support half
    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    return test_suite


@register_test_suite("aten.addmm.default")
def get_addmm_inputs():
    test_suite = VkTestSuite(
        [
            ((1, S), (S1, S), (S, S), 1.0, 1.5),
            ((S, 1), (S, S1), (S1, S1), 1.0, 1.0),
            ((M1, M2), (M1, M2), (M2, M2)),
            ((M1, M2), (M1, M2), (M2, M2), 4.2, 2.3),
            ((M1, 1), (M1, L), (L, L), 2.0, 3.0),
            ((M2), (M1, M2), (M2, M2)),
            ((6, M2), (6, M2), (M2, M2)),
        ]
    )
    # ATen matmul doesn't support half
    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    return test_suite


common_MKN_list = [
    (S2, M2, M1),
    (L, L, M1),
]


@register_test_suite("aten.linear.default")
def get_linear_inputs():
    MKN_list = common_MKN_list

    inputs_list = [((M, K), (N, K), None) for M, K, N in MKN_list]
    inputs_list += [((M, K), (N, K), (N)) for M, K, N in MKN_list]
    inputs_list += [((3, M, K), (N, K), None) for M, K, N in MKN_list]
    inputs_list += [((3, M, K), (N, K), (N)) for M, K, N in MKN_list]
    inputs_list += [((3, 6, K), (N, K), (N)) for M, K, N in MKN_list]

    test_suite = VkTestSuite(inputs_list)
    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    return test_suite


@register_test_suite("aten._weight_int8pack_mm.default")
def get_weight_int8pack_mm_inputs():
    MKN_list = [
        [1, 480, 256],
        [1, 1024, 1024],
        [1, 1024, 256],
        [3, 480, 256],
        [6, 480, 256],
        [6, 256, 1024],
        [6, 1024, 256],
        [6, 256, 256],
        [6, 256, 512],
        [4, 768, 4096],
        [1024, 1024, 1024],
    ]

    inputs_list = [((M, K), (N, K), (N)) for M, K, N in MKN_list]

    test_suite = VkTestSuite(inputs_list)
    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = ["utils::kWidthPacked"]
    test_suite.storage_types = ["utils::kTexture3D", "utils::kBuffer"]
    test_suite.prepacked_args = ["mat2", "scales"]
    test_suite.requires_prepack = True

    test_suite.arg_dtype["mat2"] = "at::kChar"
    test_suite.arg_data_range["mat2"] = (0, 100)

    test_suite.arg_data_range["scales"] = (0.0008, 0.001)

    return test_suite


@register_test_suite("aten.avg_pool2d.default")
def get_avg_pool2d_inputs():
    Test = namedtuple(
        "VkAvgPoolTest",
        [
            "self",
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        ],
    )

    test_cases = []
    for ceil_mode in [True, False]:
        for count_include_pad in [True, False]:
            for divisor_override in [None, 5]:
                test_cases += [
                    Test(
                        self=(S, M1, M2),
                        kernel_size=[2, 2],
                        stride=[1, 1],
                        padding=[0, 0],
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                        divisor_override=divisor_override,
                    ),
                ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])
    test_suite.dtypes = ["at::kFloat"]
    return test_suite


@register_test_suite(
    ["aten.max_pool2d_with_indices.default", "aten.max_pool2d.default"]
)
def get_max_pool2d_inputs():
    test_suite = VkTestSuite(
        [
            ((1, 7, 89, 77), [2, 2], [1, 1], [0, 0], [1, 1]),
        ]
    )
    return test_suite


@register_test_suite("aten.convolution.default")
def get_conv_inputs():
    Test = namedtuple(
        "ConvTest",
        [
            "self",
            "weight",
            "bias",
            "stride",
            "padding",
            "dilation",
            "transposed",
            "output_padding",
            "groups",
        ],
    )
    Test.__new__.__defaults__ = (
        None,
        None,
        None,
        [1, 1],
        [0, 0],
        [1, 1],
        False,
        [9, 0],
        1,
    )

    test_cases = [
        Test(
            self=(1, 64, 256, 256),
            weight=(64, 32, 3, 3),
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=2,
        ),
        Test(
            self=(1, 16, 3, 3),
            weight=(16, 8, 3, 3),
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=2,
        ),
        Test(
            self=(1, 6, 40, 50),
            weight=(8, 6, 3, 3),
            bias=(8,),
            stride=[1, 2],
            padding=[2, 3],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 6, 40, 50),
            weight=(6, 8, 3, 3),
            bias=(8,),
            stride=[1, 2],
            padding=[2, 3],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 1],
            groups=1,
        ),
        Test(
            self=(1, 6, 40, 50),
            weight=(8, 6, 3, 3),
            bias=None,
            stride=[1, 2],
            padding=[2, 3],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 6, 7),
            weight=(6, 1, 3),
            bias=(6,),
            stride=[1],
            padding=[0],
            dilation=[1],
            transposed=False,
            output_padding=[0],
            groups=6,
        ),
        Test(
            self=(2, 20, 30),
            weight=(10, 4, 6),
            bias=(10,),
            stride=[5],
            padding=[5],
            dilation=[3],
            transposed=False,
            output_padding=[0],
            groups=5,
        ),
        Test(
            self=(1, 9, 11),
            weight=(9, 1, 3),
            bias=None,
            stride=[1],
            padding=[0],
            dilation=[1],
            transposed=False,
            output_padding=[0],
            groups=9,
        ),
        Test(
            self=(5, 15, 30),
            weight=(20, 3, 3),
            bias=None,
            stride=[3],
            padding=[5],
            dilation=[7],
            transposed=False,
            output_padding=[0],
            groups=5,
        ),
        Test(
            self=(1, 8, 90, 77),
            weight=(1, 8, 3, 3),
            bias=(1,),
            stride=[1, 1],
            padding=[2, 2],
            dilation=[2, 2],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
    ]

    test_cases_pw = [
        Test(
            self=(1, 16, 3, 5),
            weight=(4, 16, 1, 1),
            bias=(4,),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 5, 3, 5),
            weight=(4, 5, 1, 1),
            bias=(4,),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 5, 3, 5),
            weight=(3, 5, 1, 1),
            bias=(3,),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 5, 3, 5),
            weight=(3, 5, 1, 1),
            bias=(3,),
            stride=[1, 1],
            padding=[1, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 5, 3, 5),
            weight=(3, 5, 1, 1),
            bias=(3,),
            stride=[1, 1],
            padding=[0, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 5, 3, 5),
            weight=(3, 5, 1, 1),
            bias=(3,),
            stride=[2, 1],
            padding=[1, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 8, 72, 96),
            weight=(8, 8, 1, 1),
            bias=(8,),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 16, 240, 320),
            weight=(64, 16, 1, 1),
            bias=(64,),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 16, 240, 320),
            weight=(64, 16, 1, 1),
            bias=(64,),
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 16, 240, 320),
            weight=(64, 16, 1, 1),
            bias=(64,),
            stride=[4, 4],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 16, 240, 320),
            weight=(64, 16, 1, 1),
            bias=(64,),
            stride=[1, 1],
            padding=[4, 4],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
        Test(
            self=(1, 16, 672, 512),
            weight=(64, 16, 1, 1),
            bias=(64,),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
    ]

    test_cases_dw = [
        Test(
            self=(1, XS, S, S1),
            weight=(XS, 1, 3, 3),
            bias=(XS,),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=XS,
        ),
        Test(
            self=(1, XS, S, S1),
            weight=(XS, 1, 5, 5),
            bias=(XS,),
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=XS,
        ),
        Test(
            self=(1, XS, S, S1),
            weight=(XS, 1, 3, 3),
            bias=(XS,),
            stride=[2, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=XS,
        ),
        Test(
            self=(1, XS, S, S1),
            weight=(XS, 1, 5, 5),
            bias=(XS,),
            stride=[1, 2],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=XS,
        ),
        Test(
            self=(1, S2, S, S1),
            weight=(S2, 1, 3, 3),
            bias=(S2,),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=S2,
        ),
        Test(
            self=(1, S2, S, S1),
            weight=(S2, 1, 5, 5),
            bias=(S2,),
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=S2,
        ),
        Test(
            self=(1, 8, 72, 96),
            weight=(8, 1, 3, 3),
            bias=(8,),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=8,
        ),
        Test(
            self=(1, 8, 72, 96),
            weight=(8, 1, 5, 5),
            bias=(8,),
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=8,
        ),
        Test(
            self=(1, 4, 234, 234),
            weight=(4, 1, 3, 3),
            bias=(4,),
            stride=[2, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=4,
        ),
        Test(
            self=(1, 4, 234, 234),
            weight=(4, 1, 3, 3),
            bias=(4,),
            stride=[1, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=4,
        ),
        Test(
            self=(1, 4, 234, 234),
            weight=(4, 1, 3, 3),
            bias=(4,),
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=4,
        ),
    ]

    test_suite = VkTestSuite(test_cases)
    test_suite.layouts = [
        "utils::kChannelsPacked",
    ]

    test_suite_pw = VkTestSuite(test_cases_pw)
    test_suite_pw.layouts = [
        "utils::kChannelsPacked",
    ]
    test_suite_pw.test_name_suffix = "pw"

    test_suite_dw = VkTestSuite(test_cases_dw)
    test_suite_dw.layouts = [
        "utils::kChannelsPacked",
    ]
    test_suite_dw.test_name_suffix = "dw"
    return [test_suite, test_suite_pw, test_suite_dw]


@register_test_suite("aten.native_layer_norm.default")
def get_native_layer_norm_inputs():
    test_suite = VkTestSuite(
        [
            ((S1, S2), [S2], (S2), (S2), 0.001),
            ((M, M1, M2), [M2], (M2), (M2), 0.001),
            ((S, XL, M1, M2), [M2], (M2), (M2), 0.001),
        ]
    )
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    return test_suite


@register_test_suite("aten.native_group_norm.default")
def get_native_group_norm_inputs():
    test_suite = VkTestSuite(
        [
            # (input_shape, weight_shape, bias_shape, N, C, HxW, group, eps)
            # General test cases
            ((1, 8, 4, 4), (8), (8), 1, 8, 16, 2, 0.001),
            ((2, 8, 3, 3), (8), (8), 2, 8, 9, 4, 0.001),
            ((1, 12, 2, 2), (12), (12), 1, 12, 4, 3, 0.001),
            ((3, 16, 5, 5), (16), (16), 3, 16, 25, 8, 0.001),
            ((3, 16, 13, 17), (16), (16), 3, 16, 13 * 17, 4, 0.001),
            ((1, 4, 7, 7), (4), (4), 1, 4, 49, 2, 0.001),
            ((2, 6, 1, 8), (6), (6), 2, 6, 8, 3, 0.001),
            # Single group and prime number sizes
            ((3, 7, 13, 11), (7), (7), 3, 7, 13 * 11, 1, 0.001),
            # Each channel is it's own group and prime number sizes
            ((1, 7, 13, 11), (7), (7), 1, 7, 13 * 11, 7, 0.001),
        ]
    )
    test_suite.layouts = [
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = [
        "utils::kTexture3D",
    ]
    test_suite.dtypes = [
        "at::kFloat",
        "at::kHalf",
    ]
    test_suite.arg_storage_types = {
        "out": [None, "utils::kBuffer", "utils::kBuffer"],
    }

    test_suite.prepacked_args = ["weight", "bias"]
    test_suite.requires_prepack = True

    return test_suite


def get_upsample_inputs():
    inputs_list = [
        # (input tensor shape, output 2D image size (H, W), output scaling factors)
        ((2, 2, 2, 2), None, [1, 1]),
        ((1, 1, 2, 2), None, [2, 2]),
        ((1, 1, 2, 2), None, [2, 4]),
        ((1, 1, 2, 2), None, [4, 2]),
        ((1, 1, 2, 2), [2, 2], None),
        ((1, 1, 2, 2), [2, 4], None),
        ((1, 1, 2, 2), [3, 2], None),
    ]
    return inputs_list


@register_test_suite("aten.upsample_nearest2d.vec")
def get_upsample_nearest2d_inputs():
    inputs_list = get_upsample_inputs()
    return VkTestSuite(inputs_list)


@register_test_suite("aten.upsample_bilinear2d.vec")
def get_upsample_bilinear2d_inputs():
    base_inputs_list = get_upsample_inputs()
    inputs_list = []
    for input_case in base_inputs_list:
        inputs_list.append((input_case[0], input_case[1], False, input_case[2]))
        inputs_list.append((input_case[0], input_case[1], True, input_case[2]))
    return VkTestSuite(inputs_list)


@register_test_suite(["aten.full.default", "aten.full_like.default"])
def get_full_inputs():
    test_suite = VkTestSuite(
        [
            ([S1, S2], 42.0),
            ([M, M1, M2], 3.14),
            ([L, M, M1, M2], 2.72),
        ]
    )
    return test_suite


@register_test_suite("aten.scalar_tensor.default")
def get_scalar_tensor_inputs():
    test_suite = VkTestSuite(
        [
            (42.0,),
            (3.14,),
            (2.72,),
            (0.0,),
            (-1.0,),
            (100.0,),
        ]
    )
    return test_suite


@register_test_suite(
    [
        "aten.zeros.default",
        "aten.zeros_like.default",
        "aten.ones.default",
        "aten.ones_like.default",
    ]
)
def get_ones_inputs():
    test_suite = VkTestSuite(
        [
            ([S1, S2]),
            ([M, M1, M2]),
            ([L, M, M1, M2]),
        ]
    )
    return test_suite


@register_test_suite(["aten.select.int", "aten.select_copy.int"])
def get_select_int_inputs():
    test_suite = VkTestSuite(
        [
            ((8, 8, 8), 0, -2),
            ((8, 8, 8), 1, -3),
            ((8, 8, 8), 2, -4),
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
    test_suite.layouts = ["utils::kWidthPacked", "utils::kChannelsPacked"]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    test_suite.dtypes = ["at::kFloat"]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


@register_test_suite(["aten.permute.default", "aten.permute_copy.default"])
def get_permute_inputs():
    batch_tests = [
        ((9, 2, 5, 7), out_axis) for out_axis in itertools.permutations([0, 1, 2, 3])
    ]
    channel_tests = [
        ((9, 2, 5), out_axis) for out_axis in itertools.permutations([0, 1, 2])
    ]
    wh_tests = [((9, 2), out_axis) for out_axis in itertools.permutations([0, 1])]
    test_suite = VkTestSuite(batch_tests + channel_tests + wh_tests)

    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = [
        "utils::kBuffer",
        "utils::kTexture3D",
    ]
    test_suite.dtypes = [
        "at::kFloat",
    ]
    return test_suite


@register_test_suite("aten.view_copy.default")
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
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]

    highdim_test_suite = VkTestSuite(
        [
            ((1, 1, 3, 3, 3), (9, 3)),
            ((2, 3, 4, 6, 5, 4), (6, 4, 6, 5, 4)),
            ((2, 3, 3, 7, 8), (2, 3, 3, 8 * 7)),
        ]
    )
    highdim_test_suite.storage_types = [
        "utils::kBuffer",
    ]
    highdim_test_suite.test_name_suffix = "highdim"
    highdim_test_suite.data_gen = "make_seq_tensor"

    for suite in [test_suite, highdim_test_suite]:
        suite.layouts = [
            # "utils::kWidthPacked",
            "utils::kHeightPacked",
            "utils::kChannelsPacked",
        ]

    return [test_suite, highdim_test_suite]


@register_test_suite("aten.slice_copy.Tensor")
def get_slice_out_inputs():
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

    # Slice by negative/unspecified indices
    INT64_MAX = 9223372036854775807  # represents arr[:]
    test_cases += [
        Test(self=[8, 9], dim=0, start=-2, step=1),
        Test(self=[8, 9], dim=0, start=-2, step=2),
        Test(self=[8, 9], dim=0, end=-2, step=1),
        Test(self=[8, 9], dim=0, end=-2, step=2),
        Test(self=[8, 9], dim=0, end=INT64_MAX, step=1),
        Test(self=[8, 9], dim=0, end=INT64_MAX, step=2),
        Test(self=[8, 9], dim=1, start=-2, step=1),
        Test(self=[8, 9], dim=1, start=-2, step=2),
        Test(self=[8, 9], dim=1, end=-2, step=1),
        Test(self=[8, 9], dim=1, end=-2, step=2),
        Test(self=[8, 9], dim=1, end=INT64_MAX, step=1),
        Test(self=[8, 9], dim=1, end=INT64_MAX, step=2),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.dtypes = ["at::kFloat", "at::kHalf"]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


def get_slice_view_inputs():
    Test = namedtuple("VkSliceTest", ["self", "dim", "start", "end", "step"])
    Test.__new__.__defaults__ = (None, 0, None, None, 1)

    # Slice by channel
    test_cases = [
        Test(self=[1, 17, 1, 10], dim=1, start=0, end=4),
        Test(self=[1, 17, 1, 10], dim=1, start=0, end=8),
        Test(self=[1, 17, 3, 7], dim=1, start=0, end=12),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.dtypes = ["at::kFloat"]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    test_suite.layouts = ["utils::kWidthPacked"]
    test_suite.data_gen = "make_seq_tensor"
    test_suite.is_view_op = True

    return test_suite


@register_test_suite(["aten.slice.Tensor"])
def get_slice_inputs():
    texture_test_suite = get_slice_out_inputs()
    texture_test_suite.test_name_suffix = "no_view"

    view_test_suite = get_slice_view_inputs()
    view_test_suite.test_name_suffix = "view"

    return [view_test_suite, texture_test_suite]


@register_test_suite(["aten.transpose.int"])
def get_transpose_inputs():
    Test = namedtuple("VkTransposeViewTest", ["self", "dim0", "dim1"])
    Test.__new__.__defaults__ = (None, 0, 1)

    test_cases = [
        Test(self=[M1, M2], dim0=0, dim1=1),
        Test(self=[M1, S2, M], dim0=0, dim1=1),
        Test(self=[M1, S2, M], dim0=0, dim1=2),
        Test(self=[M1, S2, M], dim0=2, dim1=1),
        Test(self=[S, M, S2, M2], dim0=3, dim1=2),
        Test(self=[S, M, S2, M2], dim0=1, dim1=2),
        Test(self=[S, M, S2, M2], dim0=3, dim1=1),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.dtypes = ["at::kFloat"]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    test_suite.layouts = ["utils::kWidthPacked", "utils::kChannelsPacked"]
    test_suite.data_gen = "make_seq_tensor"
    test_suite.is_view_op = True
    return test_suite


@register_test_suite("aten.index_select.default")
def get_index_select_inputs():
    Test = namedtuple("VkIndexSelectTest", ["self", "dim", "index"])
    Test.__new__.__defaults__ = (None, 0, None)

    test_cases = []

    for i in range(4):
        test_cases += [
            Test(self=[9, 9, 9, 9], dim=i, index=[0]),
            Test(self=[9, 9, 9, 9], dim=i, index=[2]),
            Test(self=[9, 9, 9, 9], dim=i, index=[0, 2]),
            Test(self=[9, 9, 9, 9], dim=i, index=[3, 1]),
            Test(self=[9, 9, 9, 9], dim=i, index=[5, 5]),
            Test(self=[9, 9, 9, 9], dim=i, index=[2, 3, 4, 5, 7]),
        ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = ["utils::kChannelsPacked"]
    return test_suite


@register_test_suite("aten.embedding.default")
def get_embedding_inputs():
    Test = namedtuple("EmbeddingTest", ["weight", "indices"])
    Test.__new__.__defaults__ = (None, None)

    test_cases = [
        Test(weight=[10, 9], indices=[3, 5]),
        Test(weight=[10, 9], indices=[2, 3, 4, 5, 7]),
        Test(weight=[10, 9], indices=[[0, 2], [1, 4], [7, 7]]),
        Test(weight=[10, 9], indices=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]),
    ]

    # Channels packed test cases currently fail on Mac, so they are not included.
    # However the test case definition is kept for later debugging.
    test_suite_cpack = VkTestSuite(
        [tuple(tc) + (-1, "false", "false") for tc in test_cases]
    )

    test_suite_cpack.dtypes = ["at::kFloat"]
    test_suite_cpack.layouts = ["utils::kChannelsPacked"]
    test_suite_cpack.test_name_suffix = "cpacked"

    test_suite_wpack = VkTestSuite(
        [tuple(tc) + (-1, "false", "false") for tc in test_cases]
    )

    test_suite_wpack.dtypes = ["at::kFloat"]
    test_suite_wpack.layouts = ["utils::kWidthPacked"]
    test_suite_wpack.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    test_suite_wpack.test_name_suffix = "wpacked"

    return test_suite_wpack


@register_test_suite("aten.gather.default")
def get_gather_inputs():
    Test = namedtuple("GatherTest", ["input", "dim", "index"])
    Test.__new__.__defaults__ = (None, None, None)

    test_cases = [
        # Simple 2D case
        Test(input=[4, 4], dim=1, index=[[1, 2], [2, 1], [3, 3], [3, 1]]),
        # # 1D cases
        Test(input=[10], dim=0, index=[0, 2, 5, 7, 9]),
        Test(input=[8], dim=0, index=[1, 3, 5]),
        # # 2D cases with different dims
        Test(input=[5, 8], dim=0, index=[[0, 1], [2, 3], [4, 0]]),
        Test(
            input=[5, 8],
            dim=1,
            index=[[0, 2, 4], [1, 3, 5], [6, 7, 0], [1, 2, 3], [4, 5, 6]],
        ),
        # # 3D cases
        Test(
            input=[3, 4, 5],
            dim=0,
            index=[
                [[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0], [0, 1, 2, 0, 1]]
            ],
        ),
        Test(
            input=[3, 4, 5],
            dim=1,
            index=[
                [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [0, 1, 2, 3]]
            ],
        ),
        Test(
            input=[3, 4, 5], dim=2, index=[[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]]]
        ),
    ]

    test_suite = VkTestSuite(
        [tuple(tc) + (False, "false", "false") for tc in test_cases]
    )

    test_suite.dtypes = ["at::kFloat"]
    test_suite.layouts = ["utils::kWidthPacked", "utils::kChannelsPacked"]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]

    return test_suite


@register_test_suite("aten.unsqueeze_copy.default")
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
            ((1, 10), -1),
        ]
    )

    highdim_test_suite = VkTestSuite(
        [
            ((2, 3, 4, 5, 6), 0),
            ((2, 3, 4, 5, 6), 1),
            ((2, 3, 4, 5, 6), 5),
            ((2, 3, 4, 5, 6), -1),
            ((2, 3, 4, 5, 6), -2),
            ((1, 2, 3, 4, 5), 0),
            ((1, 2, 3, 4, 5), 3),
            ((1, 2, 3, 4, 5), -1),
            ((2, 3, 4, 5), 0),
            ((1, 2, 3, 4), 1),
        ]
    )
    highdim_test_suite.storage_types = [
        "utils::kBuffer",
    ]
    highdim_test_suite.test_name_suffix = "highdim"

    for suite in [test_suite, highdim_test_suite]:
        suite.layouts = [
            "utils::kWidthPacked",
            "utils::kChannelsPacked",
        ]
        suite.data_gen = "make_seq_tensor"

    return [test_suite, highdim_test_suite]


@register_test_suite("aten.clone.default")
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

    highdim_test_suite = VkTestSuite(
        [
            ((2, 3, 4, 5, 6),),
            ((2, 3, 4, 5, 1),),
            ((1, 1, 3, 4, 5),),
            ((2, 3, 4, 5, 6, 7),),
            ((1, 2, 3, 4, 5, 6),),
        ]
    )
    highdim_test_suite.storage_types = [
        "utils::kBuffer",
    ]
    highdim_test_suite.test_name_suffix = "highdim"

    for suite in [test_suite, highdim_test_suite]:
        suite.layouts = [
            "utils::kChannelsPacked",
        ]
        suite.data_gen = "make_seq_tensor"

    return [test_suite, highdim_test_suite]


@register_test_suite("aten.repeat.default")
def get_repeat_inputs():
    test_suite_2d = VkTestSuite(
        [
            ((2, 3), [1, 4]),
            ((2, 3), [4, 1]),
            ((2, 3), [4, 4]),
            ((2, 3), [3, 1, 4]),
        ]
    )
    test_suite_2d.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    test_suite_2d.storage_types = ["utils::kTexture3D"]
    test_suite_2d.data_gen = "make_seq_tensor"
    test_suite_2d.dtypes = ["at::kFloat"]
    test_suite_2d.test_name_suffix = "2d"

    test_suite_3d = VkTestSuite(
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
            # Expanding cases
            ((2, 3), [3, 1, 4]),
            ((2, 3), [3, 3, 2, 4]),
        ]
    )
    test_suite_3d.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    test_suite_3d.storage_types = ["utils::kTexture3D"]
    test_suite_3d.data_gen = "make_seq_tensor"
    test_suite_3d.dtypes = ["at::kFloat"]
    test_suite_3d.test_name_suffix = "3d"

    return [test_suite_2d, test_suite_3d]


@register_test_suite("aten.repeat_interleave.self_int")
def get_repeat_interleave_inputs():
    test_suite_W = VkTestSuite(
        [
            ((4, 32, 256), 3, -2),
            # Test repeat on each non-packed dim
            ((16, 32, 64), 5, -2),
            ((16, 32, 64), 5, -3),
            # Test batched inputs
            ((3, 5, 32, 64), 4, -2),
            ((3, 5, 32, 64), 4, -3),
        ]
    )
    test_suite_W.layouts = [
        "utils::kWidthPacked",
    ]
    test_suite_W.data_gen = "make_seq_tensor"
    test_suite_W.dtypes = ["at::kFloat"]
    test_suite_W.test_name_suffix = "W_packed"

    test_suite_C = VkTestSuite(
        [
            # Test repeat on each non-packed dim
            ((32, 32, 16), 5, -1),
            ((32, 32, 16), 5, -2),
            # Test batched inputs
            ((3, 16, 8, 64), 4, -1),
            ((3, 16, 8, 64), 4, -2),
        ]
    )
    test_suite_C.layouts = [
        "utils::kChannelsPacked",
    ]
    test_suite_C.data_gen = "make_seq_tensor"
    test_suite_C.dtypes = ["at::kFloat"]
    test_suite_C.test_name_suffix = "C_packed"

    return [test_suite_W, test_suite_C]


@register_test_suite("aten.cat.default")
def get_cat_inputs():
    # TensorList must be specified as list of tuples
    suite_inputs = [
        # Cat on Height
        ([(M, M, 3, 5), (M, M, 0, 5)], 2),
        ([(S1, S1, 3, 5), (S1, S1, 0, 5)], 2),
        ([(M, M, 3, 5), (M, M, 4, 5)], 2),
        ([(S1, S1, 3, 5), (S1, S1, 4, 5)], 2),
        ([(M2, 3, 5), (M2, 4, 5)], 1),
        ([(S1, 3, 5), (S1, 4, 5)], 1),
        ([(3, 5), (4, 5)], 0),
        ([(3, 5), (4, 5), (1, 5)], 0),
        (
            [(3, 5)],
            0,
        ),
        # Cat on Width
        ([(M, M, 5, 3), (M, M, 5, 4)], 3),
        ([(S1, S1, 5, 3), (S1, S1, 5, 4)], 3),
        ([(M, 5, 3), (M, 5, 4)], 2),
        ([(S1, 5, 3), (S1, 5, 4)], 2),
        ([(5, 0), (5, 4)], 1),
        ([(5, 3), (5, 4)], 1),
        ([(5, 3), (5, 4), (5, 1)], 1),
        (
            [(5, 4)],
            1,
        ),
        ([(5,), (6,)], 0),
        # Cat on Batch
        ([(M, S1, 5, 4), (M1, S1, 5, 4)], 0),
        ([(S, S1, 5, 4), (S1, S1, 5, 4)], 0),
        ([(S, M, 5, 4), (S1, M, 5, 4)], 0),
        ([(S, XS, 5, 4), (S1, XS, 5, 4)], 0),
        ([(S, S2, 5, 4), (S1, S2, 5, 4)], 0),
        (
            [
                (3, 1, 2, 5),
                (3, 1, 2, 5),
                (3, 1, 2, 5),
            ],
            0,
        ),
        # Cat on Channel
        ([(M, 5, 4), (0, 5, 4), (M1, 5, 4)], 0),
        ([(S, 5, 4), (0, 5, 4), (S2, 5, 4)], 0),
        ([(M, 5, 4), (M1, 5, 4), (M2, 5, 4)], 0),
        ([(S, 5, 4), (S1, 5, 4), (S2, 5, 4)], 0),
        ([(XS, 5, 4), (XS, 5, 4), (S2, 5, 4)], 0),
        ([(XS, S, 5, 4), (XS, S1, 5, 4), (XS, S2, 5, 4)], 1),
        ([(XS, XS, 5, 4), (XS, XS, 5, 4), (XS, S2, 5, 4)], 1),
        (
            [
                (XS, 1, 2, 5),
                (XS, 1, 2, 5),
                (XS, 1, 2, 5),
            ],
            1,
        ),
    ]

    high_number_cat_inputs = []
    for num_input in [6, 9]:
        odd_size = (3, 7, 29, 31)
        even_size = (3, 8, 29, 32)
        ones = (3, 1, 1, 1)

        for input_size in [odd_size, even_size, ones]:
            input_sizes = [input_size] * num_input
            # Test cat on height, width, and batch dim
            high_number_cat_inputs.append((input_sizes, 3))
            high_number_cat_inputs.append((input_sizes, 2))
            high_number_cat_inputs.append((input_sizes, 1))
            high_number_cat_inputs.append((input_sizes, 0))

    test_suite = VkTestSuite(suite_inputs + high_number_cat_inputs)

    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = [
        "utils::kTexture3D",
        "utils::kBuffer",
    ]
    test_suite.data_gen = "make_seq_tensor"
    test_suite.dtypes = ["at::kFloat"]
    return test_suite


@register_test_suite("aten.split_with_sizes_copy.default")
def get_split_with_sizes_inputs():
    Test = namedtuple("VkSliceTest", ["self", "sizes", "dim"])
    test_cases = [
        # Split on Width
        Test(self=(S1, 7, 10, 11), sizes=[1, 3, 2, 5], dim=3),
        Test(self=(S1, 7, 10, 10), sizes=[1, 2, 3, 4], dim=3),
        Test(self=(7, 10, 11), sizes=[1, 3, 2, 5], dim=2),
        Test(self=(7, 10, 10), sizes=[1, 2, 3, 4], dim=2),
        Test(self=(7, 10, 11), sizes=[3, 8], dim=2),
        Test(self=(7, 10, 10), sizes=[1, 9], dim=2),
        Test(self=(10, 10), sizes=[1, 9], dim=1),
        Test(self=(10,), sizes=[1, 9], dim=0),
        # Split on Height
        Test(self=(S1, 7, 11, 10), sizes=[1, 3, 2, 5], dim=2),
        Test(self=(S1, 7, 10, 10), sizes=[1, 2, 3, 4], dim=2),
        Test(self=(7, 11, 10), sizes=[1, 3, 2, 5], dim=1),
        Test(self=(7, 10, 10), sizes=[1, 2, 3, 4], dim=1),
        Test(self=(7, 11, 11), sizes=[3, 8], dim=1),
        Test(self=(7, 10, 10), sizes=[10], dim=1),
        Test(self=(7, 6, 10), sizes=[1, 1, 1, 1, 1, 1], dim=1),
        Test(self=(10, 10), sizes=[1, 2, 3, 4], dim=0),
        # Split on Batch
        Test(self=(10, 7, 10, 10), sizes=[3, 6, 1], dim=0),
        Test(self=(10, 7, 10, 10), sizes=[10], dim=0),
        # Split on Channel
        Test(self=(7, 13, 4, 8), sizes=[3, 5, 2, 3], dim=1),
        Test(self=(7, 13, 4, 8), sizes=[3, 6, 1, 3], dim=1),
        Test(self=(7, 13, 4, 8), sizes=[3, 2, 2, 5, 1], dim=1),
        Test(self=(7, 13, 4, 8), sizes=[3, 3, 3, 3, 1], dim=1),
        Test(self=(13, 4, 8), sizes=[3, 5, 2, 1, 2], dim=0),
        Test(self=(13, 4, 8), sizes=[3, 3, 3, 3, 1], dim=0),
        Test(self=(13, 4, 8), sizes=[2, 9, 2], dim=0),
        Test(self=(13, 4, 8), sizes=[13], dim=0),
    ]
    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])

    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.data_gen = "make_seq_tensor"
    test_suite.dtypes = ["at::kFloat"]
    test_suite.storage_types = ["utils::kBuffer", "utils::kTexture3D"]
    return test_suite


def get_reduce_inputs(is_softmax: bool = False):
    bool_arg = False if is_softmax else True
    return [
        ((L), 0, bool_arg),
        ((L), -1, bool_arg),
        ((M, L), 0, bool_arg),
        ((M, L), 1, bool_arg),
        ((L, M), -1, bool_arg),
        ((M, L), -2, bool_arg),
        ((S, S1, S2), 0, bool_arg),
        ((S, S1, S2), 1, bool_arg),
        ((S, S1, S2), 2, bool_arg),
        ((S, S1, S2), -1, bool_arg),
        ((S, S1, S2), -2, bool_arg),
        ((S, S1, S2), -3, bool_arg),
        ((1, S, S1, S2), 1, bool_arg),
        ((1, S, S1, S2), 2, bool_arg),
        ((1, S, S1, S2), 3, bool_arg),
        ((1, S, S1, S2), -1, bool_arg),
        ((1, S, S1, S2), -2, bool_arg),
        ((1, S, S1, S2), -3, bool_arg),
        # Test batches > 1 where the reduction dim is not the concat dim
        ((S, S2, S1, 128), -1, bool_arg),
    ]


def get_reduce_per_row_inputs():
    inputs = [
        ((5, 10), 1, False),
        ((5, 16), -1, True),
        ((5, 16), -1, False),
        ((7, 21), -1, True),
        ((7, 21), -1, False),
        ((3, 7, 280), -1, True),
        ((3, 7, 280), -1, False),
        ((3, 17, 77), -1, True),
        ((3, 17, 77), -1, False),
    ]
    return inputs


@register_test_suite(["aten._softmax.default", "aten._log_softmax.default"])
def get_softmax_inputs():
    test_suite = VkTestSuite(get_reduce_inputs(is_softmax=True))
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    return test_suite


@register_test_suite(
    ["aten.amax.default", "aten.amin.default", "aten.sum.dim_IntList", "aten.mean.dim"]
)
def get_reduce_op_inputs():
    test_suite = VkTestSuite(get_reduce_inputs())
    test_suite.layouts = [
        "utils::kChannelsPacked",
        "utils::kWidthPacked",
    ]

    per_row_suite = VkTestSuite(get_reduce_per_row_inputs())
    per_row_suite.layouts = ["utils::kWidthPacked"]
    per_row_suite.storage_types = ["utils::kBuffer"]
    per_row_suite.test_name_suffix = "per_row"
    return [test_suite, per_row_suite]


@register_test_suite(["aten.argmin.default", "aten.argmax.default"])
def get_reduce_arg_op_inputs():
    test_suite = VkTestSuite(get_reduce_per_row_inputs())
    test_suite.layouts = ["utils::kWidthPacked"]
    test_suite.storage_types = ["utils::kBuffer"]
    test_suite.dtypes = ["at::kFloat"]
    return test_suite


@register_test_suite(["aten.var.dim"])
def get_var_inputs():
    test_cases = []
    shapes_and_dims = [
        ((L), 0),
        ((L), -1),
        ((M, L), 0),
        ((M, L), 1),
        ((L, M), -1),
        ((M, L), -2),
        ((S, S1, S2), 0),
        ((S, S1, S2), 1),
        ((S, S1, S2), 2),
        ((S, S1, S2), -1),
        ((S, S1, S2), -2),
        ((S, S1, S2), -3),
        ((1, S, S1, S2), 1),
        ((1, S, S1, S2), 2),
        ((1, S, S1, S2), 3),
        ((1, S, S1, S2), -1),
        ((1, S, S1, S2), -2),
        ((1, S, S1, S2), -3),
        # Test batches > 1 where the reduction dim is not the concat dim
        ((S, L, S1, L), -1),
        ((S, S2, S1, S), -2),
        ((S, S2, M, M), 2),
        ((S, M, S1, L), 3),
    ]

    for i, (shape, dim) in enumerate(shapes_and_dims):
        unbiased = (i % 2) == 0
        test_cases.append((shape, dim, unbiased, True))

    # Texture-based tests
    texture_test_suite = VkTestSuite(test_cases)
    texture_test_suite.layouts = [
        "utils::kChannelsPacked",
        "utils::kWidthPacked",
    ]
    texture_test_suite.storage_types = ["utils::kTexture3D"]
    texture_test_suite.atol = "1e-4"
    texture_test_suite.rtol = "1e-4"
    texture_test_suite.test_name_suffix = "texture"

    # Buffer-based tests
    buffer_test_suite = VkTestSuite(test_cases)
    buffer_test_suite.layouts = [
        "utils::kChannelsPacked",
        "utils::kWidthPacked",
    ]
    buffer_test_suite.storage_types = ["utils::kBuffer"]
    buffer_test_suite.atol = "1e-4"
    buffer_test_suite.rtol = "1e-4"
    buffer_test_suite.test_name_suffix = "buffer"

    return [texture_test_suite, buffer_test_suite]


@register_test_suite(
    [
        "aten.sqrt.default",
        "aten.rsqrt.default",
        "aten.exp.default",
        "aten.hardshrink.default",
        "aten.sin.default",
        "aten.neg.default",
        "aten.cos.default",
        "aten.hardswish.default",
        "aten.hardsigmoid.default",
        "aten.leaky_relu.default",
        "aten.round.default",
        "aten.tan.default",
        "aten.relu6.default",
    ]
)
def get_unary_ops_inputs():
    test_suite = VkTestSuite(
        [
            (M1,),
            (M1, M2),
            (S1, M1, M2),
            (S1, S2, S2, M2),
        ]
    )
    test_suite.storage_types = ["utils::kTexture3D", "utils::kBuffer"]
    test_suite.atol = "1e-4"
    test_suite.rtol = "1e-4"
    return test_suite


# separate test suite from unary_ops for learning purposes
@register_test_suite("aten.tan.default")
def get_tan_inputs():
    test_suite = VkTestSuite(
        [
            (M1,),
            (M1, M2),
            (S1, M1, M2),
            (S1, S2, S2, M2),
        ]
    )
    test_suite.storage_types = ["utils::kTexture3D", "utils::kBuffer"]
    test_suite.dtypes = ["at::kFloat", "at::kHalf"]
    return test_suite


@register_test_suite("aten._native_batch_norm_legit_no_training.default")
def get_native_batch_norm_inputs():
    Test = namedtuple(
        "VkSliceTest", ["self", "weight", "bias", "mean", "var", "momentum", "eps"]
    )

    test_cases = [
        Test(
            self=(1, 1, 2, 5),
            weight=(1,),
            bias=(1,),
            mean=(1,),
            var=(1,),
            momentum=0.0,
            eps=0.001,
        ),
        Test(
            self=(S2, 1, 2, 5),
            weight=(1,),
            bias=(1,),
            mean=(1,),
            var=(1,),
            momentum=0.0,
            eps=0.001,
        ),
        Test(
            self=(1, S2, 2, 5),
            weight=(S2,),
            bias=(S2,),
            mean=(S2,),
            var=(S2,),
            momentum=0.0,
            eps=0.001,
        ),
        Test(
            self=(9, S1, 2, 5),
            weight=(S1,),
            bias=(S1,),
            mean=(S1,),
            var=(S1,),
            momentum=0.0,
            eps=0.01,
        ),
        Test(
            self=(3, S1, 2, 5),
            weight=(S1,),
            bias=(S1,),
            mean=(S1,),
            var=(S1,),
            momentum=0.0,
            eps=0.001,
        ),
        Test(
            self=(3, S2, 2, 5),
            weight=(S2,),
            bias=(S2,),
            mean=(S2,),
            var=(S2,),
            momentum=0.0,
            eps=0.001,
        ),
        Test(
            self=(3, S2, 2, 5),
            weight=(S2,),
            bias=(S2,),
            mean=(S2,),
            var=(S2,),
            momentum=0.0,
            eps=0.000,
        ),
    ]

    test_suite = VkTestSuite(test_cases)
    test_suite.requires_prepack = True
    test_suite.prepacked_args = ["weight", "bias", "mean", "var"]

    return test_suite


@register_test_suite("aten.gelu.default")
def get_gelu_inputs():
    test_suite = VkTestSuite(
        [
            ((M1), "tanh"),
            ((M1, M2), "tanh"),
            ((S1, M1, M2), "tanh"),
            ((S1, S2, S2, M2), "tanh"),
        ]
    )
    return test_suite


@register_test_suite("aten.arange.start_step")
def get_arange_inputs():
    test_suite = VkTestSuite(
        [
            (1, 13),
            (1.0, 11),
            (-13, 3),
            (-11.0, 2),
            (3, 15, 3),
            (3, 23, 2),
            (3, 23.0, 4),
            (13, 1, -1),
            (-3, -13, -2),
            (13, -2.0, -4),
        ],
    )

    test_suite.layouts = [
        "utils::kChannelsPacked",
    ]
    return test_suite


@register_test_suite("aten.constant_pad_nd.default")
def get_constant_pad_nd_inputs():
    test_suite = VkTestSuite(
        [
            ([S1, S2], [1, 1], 24.0),
            ([M, M1, M2], [2, 2], 23.2),
            ([L, M, M1, M2], [3, 5], 12.2),
            ([S1, S2], [1, 1, 1, 1], 24.0),
            ([M, M1, M2], [2, 2, 2, 2], 23.2),
            ([L, M, M1, M2], [3, 5, 3, 5], 12.2),
            ([M, M1, M2], [1, 2, 3, 4, 5, 6], 23.2),
            ([L, M, M1, M2], [3, 3, 3, 3, 3, 3], 12.2),
        ]
    )
    return test_suite


@register_test_suite("aten.minimum.default")
def get_minimum_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, M2), (M2)),
            ((M1, M2), (M1, M2)),
            ((M1, M2, M), (M2, M)),
            ((M1, M1, S1, S2), (M1, M1, S1, S2)),
            ((S1, S1, S2, S), (S1, S2, S)),
            ((M1, S1, S2), (L, M1, S1, S2)),
            ((S1, S2), (L, M1, S1, S2)),
        ]
    )
    return test_suite


@register_test_suite("aten.squeeze_copy.dims")
def get_squeeze_copy_dim_inputs():
    test_suite = VkTestSuite(
        [
            ([S, S, S, 1], 3),
            ([S, 1, S, S], 1),
            ([S, 1, 1, S], [1, 2]),
            ([1, S, S, S], 0),
            ([S, S, S, S], 3),
            ([S, S, S, S], 2),
            ([S, S, S, S], 1),
            ([M, M1, 1], 2),
            ([M, 1, M1], 1),
            ([1, M1, M1], 0),
        ]
    )

    highdim_test_suite = VkTestSuite(
        [
            ([1, 2, 3, 4, 5, 1], 0),
            ([1, 2, 3, 4, 5, 1], 5),
            ([1, 2, 3, 4, 5, 1], [0, 5]),
            ([2, 1, 3, 1, 5, 6], 1),
            ([2, 1, 3, 1, 5, 6], 3),
            ([2, 1, 3, 1, 5, 6], [1, 3]),
            ([1, 1, 3, 4, 5, 6], [0, 1]),
            ([2, 3, 4, 1, 1, 6], [3, 4]),
        ]
    )
    highdim_test_suite.storage_types = [
        "utils::kBuffer",
    ]
    highdim_test_suite.test_name_suffix = "highdim"

    for suite in [test_suite, highdim_test_suite]:
        suite.layouts = [
            "utils::kWidthPacked",
            "utils::kChannelsPacked",
        ]

    return [test_suite, highdim_test_suite]


@register_test_suite("aten.flip.default")
def get_flip_inputs():
    Test = namedtuple("Flip", ["self", "dim"])
    Test.__new__.__defaults__ = (None, 0)

    test_cases = [
        Test(self=[9], dim=[0]),
        Test(self=[9, 9], dim=[0, 1]),
        Test(self=[9, 9, 9], dim=[0, 2]),
        Test(self=[9, 9, 9], dim=[0, 1, 2]),
        Test(self=[9, 9, 9, 9], dim=[0]),
        Test(self=[9, 9, 9, 9], dim=[0, 2, 3]),
        Test(self=[9, 9, 9, 9], dim=[1, 3]),
        Test(self=[9, 9, 9, 9], dim=[0, 1, 2, 3]),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])
    return test_suite


@register_test_suite("aten.expand_copy.default")
def get_expand_inputs():
    test_suite = VkTestSuite(
        [
            # Basic expansion cases
            ((1,), [5]),
            ((1, 1), [3, 4]),
            ((1, 3), [2, 3]),
            ((3, 1), [3, 4]),
            ((1, 1, 1), [2, 3, 4]),
            # Expand with same size (no-op)
            ((3, 4), [3, 4]),
            ((2, 3, 4), [2, 3, 4]),
            # Expand with additional dimensions
            ((3,), [2, 3]),
            ((3, 4), [2, 3, 4]),
            ((2, 3), [1, 2, 3]),
            # Mixed expansion cases
            ((1, 3, 1, 4), [2, 3, 5, 4]),
            ((1, 1, 3, 1), [2, 4, 3, 5]),
            # Larger tensor cases
            ((1, S1), [M, S1]),
            ((S2, 1), [S2, M1]),
            ((1, 1, S), [S1, S2, S]),
            ((1, S1, 1, S2), [M, S1, M1, S2]),
        ]
    )
    test_suite.storage_types = [
        "utils::kBuffer",
    ]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.dtypes = [
        "at::kFloat",
        "at::kHalf",
    ]
    test_suite.data_gen = "make_seq_tensor"
    return test_suite


@register_test_suite("aten.where.self")
def get_where_inputs():
    Test = namedtuple("Where", ["condition", "self", "other"])
    Test.__new__.__defaults__ = (None, None, None)

    test_cases = [
        Test(condition=[11], self=[11], other=[11]),
        Test(condition=[10, 9], self=[10, 9], other=[10, 9]),
        Test(condition=[10, 5, 3], self=[10, 5, 3], other=[10, 5, 3]),
        Test(condition=[2, 10, 5, 3], self=[2, 10, 5, 3], other=[2, 10, 5, 3]),
    ]

    test_suite = VkTestSuite([tuple(tc) for tc in test_cases])
    test_suite.arg_dtype["condition"] = "at::kBool"
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kHeightPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.storage_types = ["utils::kTexture3D", "utils::kBuffer"]
    test_suite.atol = "1e-4"
    test_suite.rtol = "1e-4"
    return test_suite


@register_test_suite("aten.pow.Tensor_Scalar")
def get_pow_tensor_scalar_inputs():
    test_suite = VkTestSuite(
        [
            ((M1,), 2.0),
            ((M2, M1), 2.0),
            ((S1, M1, M2), 0.5),
            ((S1, S2, S2, M2), 2.5),
            ((S, S1, S2), -1.0),
            ((M1, M2), 4.0),
            ((S1, S2), 1.5),
        ]
    )
    test_suite.storage_types = [
        "utils::kBuffer",
        "utils::kTexture3D",
    ]
    test_suite.layouts = [
        "utils::kWidthPacked",
        "utils::kChannelsPacked",
    ]
    test_suite.dtypes = ["at::kFloat"]
    return test_suite
