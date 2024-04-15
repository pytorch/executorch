# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
    return VkTestSuite(
        [
            ((M1, M2), (M1, M2)),
            ((M1, M2), (M1, 1), 2.0),
            ((M1, M2), (1, M2)),
            ((S, S1, S2), (S, S1, S2)),
            ((S, S1, S2), (S, S1, 1), 2.0),
            ((S, S1, S2), (S, 1, S2), 2.0),
        ]
    )


def get_mm_inputs():
    test_suite = VkTestSuite(
        [
            ((M1, L), (L, M2)),
            ((S1, S2), (S2, M)),
        ],
    )
    test_suite.prepacked_args = ["mat2"]
    return test_suite


def get_pool2d_inputs():
    test_suite = VkTestSuite(
        [
            ((S, M1, M2), [2, 2], [1, 1], [0, 0], [1, 1]),
        ]
    )
    test_suite.supports["layouts"] = ["api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED"]
    return test_suite


def get_conv2d_inputs():
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
        ]
    )
    test_suite.supports["layouts"] = ["api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED"]
    return test_suite


def get_native_layer_norm_inputs():
    test_suite = VkTestSuite(
        [
            ((S1, S2), [S2], (S2), (S2), 0.001),
            ((M, M1, M2), [M2], (M2), (M2), 0.001),
            ((L, XL, M1, M2), [M2], (M2), (M2), 0.001),
        ]
    )
    test_suite.supports["layouts"] = ["api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED"]
    return test_suite


def get_full_inputs():
    test_suite = VkTestSuite(
        [
            ([S1, S2], 42.0),
            ([M, M1, M2], 3.14),
            ([L, M, M1, M2], 2.72),
        ]
    )
    test_suite.supports["layouts"] = ["api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED"]
    return test_suite


test_suites = {
    "aten.add.Tensor": get_binary_elementwise_inputs(),
    "aten.sub.Tensor": get_binary_elementwise_inputs(),
    "aten.div.Tensor": get_binary_elementwise_inputs(),
    "aten.mul.Tensor": get_binary_elementwise_inputs(),
    "aten.mm.default": get_mm_inputs(),
    "aten.max_pool2d_with_indices.default": get_pool2d_inputs(),
    "aten.convolution.default": get_conv2d_inputs(),
    "aten.native_layer_norm.default": get_native_layer_norm_inputs(),
    "aten.full.default": get_full_inputs(),
}

prepacked_args = {"aten.mm.default": {"mat2"}}

support_exceptions = {
    "aten.max_pool2d_with_indices.default": {
        "layouts": ["api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED"]
    },
}
