# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import and register Arm TOSA operator visitors.

Importing this package loads all visitor modules so their classes can be
registered via decorators and discovered at runtime.

"""


from . import (  # noqa
    node_visitor,
    op_abs,
    op_add,
    op_amax,
    op_amin,
    op_any,
    op_avg_pool2d,
    op_bitwise_not,
    op_cat,
    op_ceil,
    op_clamp,
    op_cond_if,
    op_constant_pad_nd,
    op_cos,
    op_eq,
    op_erf,
    op_exp,
    op_floor,
    op_ge,
    op_gt,
    op_index_select,
    op_index_tensor,
    op_le,
    op_log,
    op_logical_not,
    op_lt,
    op_max_pool2d,
    op_maximum,
    op_minimum,
    op_mul,
    op_neg,
    op_permute,
    op_pow,
    op_reciprocal,
    op_repeat,
    op_rshift_tensor,
    op_rsqrt,
    op_sigmoid,
    op_sin,
    op_slice,
    op_sub,
    op_sum,
    op_tanh,
    op_to_dim_order_copy,
    op_tosa_conv2d,
    op_tosa_depthwise_conv2d,
    op_tosa_matmul,
    op_tosa_rescale,
    op_tosa_resize,
    op_tosa_table,
    op_tosa_transpose,
    op_view,
    op_where,
    ops_binary,
    ops_identity,
)
