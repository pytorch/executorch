/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/

#include "xa_nnlib_common.h"
#include "xa_type_def.h"

#ifdef ROW_UNROLL
#undef ROW_UNROLL
#endif
#define ROW_UNROLL 4

#include "xa_nnlib_common_macros.h"

/* Include the asym8uxasym8u_asym8u macros */
#include "matmul_asym8uxasym8u_asym8u_macros.h"

/*----------------------------Main function---------------------------------*/

namespace impl {
namespace HiFi {
namespace kernels {
/*
The following function is copied from xa_nn_matmul_asym8xasym8_asym8 in
xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matmul_asym8xasym8.c.

xa_nn_matmul_asym8xasym8_asym8 multiplies two quint8 matrices, and requantizes
the result to produce a quint8 output. However, it has two limitations:
1. It only works for per-tensor quantized weight.
2. It forces the weight rows to be unrolled by 4 when both input and weight are
aligned to 8b boundary.

We modify xa_nn_matmul_asym8xasym8_asym8 to allow per-channel quantized weights
as well. To do so, we make the following two changes:
1. The out_multiplier and out_shift now become arrays instead of scalars. Apart
from the function arg, we add a new macro (UNROLL_ROW_SETUP_SHIFT) which
computes the right out_shift for each channel (i.e., unrolled row of weight),
and stores it in the appropriate index of left_shift[ROW_UNROLL] and
right_shift[ROW_UNROLL].
2. We modify the ADJUST_ACC_BATCH_ASYM8b macro so that it it picks up the right
out_multiplier and out_shift for the accumulation corresponding to each channel
(i.e., unrolled row of weight).

Through experimentation, we observe that the kernel performs better when the
weight matrix is uniformly unrolled by a factor of 2 instead of 4 for 8b aligned
case. We add a case for ROW_UNROLL=2 and VEC_UNROLL=2 in
xa_nnlib_matmul_unroll_macros.h. This code is similar to the ROW_UNROLL=4 and
VEC_UNROLL=2 code in
nnlib-hifi4/xa_nnlib/algo/common/include/xa_nnlib_common_macros.h.

General information about the code:
The HiFi4 xa_nn_matmul_asym8xasym8_asym8 kernel writes the code using macros,
which are expanded to HiFi4 intrinsics in
nnlib-hifi4/xa_nnlib/algo/common/include/xa_nnlib_common_macros.h.
The code caters to two specific cases:
1. When the two input matrices (p_mat1 and p_vec1) are aligned to 8-byte
boundary, we do not need unaligned loads. In that case, 'chk_align' is true, and
the code unrolls p_mat1 by 4 and p_vec1 by 2.
2. If chk_align is false, then the code unrolls both p_mat1 and p_vec1 by a
factor of 2. The code will use macros that expand to unaligned loads via
register priming (e.g., LOAD_VEC_BATCH_ASYM8b_UNALIGNED)
3. If either p_mat1 or p_vec1 are nullptr, the code returns -1.

The choice of unrolling factors in the NNLib kernel is not controlled by the
user: it sets ROW_UNROLL to 4 by default. This choice is not goverened by any
heuristics. The performance degradation due to unaligned loads/stores also is
not clear to warrant two branches in the code (if/else branching on chk_align).

Future modifications: In future, if Tensilica provides a new version of the
xa_nn_matmul_asym8xasym8_asym8 kernel, the changes to this file would be
minimal: we just copy the entire function here, change the args for
out_multiplier and out_shift, and add SETUP_SHIFT/UNROLL_ROW_SETUP_SHIFT macro
to get the right out_shift for each unrolled row.
*/

WORD32 matmul_asym8uxasym8u_asym8u(
    UWORD8* __restrict__ p_out,
    const UWORD8* __restrict__ p_mat1,
    const UWORD8* __restrict__ p_vec1,
    const WORD32* __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,
    WORD32 mat1_zero_bias,
    WORD32 vec1_zero_bias,
    const WORD32* __restrict__ out_multiplier,
    const WORD32* __restrict__ out_shift,
    WORD32 out_zero_bias,
    bool per_channel_quantized) {
  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift[ROW_UNROLL] = {0}, right_shift[ROW_UNROLL] = {0};

#define UNROLL_ROW_SETUP_ACC_BATCH SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_ACC_BATCH SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_MAT1 SETUP_MAT1_ASYM8b
#define UNROLL_SETUP_VEC_BATCH SETUP_VEC_OFFSET_BATCH_ASYM8b
#define SETUP_BIAS SETUP_BIAS_ASYM8b
#define UNROLL_LOAD_VEC_BATCH LOAD_VEC_BATCH_ASYM8b
#define UNROLL_LOAD_ROW_MAT1 LOAD_ROW_MAT1_ASYM8b
#define LOAD_BIAS LOAD_BIAS_ASYM8b_MATMUL
#define UNROLL_ROW_KERNEL_MAT1_VEC_BATCH KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b
#define UNROLL_KERNEL_MAT1_VEC_BATCH KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b
#define UNROLL_ROW_ADD_BIAS_ACC \
  ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
#define UNROLL_ADD_BIAS_ACC_BATCH \
  ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
#define UNROLL_ROW_ADJUST_ACC ADJUST_ACC_BATCH_ROW_ASYM8b
#define UNROLL_ADJUST_ACC_BATCH ADJUST_ACC_BATCH_ASYM8b
#define UNROLL_ROW_STORE_ACC STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b
#define UNROLL_STORE_ACC_BATCH \
  STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b

  int chk_align = 0;
  CHK_MATMUL_ALIGN(
      chk_align, p_mat1, 1, p_vec1, 1, cols1, row_stride1, vec_offset, 4);

  if (chk_align) {
    for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL - 1));
         vec_itr += VEC_UNROLL) {
      SETUP_BIAS;
      for (m_itr = 0; m_itr < (rows & ~(ROW_UNROLL - 1)); m_itr += ROW_UNROLL) {
        SETUP_SHIFT;
        SETUP_ACC_BATCH;
        SETUP_VEC_BATCH;
        SETUP_MAT1;

        for (c_itr = 0; c_itr < (cols1 >> 2); c_itr++) {
          LOAD_VEC_BATCH;
          LOAD_MAT1;
          KERNEL_MAT1_VEC_BATCH;
        }

        ADD_BIAS_ACC_BATCH;
        ADJUST_ACC_BATCH;
        STORE_ACC_BATCH;
      }

#pragma no_unroll
      for (; m_itr < rows; m_itr++) {
        UNROLL_ROW_SETUP_SHIFT(0);
        UNROLL_ROW_SETUP_ACC_BATCH(0);
        SETUP_VEC_BATCH;
        UNROLL_SETUP_MAT1(0);

        for (c_itr = 0; c_itr < (cols1 >> 2); c_itr++) {
          LOAD_VEC_BATCH;
          UNROLL_LOAD_ROW_MAT1(0);
          UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0);
        }

        UNROLL_ROW_ADD_BIAS_ACC(0);
        UNROLL_ROW_ADJUST_ACC(0);
        UNROLL_ROW_STORE_ACC(0);
      }
    }
    /* Tail loop for vec unroll */
    for (; vec_itr < vec_count; vec_itr++) {
      SETUP_BIAS;
      for (m_itr = 0; m_itr < (rows & ~(ROW_UNROLL - 1)); m_itr += ROW_UNROLL) {
        SETUP_SHIFT;
        SETUP_ACC_BATCH_TAIL;
        UNROLL_SETUP_VEC_BATCH(0);
        SETUP_MAT1;

        for (c_itr = 0; c_itr < (cols1 >> 2); c_itr++) {
          UNROLL_LOAD_VEC_BATCH(0);
          LOAD_MAT1;
          KERNEL_MAT1_VEC_BATCH_TAIL;
        }

        ADD_BIAS_ACC_BATCH_TAIL;
        ADJUST_ACC_BATCH_TAIL;
        STORE_ACC_BATCH_TAIL;
      }

#pragma no_unroll
      for (; m_itr < rows; m_itr++) {
        UNROLL_ROW_SETUP_SHIFT(0);
        UNROLL_SETUP_ACC_BATCH(0, 0);
        UNROLL_SETUP_VEC_BATCH(0);
        UNROLL_SETUP_MAT1(0);

        for (c_itr = 0; c_itr < (cols1 >> 2); c_itr++) {
          UNROLL_LOAD_VEC_BATCH(0);
          UNROLL_LOAD_ROW_MAT1(0);
          UNROLL_KERNEL_MAT1_VEC_BATCH(0, 0);
        }

        LOAD_BIAS;
        UNROLL_ADD_BIAS_ACC_BATCH(0, 0);
        UNROLL_ADJUST_ACC_BATCH(0, 0);
        UNROLL_STORE_ACC_BATCH(0, 0);
      }
    }

/* Undefining the defined macro to make them available for reuse */
#undef UNROLL_ROW_SETUP_ACC_BATCH
#undef UNROLL_SETUP_ACC_BATCH
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_VEC_BATCH
#undef SETUP_BIAS
#undef SETUP_SHIFT
#undef UNROLL_LOAD_VEC_BATCH
#undef UNROLL_LOAD_ROW_MAT1
#undef LOAD_BIAS
#undef UNROLL_ROW_KERNEL_MAT1_VEC_BATCH
#undef UNROLL_KERNEL_MAT1_VEC_BATCH
#undef UNROLL_ROW_ADD_BIAS_ACC
#undef UNROLL_ADD_BIAS_ACC_BATCH
#undef UNROLL_ROW_ADJUST_ACC
#undef UNROLL_ADJUST_ACC_BATCH
#undef UNROLL_ROW_STORE_ACC
#undef UNROLL_STORE_ACC_BATCH
#undef VEC_UNROLL
#undef ROW_UNROLL
  } else if (p_mat1 && p_vec1) {
#define ROW_UNROLL 2
#define VEC_UNROLL 2
#define UNROLL_ROW_SETUP_ACC_BATCH SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b
#define UNROLL_SETUP_ACC_BATCH SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b
#define SETUP_BIAS SETUP_BIAS_ASYM8b
#define LOAD_BIAS LOAD_BIAS_ASYM8b_MATMUL
#define UNROLL_ROW_ADD_BIAS_ACC \
  ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
#define UNROLL_ADD_BIAS_ACC_BATCH \
  ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL
#define UNROLL_ROW_ADJUST_ACC ADJUST_ACC_BATCH_ROW_ASYM8b
#define UNROLL_ADJUST_ACC_BATCH ADJUST_ACC_BATCH_ASYM8b
    for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL - 1));
         vec_itr += VEC_UNROLL) {
      SETUP_BIAS;
      for (m_itr = 0; m_itr < (rows & ~(ROW_UNROLL - 1)); m_itr += ROW_UNROLL) {
        UNROLL_ROW_SETUP_SHIFT(0);
        UNROLL_ROW_SETUP_SHIFT(1);
        UNROLL_SETUP_ACC_BATCH(0, 0);
        UNROLL_SETUP_ACC_BATCH(0, 1);
        UNROLL_SETUP_ACC_BATCH(1, 0);
        UNROLL_SETUP_ACC_BATCH(1, 1);
        SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
        SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(1);
        SETUP_MAT1_ASYM8b_UNALIGNED(0);
        SETUP_MAT1_ASYM8b_UNALIGNED(1);

        int cols1_count = cols1 - cols1 % 4;
        for (c_itr = 0; c_itr < (cols1_count >> 2); c_itr++) {
          LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
          LOAD_VEC_BATCH_ASYM8b_UNALIGNED(1);
          LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
          LOAD_ROW_MAT1_ASYM8b_UNALIGNED(1);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 1);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1, 1);
        }
#pragma no_unroll
        for (c_itr = cols1_count; c_itr < cols1; c_itr++) {
          LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
          LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(1);
          LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
          LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(1);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 1);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1, 1);
        }

        ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(0);
        ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(1);
        ADJUST_ACC_BATCH_ROW_ASYM8b(0);
        ADJUST_ACC_BATCH_ROW_ASYM8b(1);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 0);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1, 0);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 1);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1, 1);
      }
      // Remaining row
      for (; m_itr < rows; m_itr++) {
        UNROLL_ROW_SETUP_SHIFT(0);
        UNROLL_SETUP_ACC_BATCH(0, 0);
        UNROLL_SETUP_ACC_BATCH(0, 1);
        SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
        SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(1);
        SETUP_MAT1_ASYM8b_UNALIGNED(0);
        int cols1_count = cols1 - cols1 % 4;

        for (c_itr = 0; c_itr < (cols1_count >> 2); c_itr++) {
          LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
          LOAD_VEC_BATCH_ASYM8b_UNALIGNED(1);
          LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 1);
        }
#pragma no_unroll
        for (c_itr = cols1_count; c_itr < cols1; c_itr++) {
          LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
          LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(1);
          LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 0);
          KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 1);
        }
        ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(0);
        ADJUST_ACC_BATCH_ROW_ASYM8b(0);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 0);
        STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 1);
      }
    }
    {
      /* Tail loop for vec unroll */
      for (; vec_itr < vec_count; vec_itr++) {
        SETUP_BIAS;
        for (m_itr = 0; m_itr < (rows & ~(ROW_UNROLL - 1));
             m_itr += ROW_UNROLL) {
          UNROLL_ROW_SETUP_SHIFT(0);
          UNROLL_ROW_SETUP_SHIFT(1);
          UNROLL_SETUP_ACC_BATCH(0, 0);
          UNROLL_SETUP_ACC_BATCH(1, 0);
          SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
          SETUP_MAT1_ASYM8b_UNALIGNED(0);
          SETUP_MAT1_ASYM8b_UNALIGNED(1);
          int cols1_count = cols1 - cols1 % 4;

          for (c_itr = 0; c_itr < (cols1_count >> 2); c_itr++) {
            LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_UNALIGNED(1);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 0);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(1, 0);
          }
#pragma no_unroll
          for (c_itr = cols1_count; c_itr < cols1; c_itr++) {
            LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(1);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 0);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(1, 0);
          }

          LOAD_BIAS;
          UNROLL_ADD_BIAS_ACC_BATCH(0, 0);
          UNROLL_ADJUST_ACC_BATCH(0, 0);
          LOAD_BIAS;
          UNROLL_ADD_BIAS_ACC_BATCH(1, 0);
          UNROLL_ADJUST_ACC_BATCH(1, 0);

          STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 0);
          STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(1, 0);
        }

        for (; m_itr < rows; m_itr++) {
          UNROLL_ROW_SETUP_SHIFT(0);
          UNROLL_SETUP_ACC_BATCH(0, 0);
          SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED(0);
          SETUP_MAT1_ASYM8b_UNALIGNED(0);
          int cols1_count = cols1 - cols1 % 4;

          for (c_itr = 0; c_itr < (cols1_count >> 2); c_itr++) {
            LOAD_VEC_BATCH_ASYM8b_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_UNALIGNED(0);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(0, 0);
          }
#pragma no_unroll
          for (c_itr = cols1_count; c_itr < cols1; c_itr++) {
            LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(0);
            LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(0);
            KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED(0, 0);
          }

          LOAD_BIAS;
          UNROLL_ADD_BIAS_ACC_BATCH(0, 0);
          UNROLL_ADJUST_ACC_BATCH(0, 0);
          STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(0, 0);
        }
      }
    }
  } else {
    return -1;
  }

#undef UNROLL_ROW_SETUP_ACC_BATCH
#undef UNROLL_SETUP_ACC_BATCH
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_VEC_BATCH
#undef SETUP_BIAS
#undef SETUP_SHIFT
#undef UNROLL_LOAD_VEC_BATCH
#undef UNROLL_LOAD_ROW_MAT1
#undef LOAD_BIAS
#undef UNROLL_ROW_KERNEL_MAT1_VEC_BATCH
#undef UNROLL_KERNEL_MAT1_VEC_BATCH
#undef UNROLL_ROW_ADD_BIAS_ACC
#undef UNROLL_ADD_BIAS_ACC_BATCH
#undef UNROLL_ROW_ADJUST_ACC
#undef UNROLL_ADJUST_ACC_BATCH
#undef UNROLL_ROW_STORE_ACC
#undef UNROLL_STORE_ACC_BATCH
#undef VEC_UNROLL
#undef ROW_UNROLL

  return 0;
}

}; // namespace kernels
}; // namespace HiFi
}; // namespace impl
