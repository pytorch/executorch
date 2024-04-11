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

#pragma once

/*
nnlib forces ROW_UNROLL to be 4 when the input and weight are both aligned to 8b
boundary. Through experimentation, we observe that the
xa_nn_matmul_asym8xasym8_asym8 kernel in nnlib performs better when the weight
matrix is uniformly unrolled by a factor of 2 instead of 4 for 8b aligned case.
We add a case for ROW_UNROLL=2 and VEC_UNROLL=2 here. This code is similar to
the ROW_UNROLL=4 and VEC_UNROLL=2 code in
nnlib-hifi4/xa_nnlib/algo/common/include/xa_nnlib_common_macros.h.
*/

// Unrolling macros that unroll both matrices by a factor of 2.
#if (ROW_UNROLL == 2 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH UNROLL_SETUP_VEC_BATCH(0) UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_ACC_BATCH         \
  UNROLL_ROW_SETUP_ACC_BATCH(0) \
  UNROLL_ROW_SETUP_ACC_BATCH(1)

#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_SETUP_ACC_BATCH(idx_row, 0)        \
  UNROLL_SETUP_ACC_BATCH(idx_row, 1)

#define SETUP_ACC_BATCH_TAIL   \
  UNROLL_SETUP_ACC_BATCH(0, 0) \
  UNROLL_SETUP_ACC_BATCH(1, 0)

#define LOAD_VEC_BATCH UNROLL_LOAD_VEC_BATCH(0) UNROLL_LOAD_VEC_BATCH(1)

#define LOAD_MAT1         \
  UNROLL_LOAD_ROW_MAT1(0) \
  UNROLL_LOAD_ROW_MAT1(1)

#define KERNEL_MAT1_VEC_BATCH         \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0) \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1)

#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 0)        \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 1)

#define KERNEL_MAT1_VEC_BATCH_TAIL   \
  UNROLL_KERNEL_MAT1_VEC_BATCH(0, 0) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(1, 0)

#define ADD_BIAS_ACC_BATCH   \
  UNROLL_ROW_ADD_BIAS_ACC(0) \
  UNROLL_ROW_ADD_BIAS_ACC(1)

#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row) \
  UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 0) UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 1)

#define ADD_BIAS_ACC_BATCH_TAIL             \
  LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(0, 0) \
      LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(1, 0)

#define STORE_ACC_BATCH   \
  UNROLL_ROW_STORE_ACC(0) \
  UNROLL_ROW_STORE_ACC(1)

#define STORE_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_STORE_ACC_BATCH(idx_row, 0) UNROLL_STORE_ACC_BATCH(idx_row, 1)

#define STORE_ACC_BATCH_TAIL   \
  UNROLL_STORE_ACC_BATCH(0, 0) \
  UNROLL_STORE_ACC_BATCH(1, 0)

#define ADJUST_ACC_BATCH_TAIL   \
  UNROLL_ADJUST_ACC_BATCH(0, 0) \
  UNROLL_ADJUST_ACC_BATCH(1, 0)

#define ADJUST_ACC_BATCH   \
  UNROLL_ROW_ADJUST_ACC(0) \
  UNROLL_ROW_ADJUST_ACC(1)

#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_ADJUST_ACC_BATCH(idx_row, 0) UNROLL_ADJUST_ACC_BATCH(idx_row, 1)

#endif /* (ROW_UNROLL == 2 && VEC_UNROLL == 2)*/
