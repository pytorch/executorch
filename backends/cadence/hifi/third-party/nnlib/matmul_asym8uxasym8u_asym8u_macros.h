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
This file modifies the macros defined in
nnlib-hifi4/xa_nnlib/algo/common/include/xa_nnlib_common_macros.h
to adjust the accumulated value for per-channel quantized scheme.
1. The ADJUST_ACC_BATCH_ASYM8b in nnlib multiplies the accumulated value in
matmul with the requantized scale. The requantized scale is an fp32 value
(in_scale*weight_scale/out_scale). This fp32 requanzied_scale is repsented as a
fixed-point computation with an int32 out_multiplier, and an in32 out_shift.
The weight_scale can be an array for per-channel quantized weight. So we allow
the left_shift/right_shift and out_multiplier to be arrays in
ADJUST_ACC_BATCH_ASYM8b.
2. The new macros SETUP_SHIFT and  UNROLL_ROW_SETUP_SHIFT are not present in
nnlib. We add these to allow finding the correct out_shift for each channel
(i.e., unrolled row of p_mat1).
*/

#include "xa_nnlib_matmul_unroll_macros.h"

#define UNROLL_ROW_SETUP_SHIFT(idx_row)                               \
  ae_int32x2 _ae_int32x2_ch_idx_##idx_row = 0;                        \
  AE_MOVT32X2(                                                        \
      _ae_int32x2_ch_idx_##idx_row,                                   \
      AE_MOVDA32X2(m_itr + idx_row, m_itr + idx_row),                 \
      per_channel_quantized);                                         \
  int _ch_idx_##idx_row = AE_MOVAD32_L(_ae_int32x2_ch_idx_##idx_row); \
  left_shift[idx_row] = AE_MAX32(0, out_shift[_ch_idx_##idx_row]);    \
  right_shift[idx_row] = AE_MAX32(0, -out_shift[_ch_idx_##idx_row]);

#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec)                      \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */ \
  ae_int32x2 _ae_int32x2_acc_##idx_row##_##idx_vec;                    \
  MPY_BY_QUANT_MULT_X2_OUT32(                                          \
      _ae_int32x2_acc_##idx_row##_##idx_vec,                           \
      AE_MOVINT32X2_FROMINT64(_ae_int64_acc_##idx_row##_##idx_vec),    \
      out_multiplier[_ch_idx_##idx_row],                               \
      left_shift[idx_row],                                             \
      right_shift[idx_row]);                                           \
  /* Add output zero point */                                          \
  (_ae_int32x2_acc_##idx_row##_##idx_vec) = AE_ADD32S(                 \
      _ae_int32x2_acc_##idx_row##_##idx_vec, AE_MOVDA32(out_zero_bias));

#if (ROW_UNROLL == 2)

#define SETUP_SHIFT         \
  UNROLL_ROW_SETUP_SHIFT(0) \
  UNROLL_ROW_SETUP_SHIFT(1)

#elif (ROW_UNROLL == 4)

#define SETUP_SHIFT         \
  UNROLL_ROW_SETUP_SHIFT(0) \
  UNROLL_ROW_SETUP_SHIFT(1) \
  UNROLL_ROW_SETUP_SHIFT(2) \
  UNROLL_ROW_SETUP_SHIFT(3)

#endif /* (ROW_UNROLL == 4)*/
