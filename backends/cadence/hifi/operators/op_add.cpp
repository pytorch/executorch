/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

#define NNLIB_MAX_DIM 4  /* Add fallback if broadcast and dim > 4 */

Tensor& add_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_CHECK_MSG(a_type == ScalarType::Float, "Input tensor not a float.\n");
  ET_CHECK_MSG(b_type == ScalarType::Float, "Input tensor not a float.\n");
  ET_CHECK_MSG(out_type == ScalarType::Float, "Output tensor not a float.\n");

  ET_CHECK(canCast(common_type, out_type));

  using CTYPE_A = float;
  using CTYPE_B = float;
  using CTYPE_IN = float;
  using CTYPE_OUT = float;
  CTYPE_IN alpha_val;
  ET_EXTRACT_SCALAR(alpha, alpha_val);

  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  int fall_back = 0;
  /*find broadcast*/
  const int a_is_broadcasted = !out.sizes().equals(a.sizes());
  const int b_is_broadcasted = !out.sizes().equals(b.sizes());
  const int broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;
  
  if( (out_type != ScalarType::Float) || (alpha_val != 1.0))
    fall_back = 1;
  
  if( (a_dim == 0) || (b_dim == 0) )
    fall_back = 1;

  if((broadcast == 1) && (max_dim > NNLIB_MAX_DIM))
    fall_back = 1;


  if (!fall_back)
  {
      const float* const a_data = a.const_data_ptr<float>();
      const float* const b_data = b.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if(broadcast == 1)
      {
         int out_shape[NNLIB_MAX_DIM];
         int inp1_shape[NNLIB_MAX_DIM];
         int inp2_shape[NNLIB_MAX_DIM];
         
         for(int i = 0; i < NNLIB_MAX_DIM; i++)
         {
            out_shape[i] = 1;
            inp1_shape[i] = 1;
            inp2_shape[i] = 1;
         }
                  
         int off_o = NNLIB_MAX_DIM - out.dim();
         int off_a = NNLIB_MAX_DIM - a.dim();
         int off_b = NNLIB_MAX_DIM - b.dim();
         
         for(int i = 0; i < out.dim(); i++)
             out_shape[i+off_o] = out.size(i);
         for(int i = 0; i < a.dim(); i++)
             inp1_shape[i+off_a] = a.size(i);
         for(int i = 0; i < b.dim(); i++)
             inp2_shape[i+off_b] = b.size(i);
         
         xa_nn_elm_add_broadcast_4D_f32xf32_f32(out_data, out_shape, a_data, inp1_shape,
                                                b_data, inp2_shape);
      }                      
      else
        xa_nn_elm_add_f32xf32_f32(out_data, a_data, b_data, out.numel());
      
  }
  else
  {
      apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
      [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
        CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
        CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
        CTYPE_IN value = a_casted + alpha_val * b_casted;

        return static_cast<CTYPE_OUT>(value);
      },
      a,
      b,
      out);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
