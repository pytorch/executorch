/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath> 

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using torch::executor::Error;

namespace impl {
namespace HiFi {
namespace native {

namespace {

ScalarType get_compute_type(ScalarType a_type, ScalarType b_type) {
  if (isFloatingType(a_type) && isFloatingType(b_type)) {
    return promoteTypes(a_type, b_type);
  } else if (isFloatingType(a_type)) {
    return a_type;
  } else if (isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor&
div_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      !isComplexType(a_type) && !isQIntType(a_type) && !isBitsType(a_type),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      !isComplexType(b_type) && !isQIntType(b_type) && !isBitsType(b_type),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);
  
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */
  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = 1;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;
  
  if ((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
    optimized = 0;
  
  if ((a_dim == 0) || (b_dim == 0) )
    optimized = 0;

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = 0;
  
  if (optimized) {
    float* a_data = a.mutable_data_ptr<float>();
    float* b_data = b.mutable_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();
    
    if (broadcast == 1) {
      
      int out_shape[kNnlibMaxDim];
      int inp1_shape[kNnlibMaxDim];
      int inp2_shape[kNnlibMaxDim];
      
      for (int i = 0; i < kNnlibMaxDim; i++)
      {
        out_shape[i] = 1;
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
      }
        
      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();
      for (int i = 0; i < out.dim(); i++)
        out_shape[i+off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i+off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i+off_b] = b.size(i);
      
      xa_nn_elm_div_broadcast_4D_f32xf32_f32(
        out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape);
    }
    else
    {
      xa_nn_elm_div_f32xf32_f32(out_data, a_data, b_data, out.numel());
    }

    return out;
  }

  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div.out", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div.out", CTYPE_B, [&]() {
      ET_SWITCH_FLOAT_TYPES(common_type, ctx, "div.out", CTYPE_IN, [&]() {
        ET_SWITCH_FLOAT_TYPES(out_type, ctx, "div.out", CTYPE_OUT, [&]() {
          torch::executor::
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
              [](const CTYPE_A val_a, const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = a_casted / b_casted;
  
                return static_cast<CTYPE_OUT>(value);
              },
              a,
              b,
              out);
        });
      });
    });
  });

  return out;
}

Tensor& div_out_mode(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    exec_aten::optional<exec_aten::string_view> mode,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);

  // Allow casting float -> integral here
  // non-bool -> bool is still disallowed
  ET_KERNEL_CHECK(
      ctx,
      !(common_type != ScalarType::Bool && out_type == ScalarType::Bool),
      InvalidArgument,
      out);
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */
  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = 1;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;
  
  if ((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
    optimized = 0;
  
  if ((a_dim == 0) || (b_dim == 0))
    optimized = 0;

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = 0;
  int mode_val = -1;
  if (mode.has_value() && mode.value() == "trunc") 
    mode_val = 0;
  else if (mode.has_value() && mode.value() == "floor")
    mode_val = 1;
  else
    optimized = 0;
      
  if (optimized) {
    float* a_data = a.mutable_data_ptr<float>();
    float* b_data = b.mutable_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    if (broadcast) {
      int out_shape[kNnlibMaxDim];
      int inp1_shape[kNnlibMaxDim];
      int inp2_shape[kNnlibMaxDim];
      
      for (int i = 0; i < kNnlibMaxDim; i++) {
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
        out_shape[i] = 1;
      }

      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();

      for (int i = 0; i < out.dim(); i++)
        out_shape[i+off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i+off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i+off_b] = b.size(i);
      
      xa_nn_elm_div_mode_broadcast_4D_f32xf32_f32(
        out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape, mode_val);
    }
    else
    {
      xa_nn_elm_div_mode_f32xf32_f32(
        out_data, a_data, b_data, out.numel(), mode_val);
    }
    
    return out;
  }

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div.out_mode", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div.out_mode", CTYPE_B, [&]() {
      ET_SWITCH_FLOAT_TYPES(common_type, ctx, "div.out_mode", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES(out_type, ctx, "div.out_mode", CTYPE_OUT, [&]() {
          torch::executor::
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
              [mode](const CTYPE_A val_a, const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = a_casted / b_casted;
                if (mode.has_value() && mode.value() == "trunc") {
                  value = std::trunc(value);
                } else if (mode.has_value() && mode.value() == "floor") {
                  value = std::floor(value);
                }
                return static_cast<CTYPE_OUT>(value);
              },
              a,
              b,
              out);
        });
      });
    });
  });

  return out;
}


} // namespace native
} // namespace HiFi
} // namespace impl
