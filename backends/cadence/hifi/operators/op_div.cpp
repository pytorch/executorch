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
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>

using executorch::aten::RuntimeContext;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

namespace {

ScalarType get_compute_type(ScalarType a_type, ScalarType b_type) {
  if (executorch::runtime::isFloatingType(a_type) &&
      executorch::runtime::isFloatingType(b_type)) {
    return executorch::runtime::promoteTypes(a_type, b_type);
  } else if (executorch::runtime::isFloatingType(a_type)) {
    return a_type;
  } else if (executorch::runtime::isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor&
div_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      !executorch::runtime::isComplexType(a_type) &&
          !executorch::runtime::isQIntType(a_type) &&
          !executorch::runtime::isBitsType(a_type),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      !executorch::runtime::isComplexType(b_type) &&
          !executorch::runtime::isQIntType(b_type) &&
          !executorch::runtime::isBitsType(b_type),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, executorch::runtime::tensor_is_real_type(out), InvalidArgument, out);

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

  bool float_types =
      (a_type == ScalarType::Float) && (b_type == ScalarType::Float);

  if ((a_dim == 0) && float_types) {
    for (int i = 0; i < b.numel(); i++)
      out.mutable_data_ptr<float>()[i] =
          a.const_data_ptr<float>()[0] / b.const_data_ptr<float>()[i];
    return out;
  }
  if ((b_dim == 0) && float_types) {
    for (int i = 0; i < a.numel(); i++)
      out.mutable_data_ptr<float>()[i] =
          a.const_data_ptr<float>()[i] / b.const_data_ptr<float>()[0];
    return out;
  }

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

      for (int i = 0; i < kNnlibMaxDim; i++) {
        out_shape[i] = 1;
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
      }

      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();
      for (int i = 0; i < out.dim(); i++)
        out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);

      xa_nn_elm_div_broadcast_4D_f32xf32_f32(
          out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape);
    } else {
      xa_nn_elm_div_f32xf32_f32(out_data, a_data, b_data, out.numel());
    }

    return out;
  }

  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::canCast(common_type, out_type),
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    torch::executor::native::utils::
        apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
            [](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
              return val_a / val_b;
            },
            ctx,
            a,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            b,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            out,
            torch::executor::native::utils::SupportedTensorDtypes::FLOATHBF16);
  });

  return out;
}

Tensor& div_out_mode(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    std::optional<std::string_view> mode,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, executorch::runtime::tensor_is_real_type(out), InvalidArgument, out);

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
        out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);

      xa_nn_elm_div_mode_broadcast_4D_f32xf32_f32(
          out_data,
          out_shape,
          a_data,
          inp1_shape,
          b_data,
          inp2_shape,
          mode_val);
    } else {
      xa_nn_elm_div_mode_f32xf32_f32(
          out_data, a_data, b_data, out.numel(), mode_val);
    }

    return out;
  }

  bool div_by_zero_error = false;
  const bool mode_is_trunc = (mode.has_value() && mode.value() == "trunc");
  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.out";

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    torch::executor::native::utils::
        apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
            [mode_is_trunc, &div_by_zero_error](
                const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
              if (executorch::runtime::is_integral_type<
                      CTYPE_COMPUTE,
                      /*includeBool=*/true>::value) {
                if (val_b == 0) {
                  div_by_zero_error = true;
                  return static_cast<CTYPE_COMPUTE>(0);
                }
              }
              CTYPE_COMPUTE value = val_a / val_b;
              if (mode_is_trunc) {
                value = std::trunc(value);
              } else {
                // We established above that the mode is either trunc or floor,
                // so it must be floor.
                value =
                    torch::executor::native::utils::floor_divide(val_a, val_b);
              }
              return value;
            },
            ctx,
            a,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            b,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            out,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBF16);
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      !div_by_zero_error,
      InvalidArgument,
      out,
      "Div mode operation encountered integer division by zero");

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
