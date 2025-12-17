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
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/backends/cadence/common/xt_macros.h>

using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace impl {
namespace HiFi {
namespace native {

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, CTYPE_IN alpha_val, Tensor& out) {
    torch::executor::apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
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
};

template <typename CTYPE_IN>
struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, CTYPE_IN, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug<CTYPE_IN> {};

} // namespace

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_realhbbf16_type(out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(a, b, out),
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType alpha_type =
      torch::executor::native::utils::get_scalar_dtype(alpha);
  ScalarType common_type =
      executorch::runtime::promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::canCast(common_type, out_type),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_alpha_type(alpha_type, common_type),
      InvalidArgument,
      out);

  float alpha_val;
  torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

  static constexpr const char op_name[] = "add.out";
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = 1;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if ((out_type != ScalarType::Float) || (alpha_val != 1.0))
    optimized = 0;

  bool float_types =
      (a_type == ScalarType::Float) && (b_type == ScalarType::Float);

  if ((a_dim == 0) && float_types) {
    for (int i = 0; i < b.numel(); i++)
      out.mutable_data_ptr<float>()[i] = a.const_data_ptr<float>()[0] +
          alpha_val * b.const_data_ptr<float>()[i];
    return out;
  }
  if ((b_dim == 0) && float_types) {
    // Precompute the value of b * alpha since it's a constant.
    const float val_b = alpha_val * b.const_data_ptr<float>()[0];
    for (int i = 0; i < a.numel(); i++)
      out.mutable_data_ptr<float>()[i] = a.const_data_ptr<float>()[i] + val_b;
    return out;
  }

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = 0;

  if (optimized) {
    const float* const a_data = a.const_data_ptr<float>();
    const float* const b_data = b.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();

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

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_add_broadcast_4D_f32xf32_f32,
          out_data,
          out_shape,
          a_data,
          inp1_shape,
          b_data,
          inp2_shape);
    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_add_f32xf32_f32,
          out_data,
          a_data,
          b_data,
          out.numel());
    }

    return out;
  }

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_alpha =
        torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(alpha);
    torch::executor::native::utils::
        apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
            [val_alpha](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
              return val_a + val_alpha * val_b;
            },
            ctx,
            a,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            b,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            out,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
