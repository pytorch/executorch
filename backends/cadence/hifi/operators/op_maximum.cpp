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
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::promoteTypes;
using torch::executor::apply_binary_elementwise_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;

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
struct MaximumInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct MaximumInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value =
              torch::executor::native::utils::max_override(a_casted, b_casted);

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct MaximumInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& maximum_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  bool optimized = true;
  /*find broadcast*/
  bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  bool broadcast = (a_is_broadcasted || b_is_broadcasted);

  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if ((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
    optimized = false;
  if ((broadcast == true) && (max_dim > kNnlibMaxDim))
    optimized = false;

  if (optimized) {
    float* a_data = a.mutable_data_ptr<float>();
    float* b_data = b.mutable_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    if (broadcast == true) {
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

      for (int i = 0; i < out.dim(); i++) {
        out_shape[i + off_o] = out.size(i);
      }

      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);

      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);

      xa_nn_elm_maximum_broadcast_4D_f32xf32_f32(
          out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape);
    } else {
      xa_nn_elm_maximum_f32xf32_f32(out_data, a_data, b_data, out.numel());
    }
    return out;
  }
  ET_SWITCH_REALHB_TYPES(a_type, ctx, "maximum.out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, "maximum.out", CTYPE_B, [&]() {
      using CTYPE_IN = typename torch::executor::
          promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
      ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
      ET_SWITCH_REALHB_TYPES(out_type, ctx, "maximum.out", CTYPE_OUT, [&]() {
        MaximumInner<
            can_cast<CTYPE_IN, CTYPE_OUT>::value,
            CTYPE_A,
            CTYPE_B,
            CTYPE_IN,
            CTYPE_OUT>::run(a, b, out);
      });
    });
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
