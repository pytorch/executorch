/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/backends/cadence/hifi/operators/operators.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

// Defined by //executorch/kernels/portable/cpu:op_log2.
::executorch::aten::Tensor& log2_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace executor
} // namespace torch

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::resize_tensor;

// log2(x) = ln(x) / ln(2), with nnlib's natural log (2 ULP). The result is
// within 2 ULP of log2 but not exact at powers of two.
Tensor& log2_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  // nnlib takes restrict pointers, so in-place calls use the portable kernel.
  if (in.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float ||
      in.const_data_ptr() == out.const_data_ptr() ||
      resize_tensor(out, in.sizes()) != Error::Ok) {
    return ::torch::executor::native::log2_out(ctx, in, out);
  }
  const int count = static_cast<int>(in.numel());
  if (count == 0) {
    return out;
  }
  float* y = out.mutable_data_ptr<float>();
  if (xa_nn_elm_logn_f32_f32(y, in.const_data_ptr<float>(), count) != 0) {
    return ::torch::executor::native::log2_out(ctx, in, out);
  }
  constexpr float kInvLn2 = 1.44269504088896340736f;
  xtfloatx2* p = reinterpret_cast<xtfloatx2*>(y);
  const xtfloatx2* q = p;
  ae_valign load_align = XT_LASX2PP(q);
  ae_valign store_align = AE_ZALIGN64();
  const xtfloatx2 scale = kInvLn2;
  int i = 0;
  for (; i + 2 <= count; i += 2) {
    xtfloatx2 v;
    XT_LASX2IP(v, load_align, q);
    XT_SASX2IP(XT_MUL_SX2(v, scale), store_align, p);
  }
  XT_SASX2POSFP(store_align, p);
  if (i < count) {
    y[i] *= kInvLn2;
  }
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
