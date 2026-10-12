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

// Defined by //executorch/kernels/portable/cpu:op_relu.
::executorch::aten::Tensor& relu_out(
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

Tensor& relu_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  if (in.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float ||
      resize_tensor(out, in.sizes()) != Error::Ok) {
    return ::torch::executor::native::relu_out(ctx, in, out);
  }
  const int count = static_cast<int>(in.numel());
  if (count == 0) {
    return out;
  }
  const float* x = in.const_data_ptr<float>();
  float* y = out.mutable_data_ptr<float>();
  // nnlib takes restrict pointers, so in-place calls use a two-lane loop.
  if (x != y) {
    xa_nn_vec_relu_std_f32_f32(y, x, count);
    return out;
  }
  xtfloatx2* p = reinterpret_cast<xtfloatx2*>(y);
  const xtfloatx2* q = p;
  ae_valign load_align = XT_LASX2PP(q);
  ae_valign store_align = AE_ZALIGN64();
  const xtfloatx2 zero = 0.0f;
  int i = 0;
  for (; i + 2 <= count; i += 2) {
    xtfloatx2 v;
    XT_LASX2IP(v, load_align, q);
    XT_SASX2IP(XT_MAX_SX2(v, zero), store_align, p);
  }
  XT_SASX2POSFP(store_align, p);
  if (i < count) {
    y[i] = y[i] > 0.0f ? y[i] : 0.0f;
  }
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
