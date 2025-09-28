/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <xa_type_def.h>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

void dequantize_per_tensor_asym8s_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    __ET_UNUSED int64_t quant_min,
    __ET_UNUSED int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  const size_t numel = out.numel();
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  xa_nn_elm_dequantize_asym8s_f32(
      out_data, input_data, zero_point, scale, numel);
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
