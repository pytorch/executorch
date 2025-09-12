/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <xa_type_def.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

void quantize_per_tensor_asym8s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();
  xa_nn_elm_quantize_f32_asym8s(out_data, input_data, scale, zero_point, numel);
}

} // namespace native
} // namespace HiFi
} // namespace impl
