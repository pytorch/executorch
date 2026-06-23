/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_embedding_byte.h>

#include <cstdint>
#include <cstdlib>

#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

Tensor& quantized_embedding_byte_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& weight_zero_points,
    const Tensor& indices,
    ET_UNUSED bool pruned_weights,
    Tensor& out) {
  size_t embedding_dim = weight.size(1);

  size_t num_groups = 1;
  if (weight_scales.dim() == 2) {
    num_groups = weight_scales.size(1);
  }

  float* out_data = out.mutable_data_ptr<float>();
  const int64_t* indices_ptr = indices.const_data_ptr<int64_t>();

  const float* scales = weight_scales.const_data_ptr<float>();

  ScalarType dtype = weight.scalar_type();

#define typed_quantized_embedding_byte(ctype, dtype)                  \
  case ScalarType::dtype: {                                           \
    ctype zp = 0;                                                     \
    if (weight_zero_points.has_value()) {                             \
      zp = weight_zero_points                                         \
               ->const_data_ptr<ctype>()[index * num_groups + group]; \
    }                                                                 \
    const size_t output_group_start_offset =                          \
        embedding_dim * index + group * embedding_group_size;         \
    const ctype* w_group =                                            \
        weight.const_data_ptr<ctype>() + output_group_start_offset;   \
    for (size_t j = 0; j < embedding_group_size; ++j) {               \
      float val = ((float)w_group[j] - zp) * scale;                   \
      *out_data++ = val;                                              \
    }                                                                 \
    break;                                                            \
  }

  size_t embedding_group_size = embedding_dim / num_groups;
  for (size_t i = 0, e = indices.numel(); i < e; i++) {
    int64_t index = indices_ptr[i];
    for (size_t group = 0; group < num_groups; group++) {
      float scale = scales[index * num_groups + group];
      switch (dtype) {
        ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_embedding_byte)
        default:
          ET_DCHECK_MSG(
              false, "Unhandled dtype %s", torch::executor::toString(dtype));
      }
    }
  }

#undef typed_quantized_embedding_byte
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
