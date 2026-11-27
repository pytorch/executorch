/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include "arm_nnsupportfunctions.h"

namespace cortex_m {
namespace native {
namespace {

constexpr size_t kMaxCopyDims = executorch::runtime::kTensorDimensionLimit;
using TensorList = executorch::aten::ArrayRef<Tensor>;

struct CopyLoop {
  size_t count;
  ptrdiff_t src_stride;
  size_t dst_stride;
  size_t index;
};

#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
bool validate_copy_bounds(
    TensorList inputs,
    Int64ArrayRef shapes,
    Int64ArrayRef src_strides,
    Int64ArrayRef dst_strides,
    Int64ArrayRef src_offsets,
    Int64ArrayRef dst_origins,
    Int64ArrayRef dst_steps,
    const Tensor& out) {
  const size_t rank = dst_strides.size();
  if (inputs.empty() || inputs.size() > SIZE_MAX / kMaxCopyDims ||
      out.scalar_type() != ScalarType::Char || rank > kMaxCopyDims ||
      rank != static_cast<size_t>(out.dim()) ||
      shapes.size() != inputs.size() * rank ||
      src_strides.size() != shapes.size() ||
      dst_origins.size() != shapes.size() || dst_steps.size() != rank ||
      src_offsets.size() != inputs.size()) {
    return false;
  }
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dst_strides[dim] < 0 || dst_strides[dim] > INT32_MAX ||
        dst_steps[dim] <= 0 || dst_steps[dim] > INT32_MAX ||
        (out.numel() != 0 && out.size(dim) > 1 &&
         dst_strides[dim] != out.strides()[dim])) {
      return false;
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& input = inputs[i];
    const size_t start = i * rank;
    if (input.scalar_type() != ScalarType::Char || src_offsets[i] < 0 ||
        src_offsets[i] > INT32_MAX) {
      return false;
    }
    bool empty = false;
    for (size_t dim = 0; dim < rank; ++dim) {
      const int64_t count = shapes[start + dim];
      const int64_t origin = dst_origins[start + dim];
      if (count < 0 || count > INT32_MAX || origin < 0 ||
          origin > out.size(dim) || src_strides[start + dim] < INT32_MIN ||
          src_strides[start + dim] > INT32_MAX) {
        return false;
      }
      if (count > 0 &&
          (origin == out.size(dim) ||
           (count - 1) * dst_steps[dim] > out.size(dim) - 1 - origin)) {
        return false;
      }
      empty |= count == 0;
    }
    if (empty) {
      continue;
    }
    if (src_offsets[i] >= input.numel() || out.numel() == 0) {
      return false;
    }

    uint32_t src_before = static_cast<uint32_t>(src_offsets[i]);
    uint32_t src_after = input.numel() - src_before - 1;
    uint32_t dst_remaining = out.numel() - 1;
    for (size_t dim = 0; dim < rank; ++dim) {
      const int32_t count = static_cast<int32_t>(shapes[start + dim]);
      const int32_t stride = static_cast<int32_t>(src_strides[start + dim]);
      const int64_t src_extent = static_cast<int64_t>(count - 1) * stride;
      const uint64_t src_span = src_extent < 0 ? -src_extent : src_extent;
      uint32_t& src_remaining = src_extent < 0 ? src_before : src_after;
      const uint32_t dst_last =
          static_cast<uint32_t>(dst_origins[start + dim]) +
          static_cast<uint32_t>(count - 1) *
              static_cast<uint32_t>(dst_steps[dim]);
      const uint64_t dst_extent = static_cast<uint64_t>(dst_last) *
          static_cast<uint32_t>(dst_strides[dim]);
      if (src_span > src_remaining || dst_extent > dst_remaining) {
        return false;
      }
      src_remaining -= static_cast<uint32_t>(src_span);
      dst_remaining -= static_cast<uint32_t>(dst_extent);
    }
  }
  return true;
}
#endif

void copy_region(
    const Tensor& input,
    Int64ArrayRef shape,
    Int64ArrayRef src_strides,
    Int64ArrayRef dst_strides,
    int64_t src_offset,
    Int64ArrayRef dst_origin,
    Int64ArrayRef dst_steps,
    Tensor& out) {
  CopyLoop loops[kMaxCopyDims];
  size_t loop_count = 0;
  size_t copy_length = 1;
  size_t dst = 0;
  for (size_t position = shape.size(); position > 0; --position) {
    const size_t dim = out.dim_order()[position - 1];
    const size_t count = static_cast<size_t>(shape[dim]);
    if (count == 0) {
      return;
    }
    dst += static_cast<size_t>(dst_origin[dim]) *
        static_cast<size_t>(dst_strides[dim]);
    if (count == 1) {
      continue;
    }
    const size_t dst_stride = static_cast<size_t>(dst_strides[dim]) *
        static_cast<size_t>(dst_steps[dim]);
    const ptrdiff_t src_stride = static_cast<ptrdiff_t>(src_strides[dim]);
    // Coalesce the common contiguous suffix into one memcpy per outer index.
    if (loop_count == 0 && src_stride == static_cast<ptrdiff_t>(copy_length) &&
        dst_stride == copy_length) {
      copy_length *= count;
    } else {
      loops[loop_count++] = {count, src_stride, dst_stride, 0};
    }
  }

  const int8_t* source = input.const_data_ptr<int8_t>();
  int8_t* output = out.mutable_data_ptr<int8_t>();
  ptrdiff_t src = src_offset;
  if (loop_count == 0) {
    arm_memcpy_s8(output + dst, source + src, copy_length);
    return;
  }
  // Run the innermost loop separately so its counter and strides can stay in
  // registers across short copies. Rewind its offsets before carrying into the
  // outer dimensions.
  const CopyLoop inner = loops[0];
  while (true) {
    size_t remaining = inner.count;
    while (true) {
      arm_memcpy_s8(output + dst, source + src, copy_length);
      if (--remaining == 0) {
        break;
      }
      src += inner.src_stride;
      dst += inner.dst_stride;
    }
    src -= static_cast<ptrdiff_t>(inner.count - 1) * inner.src_stride;
    dst -= (inner.count - 1) * inner.dst_stride;
    size_t level = 1;
    for (; level < loop_count; ++level) {
      CopyLoop& loop = loops[level];
      if (++loop.index < loop.count) {
        src += loop.src_stride;
        dst += loop.dst_stride;
        break;
      }
      src -= static_cast<ptrdiff_t>(loop.count - 1) * loop.src_stride;
      dst -= (loop.count - 1) * loop.dst_stride;
      loop.index = 0;
    }
    if (level == loop_count) {
      break;
    }
  }
}

} // namespace

// cppcheck-suppress unusedFunction
Tensor& strided_copy_out(
    // cppcheck-suppress constParameterReference
    [[maybe_unused]] KernelRuntimeContext& context,
    TensorList inputs,
    Int64ArrayRef shapes,
    Int64ArrayRef src_strides,
    Int64ArrayRef dst_strides,
    Int64ArrayRef src_offsets,
    Int64ArrayRef dst_origins,
    Int64ArrayRef dst_steps,
    Tensor& out) {
#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
  if (!validate_copy_bounds(
          inputs,
          shapes,
          src_strides,
          dst_strides,
          src_offsets,
          dst_origins,
          dst_steps,
          out)) {
    ET_LOG(Error, "strided_copy_out: invalid copy plan or dtype");
    context.fail(Error::InvalidArgument);
    return out;
  }
#endif
  if (out.numel() == 0) {
    return out;
  }
  const size_t rank = dst_strides.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    copy_region(
        inputs[i],
        shapes.slice(i * rank, rank),
        src_strides.slice(i * rank, rank),
        dst_strides,
        src_offsets[i],
        dst_origins.slice(i * rank, rank),
        dst_steps,
        out);
  }
  return out;
}

} // namespace native
} // namespace cortex_m
