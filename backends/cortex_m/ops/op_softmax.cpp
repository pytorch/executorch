/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include <cmath>
#include <cstdint>
#include <limits>

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

namespace {

constexpr int32_t kCmsisSoftmaxZeroPoint = -128;

inline bool is_int8_tensor(const Tensor& tensor) {
  return tensor.scalar_type() == ScalarType::Char;
}

inline bool is_last_dim(const Tensor& tensor, int64_t dim) {
  const auto rank = tensor.dim();
  const int64_t positive_dim = dim >= 0 ? dim : dim + rank;
  return positive_dim == static_cast<int64_t>(rank - 1);
}

inline int64_t normalize_dim(const Tensor& tensor, int64_t dim) {
  const auto rank = tensor.dim();
  const int64_t positive_dim = dim >= 0 ? dim : dim + rank;
  return positive_dim;
}

} // namespace

Tensor& softmax_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t dim,
    int64_t input_zero_point,
    int64_t output_zero_point,
    int64_t input_multiplier,
    int64_t input_shift,
    int64_t diff_min,
    Tensor& out) {
  if (!is_int8_tensor(input) || !is_int8_tensor(out)) {
    ET_LOG(
        Error,
        "softmax_out: only int8 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (!is_last_dim(input, dim)) {
    ET_LOG(
        Error,
        "softmax_out: only last-dimension softmax is supported (dim=%lld, rank=%zu)",
        static_cast<long long>(dim),
        static_cast<size_t>(input.dim()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const int32_t output_zp_val = static_cast<int32_t>(output_zero_point);
  const int32_t input_multiplier_val = static_cast<int32_t>(input_multiplier);
  const int32_t input_shift_val = static_cast<int32_t>(input_shift);
  const int32_t diff_min_val = static_cast<int32_t>(diff_min);

  validate_single_quant_params(
      Scalar(static_cast<int32_t>(input_zero_point)),
      Scalar(input_multiplier_val),
      Scalar(input_shift_val),
      "softmax input");

  const auto positive_dim = normalize_dim(input, dim);
  const int64_t row_size64 = input.size(positive_dim);
  if (row_size64 <= 0 || row_size64 > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "softmax_out: row size must fit in int32 (row_size=%lld)",
        static_cast<long long>(row_size64));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const int32_t row_size = static_cast<int32_t>(row_size64);
  const int64_t num_rows64 = input.numel() / row_size64;
  if (num_rows64 <= 0 || num_rows64 > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "softmax_out: num_rows must fit in int32 (num_rows=%lld)",
        static_cast<long long>(num_rows64));
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int32_t num_rows = static_cast<int32_t>(num_rows64);

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  if (num_rows <= 0 || row_size <= 0) {
    ET_LOG(
        Error,
        "softmax_out: invalid args (dim=%ld, rows=%d, row_size=%d)",
        static_cast<long>(dim),
        num_rows,
        row_size);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (output_zp_val != kCmsisSoftmaxZeroPoint) {
    ET_LOG(
        Error,
        "softmax_out: expected output zero_point=%d (got zero_point=%d)",
        kCmsisSoftmaxZeroPoint,
        output_zp_val);
    context.fail(Error::InvalidArgument);
    return out;
  }

  arm_softmax_s8(
      input_data,
      num_rows,
      row_size,
      input_multiplier_val,
      input_shift_val,
      diff_min_val,
      output_data);

  return out;
}

} // namespace native
} // namespace cortex_m
