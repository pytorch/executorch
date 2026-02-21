/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_softmax.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;
using ::impl::generic::kernels::quantize;

/**
 * @brief Compute position mask for incremental causal masking
 *
 * Mask semantics: maskArray[i] = true means mask out (don't attend to position
 * i) For posValue = P, elements 0..P are attended (false), elements P+1.. are
 * masked (true).
 *
 * @param maskArray Output mask array to populate
 * @param size Size of the mask array (softmax dimension size)
 * @param posValue Current position value (elements 0..posValue are attended)
 */
void computePositionMask(bool* maskArray, size_t size, int64_t posValue) {
  for (size_t i = 0; i < size; ++i) {
    maskArray[i] = (static_cast<int64_t>(i) > posValue);
  }
}

/**
 * @brief Update position mask incrementally for next row
 *
 * This is an O(1) operation per row instead of O(n) full recomputation.
 * Unmasks position (lastUnmaskedPos + 1) to newPosValue.
 *
 * @param maskArray Mask array to update in-place
 * @param size Size of the mask array
 * @param lastUnmaskedPos Reference to track highest unmasked position
 * @param newPosValue New position value to unmask up to
 */
void updatePositionMaskIncremental(
    bool* maskArray,
    size_t size,
    int64_t& lastUnmaskedPos,
    int64_t newPosValue) {
  // Clamp to a local variable to maintain clear semantics and avoid modifying
  // the parameter, which could cause confusion about caller-side effects.
  const int64_t clampedPosValue =
      std::min(newPosValue, static_cast<int64_t>(size) - 1);

  while (lastUnmaskedPos < clampedPosValue) {
    lastUnmaskedPos++;
    if (lastUnmaskedPos >= 0 && lastUnmaskedPos < static_cast<int64_t>(size)) {
      maskArray[lastUnmaskedPos] = false;
    }
  }
}

/**
 * @brief Core implementation of quantized softmax with optional causal masking.
 *
 * Algorithm Overview:
 * ===================
 * This function computes softmax on quantized input tensors with support for
 * position-based causal masking, commonly used in transformer attention layers.
 *
 * Softmax Formula (numerically stable version):
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))) for all j
 *
 * The computation proceeds in these phases:
 *   1. Dequantize: Convert quantized input to float using in_scale and in_zero_point
 *   2. Find max: Compute max over unmasked positions (for numerical stability)
 *   3. Exp & sum: Compute exp(x - max) and accumulate sum for unmasked positions
 *   4. Normalize: Divide by sum to get probabilities
 *   5. Quantize: Convert back to quantized output using out_scale and out_zero_point
 *
 * Causal Masking (mask_type == 1):
 * ================================
 * Implements incremental causal attention where each row can attend to
 * progressively more positions. For base position P and row index i:
 *   - Positions 0 to (P + i) are attended (included in softmax)
 *   - Positions (P + i + 1) onwards are masked (set to 0 probability)
 *
 * This creates a lower-triangular attention pattern commonly used in
 * autoregressive language models to prevent attending to future tokens.
 *
 * Memory Layout:
 * ==============
 * Input is treated as a 2D tensor of shape [outerSize, lastDimSize] where:
 *   - outerSize = total_elements / lastDimSize (number of rows)
 *   - lastDimSize = size of the last dimension (softmax is computed over this)
 *
 * @tparam T Quantized data type (int8, uint8, int16, etc.)
 */
template <typename T>
void quantized_softmax_per_tensor_(
    const Tensor& input,
    ET_UNUSED const Tensor& mask,
    int64_t dim,
    int64_t mask_type,
    const Tensor& pos,
    const float in_scale,
    const int64_t in_zero_point,
    const float out_scale,
    const int64_t out_zero_point,
    Tensor& out) {
  const T* __restrict__ in_data = input.const_data_ptr<T>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();

  float out_inv_scale = 1.0f / out_scale;
  if (dim < 0) {
    dim += input.dim();
  }

  const size_t num_dims = input.dim();
  const size_t lastDimSize = input.size(num_dims - 1);
  const size_t outerSize = input.numel() / lastDimSize;

  // Validate dimension: this implementation only supports softmax over the last
  // dimension. The dim parameter after normalization should equal (num_dims -
  // 1).
  ET_DCHECK_MSG(
      dim == static_cast<int64_t>(num_dims - 1),
      "quantized_softmax_per_tensor_ only supports softmax over the last "
      "dimension. Got dim=%ld, expected dim=%zu",
      static_cast<long>(dim),
      num_dims - 1);

  const int64_t input_size = input.numel();
  std::vector<float> x(input_size);  // Working buffer for dequantized values

  // ========================================================================
  // Mask Initialization (for mask_type == 1: position-based causal masking)
  // ========================================================================
  // positionMask[i] = true means position i is masked (excluded from softmax)
  // positionMask[i] = false means position i is attended (included in softmax)
  //
  // Initial state based on basePosValue (from pos tensor):
  //   - If basePosValue < 0: all positions masked (edge case)
  //   - If basePosValue >= lastDimSize: no positions masked
  //   - Otherwise: positions 0..basePosValue unmasked, rest masked
  // ========================================================================
  std::unique_ptr<bool[]> positionMask;
  int64_t lastUnmaskedPos = -1;  // Tracks highest unmasked index for incremental updates
  int64_t basePosValue = 0;

  if (mask_type == 1 && pos.numel() > 0) {
    positionMask = std::make_unique<bool[]>(lastDimSize);

    if (pos.scalar_type() == ::executorch::aten::ScalarType::Short) {
      basePosValue = static_cast<int64_t>(pos.const_data_ptr<int16_t>()[0]);
    } else {
      basePosValue = pos.const_data_ptr<int64_t>()[0];
    }

    if (basePosValue < 0) {
      std::fill(positionMask.get(), positionMask.get() + lastDimSize, true);
      lastUnmaskedPos = -1;
    } else if (basePosValue >= static_cast<int64_t>(lastDimSize)) {
      std::fill(positionMask.get(), positionMask.get() + lastDimSize, false);
      lastUnmaskedPos = static_cast<int64_t>(lastDimSize) - 1;
    } else {
      computePositionMask(positionMask.get(), lastDimSize, basePosValue);
      lastUnmaskedPos = basePosValue;
    }
  }

  // Determine if incremental mask updates are needed. This is true only when:
  // - mask_type == 1 (position-based causal masking is enabled)
  // - positionMask was allocated (pos tensor has elements)
  // - basePosValue >= 0 (not all positions are masked from the start)
  // By computing this once outside the loop, we avoid redundant checks on every
  // iteration since basePosValue doesn't change during the loop.
  const bool needsIncrementalMaskUpdate =
      (mask_type == 1 && positionMask && basePosValue >= 0);

  // ========================================================================
  // Main Loop: Process each row independently
  // ========================================================================
  // For each row idx in [0, outerSize):
  //   1. Update mask if using incremental causal masking
  //   2. Dequantize input values
  //   3. Find max over unmasked positions (numerical stability)
  //   4. Compute exp(x - max) for unmasked, 0 for masked positions
  //   5. Normalize by sum to get probabilities
  //   6. Quantize and store output
  // ========================================================================
  for (size_t idx = 0; idx < outerSize; ++idx) {
    const size_t base = idx * lastDimSize;

    // Step 1: Incremental mask update for causal attention
    // For row idx, unmask positions up to (basePosValue + idx)
    // This is O(1) amortized per row instead of O(n) full recomputation
    if (needsIncrementalMaskUpdate) {
      int64_t newPosValue = basePosValue + static_cast<int64_t>(idx);
      updatePositionMaskIncremental(
          positionMask.get(), lastDimSize, lastUnmaskedPos, newPosValue);
    }

    // Step 2: Dequantize input values
    // x_float = (x_quant - zero_point) * scale
    for (size_t i = 0; i < lastDimSize; ++i) {
      x[base + i] = dequantize<T>(
          in_data[base + i], in_scale, static_cast<int32_t>(in_zero_point));
    }

    // Step 3: Find max over unmasked positions for numerical stability
    // Subtracting max prevents exp() overflow for large values
    float max_in = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < lastDimSize; ++i) {
      bool isMasked =
          (mask_type == 1 && positionMask) ? positionMask[i] : false;
      if (!isMasked) {
        max_in = std::max(max_in, x[base + i]);
      }
    }

    // Handle edge case: all positions masked (use 0 as neutral max)
    if (max_in == -std::numeric_limits<float>::infinity()) {
      max_in = 0.0f;
    }

    // Step 4: Compute exp(x - max) and accumulate sum
    // Masked positions get 0, unmasked positions get exp(x - max)
    float temp_sum = 0.0f;
    for (size_t i = 0; i < lastDimSize; ++i) {
      bool isMasked =
          (mask_type == 1 && positionMask) ? positionMask[i] : false;
      if (isMasked) {
        x[base + i] = 0.0f;  // Masked positions contribute 0 probability
      } else {
        x[base + i] = std::exp(x[base + i] - max_in);
        temp_sum += x[base + i];
      }
    }

    // Step 5 & 6: Normalize and quantize output
    // softmax_i = exp_i / sum, then quantize to output type
    float recip = (temp_sum > 0.0f) ? (1.0f / temp_sum) : 0.0f;
    for (size_t i = 0; i < lastDimSize; ++i) {
      float res = x[base + i] * recip;
      out_data[base + i] =
          quantize<T>(res, out_inv_scale, static_cast<int32_t>(out_zero_point));
    }
  }
}

/**
 * @brief Wrapper that extracts quantization parameters from tensors.
 *
 * This function extracts scalar quantization parameters from input tensors
 * and delegates to quantized_softmax_per_tensor_ for the actual computation.
 * Used when quantization parameters are provided as single-element tensors
 * rather than scalar values.
 */
template <typename T>
void quantized_softmax_(
    const Tensor& input,
    const Tensor& mask,
    const int64_t dim,
    int64_t mask_type,
    const Tensor& pos,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const Tensor& out_scale,
    const Tensor& out_zero_point,
    Tensor& out) {
  // Extract the zero point and scale for input tensor.
  float input_scale = in_scale.const_data_ptr<float>()[0];
  int64_t input_zero_point = in_zero_point.const_data_ptr<int64_t>()[0];
  float output_scale = out_scale.const_data_ptr<float>()[0];
  int64_t output_zero_point = out_zero_point.const_data_ptr<int64_t>()[0];
  quantized_softmax_per_tensor_<T>(
      input,
      mask,
      dim,
      mask_type,
      pos,
      input_scale,
      input_zero_point,
      output_scale,
      output_zero_point,
      out);
}

} // namespace

Tensor& quantized_softmax_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& mask,
    int64_t dim,
    int64_t mask_type,
    const Tensor& pos,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const Tensor& out_scale,
    const Tensor& out_zero_point,
    Tensor& out) {
#define typed_quantized_softmax(ctype, dtype) \
  case ScalarType::dtype: {                   \
    quantized_softmax_<ctype>(                \
        input,                                \
        mask,                                 \
        dim,                                  \
        mask_type,                            \
        pos,                                  \
        in_scale,                             \
        in_zero_point,                        \
        out_scale,                            \
        out_zero_point,                       \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_softmax)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_softmax
  return out;
}

Tensor& quantized_softmax_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& mask,
    int64_t dim,
    int64_t mask_type,
    const Tensor& pos,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
#define typed_quantized_softmax(ctype, dtype) \
  case ScalarType::dtype: {                   \
    quantized_softmax_per_tensor_<ctype>(     \
        input,                                \
        mask,                                 \
        dim,                                  \
        mask_type,                            \
        pos,                                  \
        in_scale,                             \
        in_zero_point,                        \
        out_scale,                            \
        out_zero_point,                       \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_softmax)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_softmax
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
