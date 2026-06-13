#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstdint>
#include <variant>
#include <vector>

/*
 * This file contains types and methods related to quantization parameters.
 * Quant params, in combination with dtype, should provide enough information
 * to interpret raw tensor memory and inform kernel dispatch.
 */

namespace executorch::backends::xnnpack::core {

/*
 * Represents quantization parameters for per-tensor quantization. This means
 * that there is a single scale and zero point for the entire tensor.
 *
 * For a tensor of shape [A, B, C], this is equivalent to a block size of
 * [A, B, C].
 */
struct PerTensorQuantParams {
  DType scale_dtype = DType::Float32;
  float scale = 0.0f;
  int32_t zero_point = 0;
  bool has_zero_point = false;

  bool operator==(const PerTensorQuantParams& o) const {
    return scale_dtype == o.scale_dtype && scale == o.scale &&
        zero_point == o.zero_point && has_zero_point == o.has_zero_point;
  }
};

/*
 * Represents per-axis quantization parameters. Scale and zero point are
 * shared by all elements with the same index along the target axis.
 *
 * For a tensor of shape [A, B, C] and axis=1, this is equivalent to a block
 * size of [A, 1, C] with a scale shape [1, B, 1].
 */
struct PerAxisQuantParams {
  int8_t axis;
  DType scale_dtype = DType::Float32;
  bool has_zero_point = false;

  bool operator==(const PerAxisQuantParams& o) const {
    return axis == o.axis && scale_dtype == o.scale_dtype &&
        has_zero_point == o.has_zero_point;
  }
};

/*
 * Represents per-row quantization parameters. Scale and zero point are
 * shared by all elements with the same indices along non-target axes; `axis`
 * is the reduced dim, negative values index from the end, and it defaults to
 * -1 (the last dim, i.e. per-token).
 *
 * For a tensor of shape [A, B, C] and axis=1, this is equivalent to a block
 * size of [1, B, 1] with a scale shape of [A, 1, C].
 */
struct PerRowQuantParams {
  int8_t axis = -1;
  DType scale_dtype = DType::Float32;
  bool has_zero_point = false;
  // When true, this is a dynamically-quantized activation (XNNPACK qdint8):
  // the per-row scale/zero point are computed at runtime rather than stored.
  // `axis` is the reduced (channel) dim, so the number of trailing "row" dims
  // (XNNPACK's num_nonbatch_dims) is -axis for the usual negative axis.
  bool is_dynamic = false;

  bool operator==(const PerRowQuantParams& o) const {
    return axis == o.axis && scale_dtype == o.scale_dtype &&
        has_zero_point == o.has_zero_point && is_dynamic == o.is_dynamic;
  }
};

/*
 * Represents per-block quantization parameters. Elements are grouped along
 * `axis` into groups of `block_size`. Elements within a group share a scale
 * and zero point. The block size must evenly divide the input tensor shape
 * along the target axis.
 *
 * For a tensor of shape [A, B, C] and axis=1, blocks are size
 * [1, block_size, 1] with a scale shape of [A, B / block_size, C].
 */
struct PerBlockQuantParams {
  int8_t axis;
  int32_t block_size;
  DType scale_dtype = DType::Float32;
  bool has_zero_point = false;

  bool operator==(const PerBlockQuantParams& o) const {
    return axis == o.axis && block_size == o.block_size &&
        scale_dtype == o.scale_dtype && has_zero_point == o.has_zero_point;
  }
};

/*
 * Quantization parameter descriptor. Describes the type and granularity of
 * the quantization scheme. Does not contain the actual scale and zero point
 * data, as these are stored in the auxialliary storage on the tensor.
 */
using QuantParams = std::variant<
    PerTensorQuantParams,
    PerAxisQuantParams,
    PerRowQuantParams,
    PerBlockQuantParams>;

QuantParams qint8_per_channel_sym(int8_t axis);
QuantParams qint8_per_tensor_sym(float scale);
QuantParams quint8_per_tensor_asym(float scale, int32_t zero_point);
QuantParams quint8_per_row_asym(int8_t axis);
QuantParams quint8_per_token_asym();
QuantParams qint4_blockwise_sym(int8_t axis, int32_t block_size);

/*
 * Returns true if the given dtype is quantized. Quantized types
 * require additional metadata to interpret.
 */
bool is_quantized(DType dtype);

/*
 * Returns true if the dtype's elements are smaller than a byte (e.g. packed
 * 4-bit), and so are not individually byte-addressable.
 */
bool is_subbyte(DType dtype);

/*
 * Returns the size in bytes of a single element. Precondition: the dtype is
 * byte-aligned (!is_subbyte); sub-byte types have no whole-byte stride.
 */
size_t byte_stride(DType dtype);

/*
 * Returns true if the given quant params have a zero point.
 */
bool is_asymmetric(const QuantParams& params);

/*
 * Returns the number of auxilliary storage buffers required to
 * store the parameters (scales + zero points) for the given quant
 * scheme.
 */
uint8_t aux_buffer_count(DType dtype, const QuantParams& params);
runtime::Result<std::vector<size_t>> compute_aux_storage_sizes(
    runtime::Span<const uint64_t> sizes,
    DType dtype,
    const QuantParams& params);

} // namespace executorch::backends::xnnpack::core
