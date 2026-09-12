#include <executorch/backends/xnnpack/runtime/core/quant_params.h>

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#include <cstdlib>

namespace executorch::backends::xnnpack::core {

using executorch::runtime::Span;

QuantParams qint8_per_channel_sym(int8_t axis) {
  return PerAxisQuantParams{.axis = axis, .has_zero_point = false};
}

QuantParams qint8_per_tensor_sym(float scale) {
  return PerTensorQuantParams{
      .scale = scale, .zero_point = 0, .has_zero_point = false};
}

QuantParams quint8_per_tensor_asym(float scale, int32_t zero_point) {
  return PerTensorQuantParams{
      .scale = scale, .zero_point = zero_point, .has_zero_point = true};
}

QuantParams quint8_per_row_asym(int8_t axis) {
  return PerRowQuantParams{.axis = axis, .has_zero_point = true};
}

QuantParams quint8_per_token_asym() {
  return PerRowQuantParams{.axis = -1, .has_zero_point = true};
}

QuantParams qint4_blockwise_sym(int8_t axis, int32_t block_size) {
  return PerBlockQuantParams{
      .axis = axis, .block_size = block_size, .has_zero_point = false};
}

bool is_quantized(DType dtype) {
  switch (dtype) {
    case DType::Float32:
    case DType::Float16:
    case DType::BFloat16:
    case DType::Int64:
    case DType::UInt64:
      return false;
    case DType::QInt8:
    case DType::QInt4:
    case DType::QInt32:
    case DType::QUInt8:
      return true;
  }
}

bool is_subbyte(DType dtype) {
  switch (dtype) {
    case DType::QInt4:
      return true;
    case DType::Float32:
    case DType::Float16:
    case DType::BFloat16:
    case DType::Int64:
    case DType::UInt64:
    case DType::QInt8:
    case DType::QInt32:
    case DType::QUInt8:
      return false;
  }
}

size_t byte_stride(DType dtype) {
  switch (dtype) {
    case DType::QInt8:
    case DType::QUInt8:
      return 1;
    case DType::Float16:
    case DType::BFloat16:
      return 2;
    case DType::Float32:
    case DType::QInt32:
      return 4;
    case DType::Int64:
    case DType::UInt64:
      return 8;
    case DType::QInt4:
      // Sub-byte; no whole-byte stride. Guard callers with is_subbyte().
      abort();
  }
}

bool is_asymmetric(const QuantParams& params) {
  return std::visit([](const auto& p) { return p.has_zero_point; }, params);
}

uint8_t aux_buffer_count(DType dtype, const QuantParams& params) {
  if (!is_quantized(dtype))
    return 0;

  uint8_t count = 1; // scales
  if (is_asymmetric(params))
    count++; // zero_points
  return count;
}

static runtime::Result<size_t> scale_element_count(
    Span<const uint64_t> sizes,
    const QuantParams& params) {
  return std::visit(
      overloaded{
          [](const PerTensorQuantParams&) -> runtime::Result<size_t> {
            return 1;
          },
          [&](const PerAxisQuantParams& p) -> runtime::Result<size_t> {
            ET_CHECK_OR_RETURN_ERROR(
                p.axis >= 0 && static_cast<size_t>(p.axis) < sizes.size(),
                InvalidArgument,
                "Per-axis quant axis %d is out of range for a %zu-dim tensor",
                static_cast<int>(p.axis),
                sizes.size());
            return sizes[p.axis];
          },
          [&](const PerRowQuantParams& p) -> runtime::Result<size_t> {
            int rank = static_cast<int>(sizes.size());
            int axis = p.axis < 0 ? p.axis + rank : p.axis;
            ET_CHECK_OR_RETURN_ERROR(
                axis >= 0 && axis < rank,
                InvalidArgument,
                "Per-row quant axis %d is out of range for a %d-dim tensor",
                static_cast<int>(p.axis),
                rank);
            size_t count = 1;
            for (size_t i = 0; i < sizes.size(); i++) {
              if (i != static_cast<size_t>(axis))
                count *= sizes[i];
            }
            return count;
          },
          [&](const PerBlockQuantParams& p) -> runtime::Result<size_t> {
            ET_CHECK_OR_RETURN_ERROR(
                p.axis >= 0 && static_cast<size_t>(p.axis) < sizes.size(),
                InvalidArgument,
                "Per-block quant axis %d is out of range for a %zu-dim tensor",
                static_cast<int>(p.axis),
                sizes.size());
            ET_CHECK_OR_RETURN_ERROR(
                p.block_size > 0,
                InvalidArgument,
                "Per-block quant block_size must be positive, got %d",
                p.block_size);
            auto axis = static_cast<size_t>(p.axis);
            ET_CHECK_OR_RETURN_ERROR(
                sizes[axis] % static_cast<uint64_t>(p.block_size) == 0,
                InvalidArgument,
                "Per-block quant block_size %d must evenly divide axis %d (size %zu)",
                p.block_size,
                static_cast<int>(p.axis),
                static_cast<size_t>(sizes[axis]));
            size_t num_blocks = sizes[axis] / p.block_size;
            size_t other_dims = 1;
            for (size_t i = 0; i < sizes.size(); i++) {
              if (i != axis)
                other_dims *= sizes[i];
            }
            return num_blocks * other_dims;
          },
      },
      params);
}

static DType scale_dtype_of(const QuantParams& params) {
  return std::visit(
      overloaded{
          [](const PerTensorQuantParams& p) { return p.scale_dtype; },
          [](const PerAxisQuantParams& p) { return p.scale_dtype; },
          [](const PerRowQuantParams& p) { return p.scale_dtype; },
          [](const PerBlockQuantParams& p) { return p.scale_dtype; },
      },
      params);
}

runtime::Result<std::vector<size_t>> compute_aux_storage_sizes(
    Span<const uint64_t> sizes,
    DType dtype,
    const QuantParams& params) {
  std::vector<size_t> result;

  ET_UNWRAP(num_scales, scale_element_count(sizes, params));
  const uint64_t scale_shape[] = {static_cast<uint64_t>(num_scales)};
  ET_UNWRAP(
      scale_bytes, compute_storage_size(scale_shape, scale_dtype_of(params)));
  result.push_back(scale_bytes);

  if (is_asymmetric(params)) {
    auto zp_bytes = num_scales * sizeof(int32_t);
    result.push_back(zp_bytes);
  }

  return result;
}

} // namespace executorch::backends::xnnpack::core
