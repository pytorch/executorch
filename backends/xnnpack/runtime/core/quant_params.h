#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>

#include <cstdint>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::core {

struct PerTensorQuant {
    DType scale_dtype = DType::Float32;
    float scale = 0.0f;
    int32_t zero_point = 0;

    bool operator==(const PerTensorQuant& o) const {
        return scale_dtype == o.scale_dtype
            && scale == o.scale
            && zero_point == o.zero_point;
    }
};

struct PerAxisQuant {
    int8_t axis;
    DType scale_dtype = DType::Float32;

    bool operator==(const PerAxisQuant& o) const {
        return axis == o.axis && scale_dtype == o.scale_dtype;
    }
};

struct BlockwiseQuant {
    int8_t axis;
    int32_t block_size;
    DType scale_dtype = DType::Float32;

    bool operator==(const BlockwiseQuant& o) const {
        return axis == o.axis
            && block_size == o.block_size
            && scale_dtype == o.scale_dtype;
    }
};

using QuantParams = std::variant<PerTensorQuant, PerAxisQuant, BlockwiseQuant>;

QuantParams qint8_per_channel_sym(int8_t axis);
QuantParams qint8_per_tensor_sym(float scale);
QuantParams quint8_per_tensor_asym(float scale, int32_t zero_point);
QuantParams quint8_per_token_asym(int8_t axis);
QuantParams qint4_blockwise_sym(int8_t axis, int32_t block_size);

bool is_quantized(DType dtype);
bool is_asymmetric(DType dtype);
size_t element_size(DType dtype);
uint8_t aux_buffer_count(DType dtype, const QuantParams& params);
std::vector<size_t> compute_aux_storage_sizes(
    Span<const uint64_t> sizes,
    DType dtype,
    const QuantParams& params);

}
