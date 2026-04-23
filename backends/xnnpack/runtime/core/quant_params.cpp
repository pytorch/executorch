#include <executorch/backends/xnnpack/runtime/core/quant_params.h>

#include <cstdlib>

namespace executorch::backends::xnnpack::core {

QuantParams qint8_per_channel_sym(int8_t axis) {
    return PerAxisQuant { .axis = axis };
}

QuantParams qint8_per_tensor_sym(float scale) {
    return PerTensorQuant { .scale = scale, .zero_point = 0 };
}

QuantParams quint8_per_tensor_asym(float scale, int32_t zero_point) {
    return PerTensorQuant { .scale = scale, .zero_point = zero_point };
}

QuantParams quint8_per_token_asym(int8_t axis) {
    return PerAxisQuant { .axis = axis };
}

QuantParams qint4_blockwise_sym(int8_t axis, int32_t block_size) {
    return BlockwiseQuant { .axis = axis, .block_size = block_size };
}

bool is_quantized(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return false;
        case DType::QInt8Sym:
        case DType::QInt4Sym:
        case DType::QInt32Sym:
        case DType::QUInt8Asym:
            return true;
    }
}

bool is_asymmetric(DType dtype) {
    switch (dtype) {
        case DType::QUInt8Asym:
            return true;
        default:
            return false;
    }
}

size_t element_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:    return 4;
        case DType::QInt8Sym:   return 1;
        case DType::QUInt8Asym: return 1;
        case DType::QInt32Sym:  return 4;
        default:
            abort();
    }
}

uint8_t aux_buffer_count(DType dtype, const QuantParams& params) {
    (void)params;
    if (!is_quantized(dtype)) return 0;

    uint8_t count = 1; // scales
    if (is_asymmetric(dtype)) count++; // zero_points
    return count;
}

static size_t scale_element_count(
    core::Span<const uint64_t> sizes,
    const QuantParams& params) {
    return std::visit(overloaded {
        [](const PerTensorQuant&) -> size_t {
            return 1;
        },
        [&](const PerAxisQuant& p) -> size_t {
            size_t count = 1;
            for (size_t i = 0; i < sizes.size(); i++) {
                if (i != static_cast<size_t>(p.axis)) count *= sizes[i];
            }
            return count;
        },
        [&](const BlockwiseQuant& p) -> size_t {
            auto axis = static_cast<size_t>(p.axis);
            size_t num_blocks = (sizes[axis] + p.block_size - 1) / p.block_size;
            size_t other_dims = 1;
            for (size_t i = 0; i < sizes.size(); i++) {
                if (i != axis) other_dims *= sizes[i];
            }
            return num_blocks * other_dims;
        },
    }, params);
}

static DType scale_dtype_of(const QuantParams& params) {
    return std::visit(overloaded {
        [](const PerTensorQuant& p) { return p.scale_dtype; },
        [](const PerAxisQuant& p) { return p.scale_dtype; },
        [](const BlockwiseQuant& p) { return p.scale_dtype; },
    }, params);
}

std::vector<size_t> compute_aux_storage_sizes(
    Span<const uint64_t> sizes,
    DType dtype,
    const QuantParams& params) {
    std::vector<size_t> result;

    auto num_scales = scale_element_count(sizes, params);
    auto scale_bytes = num_scales * element_size(scale_dtype_of(params));
    result.push_back(scale_bytes);

    if (is_asymmetric(dtype)) {
        auto zp_bytes = num_scales * sizeof(int32_t);
        result.push_back(zp_bytes);
    }

    return result;
}

}
