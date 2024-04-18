//
//  multiarray.mm
//  coremlexecutorch
//
//  Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <multiarray.h>

#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <functional>
#import <numeric>
#import <objc_array_util.h>
#import <optional>
#import <vector>

namespace  {
using namespace executorchcoreml;

// Returns BNNSDataLayout and sets strides from the multi-array strides.
///
/// BNNS requires strides to be non-decreasing order;
/// `bnns_strides[i] <= bnns_strides[i + 1]`. BNNSDataLayout defines
/// how each dimension is mapped to the stride.
///
/// @param multi_array_strides  The multiarray strides.
/// @param bnns_strides   The bnns strides.
/// @retval The `BNNSDataLayout`.
std::optional<BNNSDataLayout> get_bnns_data_layout(const std::vector<ssize_t>& multi_array_strides,
                                                   size_t *bnns_strides) {
    bool first_major = false;
    uint32_t rank = static_cast<uint32_t>(multi_array_strides.size());
    if (rank > BNNS_MAX_TENSOR_DIMENSION) {
        return std::nullopt;
    }
    
    if (std::is_sorted(multi_array_strides.begin(), multi_array_strides.end(), std::less())) {
        first_major = false;
        std::copy(multi_array_strides.begin(), multi_array_strides.end(), bnns_strides);
    } else if (std::is_sorted(multi_array_strides.begin(), multi_array_strides.end(), std::greater()) ) {
        first_major = true;
        std::copy(multi_array_strides.rbegin(), multi_array_strides.rend(), bnns_strides);
    } else {
        return std::nullopt;
    }
    
    // See BNNSDataLayout's raw value how this bitwise-or makes sense.
    return (BNNSDataLayout) (0x08000 +                    // flags as canonical first/last major type
                             0x10000 * rank +             // set dimensionality
                             (first_major ? 1 : 0));      // set first/last major bit
}

/// Returns `BNNSDataType` from `MultiArray::DataType`.
///
/// @param datatype  The multiarray datatype.
/// @retval The `BNNSDataType`.
std::optional<BNNSDataType> get_bnns_data_type(MultiArray::DataType datatype) {
    switch (datatype) {
        case MultiArray::DataType::Bool: {
            return BNNSDataTypeBoolean;
        }
        case MultiArray::DataType::Byte: {
            return BNNSDataTypeUInt8;
        }
        case MultiArray::DataType::Char: {
            return BNNSDataTypeInt8;
        }
        case MultiArray::DataType::Short: {
            return BNNSDataTypeInt16;
        }
        case MultiArray::DataType::Int32: {
            return BNNSDataTypeInt32;
        }
        case MultiArray::DataType::Int64: {
            return BNNSDataTypeInt64;
        }
        case MultiArray::DataType::Float16: {
            return BNNSDataTypeFloat16;
        }
        case MultiArray::DataType::Float32: {
            return BNNSDataTypeFloat32;
        }
        default: {
            return std::nullopt;
        }
    }
}

/// Initializes BNNS array descriptor from multi array.
///
/// @param bnns_descriptor   The descriptor to be initialized.
/// @param multi_array  The multiarray.
/// @retval `true` if the initialization succeeded otherwise `false`.
bool init_bnns_descriptor(BNNSNDArrayDescriptor& bnns_descriptor, const MultiArray& multi_array) {
    const auto& layout = multi_array.layout();
    if (layout.num_elements() == 1) {
        return false;
    }
    
    auto bnns_datatype = get_bnns_data_type(layout.dataType());
    if (!bnns_datatype) {
        return false;
    }
    
    std::memset(&bnns_descriptor, 0, sizeof(bnns_descriptor));
    auto bnns_layout = get_bnns_data_layout(layout.strides(), bnns_descriptor.stride);
    if (!bnns_layout) {
        return false;
    }
    
    const auto& shape = layout.shape();
    std::copy(shape.begin(), shape.end(), bnns_descriptor.size);
    bnns_descriptor.layout = bnns_layout.value();
    bnns_descriptor.data_scale = 1.0f;
    bnns_descriptor.data_bias = 0.0f;
    bnns_descriptor.data_type = bnns_datatype.value();
    bnns_descriptor.data = multi_array.data();
    
    return true;
}

bool copy_using_bnns(const MultiArray& src, MultiArray& dst) {
    if (dst.layout().num_bytes() < src.layout().num_bytes()) {
        return false;
    }
    BNNSNDArrayDescriptor src_descriptor;
    if (!init_bnns_descriptor(src_descriptor, src)) {
        return false;
    }
    
    BNNSNDArrayDescriptor dst_descriptor;
    if (!init_bnns_descriptor(dst_descriptor, dst)) {
        return false;
    }
    
    return BNNSCopy(&dst_descriptor, &src_descriptor, NULL) == 0;
}

std::vector<MultiArray::MemoryLayout> get_layouts(const std::vector<MultiArray>& arrays) {
    std::vector<MultiArray::MemoryLayout> result;
    result.reserve(arrays.size());
    
    std::transform(arrays.begin(), arrays.end(), std::back_inserter(result), [](const auto& array) {
        return array.layout();
    });
    
    return result;
}

std::vector<void *> get_datas(const std::vector<MultiArray>& arrays) {
    std::vector<void *> result;
    result.reserve(arrays.size());
    
    std::transform(arrays.begin(), arrays.end(), std::back_inserter(result), [](const auto& array) {
        return array.data();
    });
    
    return result;
}

// We can coalesce two adjacent dimensions if either dim has size 1 or if `shape[n] * stride[n] == stride[n + 1]`.
bool can_coalesce_dimensions(const std::vector<size_t>& shape,
                             const std::vector<ssize_t>& strides,
                             size_t dim1,
                             size_t dim2) {
    auto shape1 = shape[dim1];
    auto shape2 = shape[dim2];
    if (shape1 == 1 || shape2 == 1) {
        return true;
    }
    
    auto stride1 = strides[dim1];
    auto stride2 = strides[dim2];
    return shape1 * stride1 == stride2;
}

bool can_coalesce_dimensions(const std::vector<size_t>& shape,
                             const std::vector<std::vector<ssize_t>>& all_strides,
                             size_t dim1,
                             size_t dim2) {
    for (const auto& strides : all_strides) {
        if (!::can_coalesce_dimensions(shape, strides, dim1, dim2)) {
            return false;
        }
    }
    
    return true;
}

void update_strides(std::vector<std::vector<ssize_t>>& all_strides,
                    size_t dim1,
                    size_t dim2) {
    for (auto& strides : all_strides) {
        strides[dim1] = strides[dim2];
    }
}

std::vector<MultiArray::MemoryLayout> coalesce_dimensions(std::vector<MultiArray::MemoryLayout> layouts) {
    if (layouts.size() == 0) {
        return {};
    }
    
    std::vector<size_t> shape = layouts.back().shape();
    // reverse shape.
    std::reverse(shape.begin(), shape.end());
    std::vector<std::vector<ssize_t>> all_strides;
    // reverse strides.
    all_strides.reserve(layouts.size());
    std::transform(layouts.begin(), layouts.end(), std::back_inserter(all_strides), [](const MultiArray::MemoryLayout& layout) {
        auto strides = layout.strides();
        std::reverse(strides.begin(), strides.end());
        return strides;
    });
    size_t rank = layouts[0].rank();
    size_t prev_dim = 0;
    for (size_t dim = 1; dim < rank; ++dim) {
        if (::can_coalesce_dimensions(shape, all_strides, prev_dim, dim)) {
            if (shape[prev_dim] == 1) {
                ::update_strides(all_strides, prev_dim, dim);
            }
            shape[prev_dim] *= shape[dim];
        } else {
            ++prev_dim;
            if (prev_dim != dim) {
                ::update_strides(all_strides, prev_dim, dim);
                shape[prev_dim] = shape[dim];
            }
        }
    }
    
    if (rank == prev_dim + 1) {
        return layouts;
    }
    
    shape.resize(prev_dim + 1);
    for (auto& strides : all_strides) {
        strides.resize(prev_dim + 1);
    }
    
    std::vector<MultiArray::MemoryLayout> result;
    result.reserve(layouts.size());
    std::reverse(shape.begin(), shape.end());
    for (size_t i = 0; i < layouts.size(); ++i) {
        std::reverse(all_strides[i].begin(), all_strides[i].end());
        result.emplace_back(layouts[i].dataType(), shape, std::move(all_strides[i]));
    }
    
    return result;
}

enum class Direction : uint8_t {
    Forward = 0,
    Backward
};

void set_data_pointers(std::vector<void *>& data_pointers,
                       ssize_t index,
                       size_t dim,
                       Direction direction,
                       const std::vector<MultiArray::MemoryLayout>& layouts) {
    for (size_t i = 0; i < layouts.size(); ++i) {
        const auto& layout = layouts[i];
        const ssize_t stride = layout.strides()[dim];
        const size_t num_bytes = layout.num_bytes();
        ssize_t offset = 0;
        switch (direction) {
            case Direction::Forward: {
                offset = stride * index * num_bytes;
                break;
            }
            case Direction::Backward: {
                offset = - stride * index * num_bytes;
                break;
            }
        }
        data_pointers[i] = (void *)(static_cast<uint8_t *>(data_pointers[i]) + offset);
    }
}

void increment_data_pointers(std::vector<void *>& data_pointers,
                             size_t index,
                             size_t dim,
                             const std::vector<MultiArray::MemoryLayout>& layouts) {
    set_data_pointers(data_pointers, index, dim, Direction::Forward, layouts);
}

void decrement_data_pointers(std::vector<void *>& data_pointers,
                             size_t index,
                             size_t dim,
                             const std::vector<MultiArray::MemoryLayout>& layouts) {
    set_data_pointers(data_pointers, index, dim, Direction::Backward, layouts);
}

class MultiArrayIterator final {
public:
    explicit MultiArrayIterator(const std::vector<MultiArray>& arrays)
    :datas_(get_datas(arrays)), 
    layouts_(coalesce_dimensions(get_layouts(arrays)))
    {}
    
private:
    template<typename FN>
    void exec(FN&& fn, const std::vector<MultiArray::MemoryLayout>& layouts, std::vector<void *> datas, size_t n) {
        const auto& layout = layouts.back();
        // Avoid function call for rank <= 2.
        switch (n) {
            case 0: {
                break;
            }
            case 1: {
                for (size_t i = 0; i < layout.shape()[0]; ++i) {
                    ::increment_data_pointers(datas, i, 0, layouts);
                    fn(datas);
                    ::decrement_data_pointers(datas, i, 0, layouts);
                }
                break;
            }
            case 2: {
                for (size_t i = 0; i < layout.shape()[1]; ++i) {
                    ::increment_data_pointers(datas, i, 1, layouts);
                    for (size_t j = 0; j < layout.shape()[0]; ++j) {
                        ::increment_data_pointers(datas, j, 0, layouts);
                        fn(datas);
                        ::decrement_data_pointers(datas, j, 0, layouts);
                    }
                    ::decrement_data_pointers(datas, i, 1, layouts);
                }
                
                break;
            }
                
            default: {
                const size_t bound = layouts.back().shape()[n - 1];
                for (size_t index = 0; index < bound; ++index) {
                    ::increment_data_pointers(datas, index, n - 1, layouts);
                    exec(std::forward<FN>(fn), layouts, datas, n - 1);
                    ::decrement_data_pointers(datas, index, n - 1, layouts);
                }
            }
        }
    }
    
public:
    template<typename FN>
    void exec(FN&& fn) {
        std::vector<void *> datas = datas_;
        exec(fn, layouts_, datas, layouts_[0].rank());
    }
    
private:
    std::vector<void *> datas_;
    std::vector<MultiArray::MemoryLayout> layouts_;
};

/// BNNS has no double type, so we handle the conversions here.
template<typename T1, typename T2>
inline void copy_value(void *dst, const void *src) {
    const T2 *src_ptr = static_cast<const T2 *>(src);
    T1 *dst_ptr = static_cast<T1 *>(dst);
    *dst_ptr = static_cast<T1>(*src_ptr);
}

template<typename T>
void copy(void *dst,
          MultiArray::DataType dst_data_type,
          const void *src) {
    switch (dst_data_type) {
        case MultiArray::DataType::Bool: {
            ::copy_value<bool, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Byte: {
            ::copy_value<uint8_t, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Char: {
            ::copy_value<int8_t, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Short: {
            ::copy_value<int16_t, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Int32: {
            ::copy_value<int32_t, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Int64: {
            ::copy_value<int64_t, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Float16: {
            ::copy_value<_Float16, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Float32: {
            ::copy_value<float, T>(dst, src);
            break;
        }
            
        case MultiArray::DataType::Float64: {
            ::copy_value<double, T>(dst, src);
            break;
        }
    }
}

void copy(void *dst,
          MultiArray::DataType dst_data_type,
          const void *src,
          MultiArray::DataType src_data_type) {
    switch (src_data_type) {
        case MultiArray::DataType::Bool: {
            ::copy<uint8_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Byte: {
            ::copy<uint8_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Char: {
            ::copy<int8_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Short: {
            ::copy<int16_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Int32: {
            ::copy<int32_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Int64: {
            ::copy<int64_t>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Float16: {
            ::copy<_Float16>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Float32: {
            ::copy<float>(dst, dst_data_type, src);
            break;
        }
            
        case MultiArray::DataType::Float64: {
            ::copy<double>(dst, dst_data_type, src);
            break;
        }
    }
}

void copy(const MultiArray& src, MultiArray& dst, MultiArray::CopyOptions options) {
    if (options.use_bnns && copy_using_bnns(src, dst)) {
        return;
    }
    
    if (options.use_memcpy &&
        src.layout().dataType() == dst.layout().dataType() &&
        src.layout().is_packed() &&
        dst.layout().is_packed()) {
        std::memcpy(dst.data(), src.data(), src.layout().num_elements() * src.layout().num_bytes());
        return;
    }
    
    auto iterator = MultiArrayIterator({src, dst});
    iterator.exec([&](const std::vector<void *>& datas){
        void *src_data = datas[0];
        void *dst_data = datas[1];
        ::copy(dst_data, dst.layout().dataType(), src_data, src.layout().dataType());
    });
}

ssize_t get_data_offset(const std::vector<size_t>& indices, const std::vector<ssize_t>& strides) {
    ssize_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += static_cast<ssize_t>(indices[i]) * strides[i];
    }
    
    return offset;
}

ssize_t get_data_offset(size_t index, const std::vector<size_t>& shape, const std::vector<ssize_t>& strides) {
    size_t div = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());;
    size_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        div /= shape[i];
        size_t dim_index = index / div;
        offset += dim_index * strides[i];
        index %= div;
    }
    
    return offset;
}
}

namespace executorchcoreml {

size_t MultiArray::MemoryLayout::num_elements() const noexcept {
    if (shape_.size() == 0) {
        return 0;
    }
    
    return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
}

bool MultiArray::MemoryLayout::is_packed() const noexcept {
    if (strides_.size() < 2) {
        return true;
    }
    
    ssize_t expectedStride = 1;
    auto stridesIt = strides_.crbegin();
    for (auto shapeIt = shape_.crbegin(); shapeIt!= shape_.crend(); shapeIt++) {
        if (*stridesIt != expectedStride) {
            return false;
        }
        expectedStride = expectedStride * (*shapeIt);
        stridesIt++;
    }
    
    return true;
}

size_t MultiArray::MemoryLayout::num_bytes() const noexcept {
    switch (dataType()) {
        case MultiArray::DataType::Bool: {
            return 1;
        }
        case MultiArray::DataType::Byte: {
            return 1;
        }
        case MultiArray::DataType::Char: {
            return 1;
        }
        case MultiArray::DataType::Short: {
            return 2;
        }
        case MultiArray::DataType::Int32: {
            return 4;
        }
        case MultiArray::DataType::Int64: {
            return 8;
        }
        case MultiArray::DataType::Float16: {
            return 2;
        }
        case MultiArray::DataType::Float32: {
            return 4;
        }
        case MultiArray::DataType::Float64: {
            return 8;
        }
    }
}

void MultiArray::copy(MultiArray& dst, CopyOptions options) const noexcept {
    assert(layout().shape() == dst.layout().shape());
    ::copy(*this, dst, options);
}

std::optional<MLMultiArrayDataType> to_ml_multiarray_data_type(MultiArray::DataType data_type) {
    switch (data_type) {
        case MultiArray::DataType::Float16: {
            return MLMultiArrayDataTypeFloat16;
        }
        case MultiArray::DataType::Float32: {
            return MLMultiArrayDataTypeFloat32;
        }
        case MultiArray::DataType::Float64: {
            return MLMultiArrayDataTypeDouble;
        }
        case MultiArray::DataType::Int32: {
            return MLMultiArrayDataTypeInt32;
        }
        default: {
            return std::nullopt;
        }
    }
}

std::optional<MultiArray::DataType> to_multiarray_data_type(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeFloat16: {
            return MultiArray::DataType::Float16;
        }
        case MLMultiArrayDataTypeFloat32: {
            return MultiArray::DataType::Float32;
        }
        case MLMultiArrayDataTypeFloat64: {
            return MultiArray::DataType::Float64;
        }
        case MLMultiArrayDataTypeInt32: {
            return MultiArray::DataType::Int32;
        }
        default: {
            return std::nullopt;
        }
    }
}

void *MultiArray::data(const std::vector<size_t>& indices) const noexcept {
    assert(indices.size() == layout().shape().size());
    uint8_t *ptr = static_cast<uint8_t *>(data());
    ssize_t offset = ::get_data_offset(indices, layout().strides());
    return ptr + offset * layout().num_bytes();
}

void *MultiArray::data(size_t index) const noexcept {
    assert(index < layout().num_elements());
    uint8_t *ptr = static_cast<uint8_t *>(data());
    ssize_t offset = ::get_data_offset(index, layout().shape(), layout().strides());
    return ptr + offset * layout().num_bytes();
}

} // namespace executorchcoreml
