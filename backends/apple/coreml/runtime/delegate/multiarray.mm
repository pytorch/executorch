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
#import <vector>

namespace  {
using namespace executorchcoreml;

template<typename T>
struct TypedMultiArray {
    explicit TypedMultiArray(T *data, MultiArray::MemoryLayout layout) noexcept
    :data(data), layout(std::move(layout))
    {}
    
    T *data;
    MultiArray::MemoryLayout layout;
};

#pragma mark - BNNS

template<typename T1, typename T2>
struct BNNSCopier {
    static bool supported() noexcept {
        return false;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dstNNSDesc) noexcept {}
};

// float -> _Float16
template<>
struct BNNSCopier<float, _Float16> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeFloat32;
        dst_bnns_desc->data_type = BNNSDataTypeFloat16;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

// float -> int32_t
template<>
struct BNNSCopier<float, int32_t> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeFloat32;
        dst_bnns_desc->data_type = BNNSDataTypeInt32;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

// _Float16 -> float
template<>
struct BNNSCopier<_Float16, float> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeFloat16;
        dst_bnns_desc->data_type = BNNSDataTypeFloat32;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

// _Float16 -> int32_t
template<>
struct BNNSCopier<_Float16, int32_t> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeFloat16;
        dst_bnns_desc->data_type = BNNSDataTypeInt32;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

// int32_t -> _Float16
template<>
struct BNNSCopier<int32_t, _Float16> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeInt32;
        dst_bnns_desc->data_type = BNNSDataTypeFloat16;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

// int32_t -> float
template<>
struct BNNSCopier<int32_t, float> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(BNNSNDArrayDescriptor *src_bnns_desc, BNNSNDArrayDescriptor *dst_bnns_desc) noexcept {
        src_bnns_desc->data_type = BNNSDataTypeInt32;
        dst_bnns_desc->data_type = BNNSDataTypeFloat32;
        BNNSCopy(src_bnns_desc, dst_bnns_desc, NULL);
    }
};

/// Returns BNNSDataLayout and sets strides from the multi-array strides.
///
/// BNNS requires strides to be non-decreasing order;
/// `bnns_strides[i] <= bnns_strides[i + 1]`. BNNSDataLayout defines
/// how each dimension is mapped to the stride.
///
/// @param multi_array_strides  The multiarray strides.
/// @param bnns_strides   The bnns strides.
/// @retval The `BNNSDataLayout`.
BNNSDataLayout get_bnns_data_layout(const std::vector<ssize_t>& multi_array_strides, size_t *bnns_strides) {
    uint32_t firstMajorFlag = 1;
    uint32_t rank = static_cast<uint32_t>(multi_array_strides.size());
    if (rank > BNNS_MAX_TENSOR_DIMENSION) {
        return (BNNSDataLayout)-1;
    }
    
    if (std::is_sorted(multi_array_strides.begin(), multi_array_strides.end(), std::less())) {
        firstMajorFlag = 0;
        std::copy(multi_array_strides.begin(), multi_array_strides.end(), bnns_strides);
    } else if (std::is_sorted(multi_array_strides.begin(), multi_array_strides.end(), std::greater()) ) {
        firstMajorFlag = 1;
        std::copy(multi_array_strides.rbegin(), multi_array_strides.rend(), bnns_strides);
    } else {
        return (BNNSDataLayout)-1;
    }
    
    // See BNNSDataLayout's raw value how this bitwise-or makes sense.
    return (BNNSDataLayout)((rank << 16) | (8 << 12) | firstMajorFlag);
}

/// Initializes BNNSNDArrayDescriptor for the shape and strides.
///
/// @param layout  The memory layout.
/// @param desc   The ``BNNSNDArrayDescriptor`  to be initialized.
/// @retval `true` if the initialization succeeded otherwise `false`.
bool init_bnns_array_descriptor(const MultiArray::MemoryLayout& layout, BNNSNDArrayDescriptor *desc) {
    BNNSDataLayout bnns_layout = get_bnns_data_layout(layout.strides(), desc->stride);
    if (bnns_layout == (BNNSDataLayout)-1) {
        return false;
    }
    
    std::memset(desc, 0, sizeof(*desc));
    const auto& shape = layout.shape();
    std::copy(shape.begin(), shape.end(), desc->size);
    desc->layout = bnns_layout;
    desc->data_scale = 1.0f;
    desc->data_bias = 0.0f;
    
    return true;
}

template<typename T1, typename T2>
struct MultiArrayBNNSCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        if (!BNNSCopier<T1, T2>::supported()) {
            return false;
        }
        
        BNNSNDArrayDescriptor src_bnns_array;
        BNNSNDArrayDescriptor dst_bnns_array;
        if (!init_bnns_array_descriptor(src.layout, &src_bnns_array) || !init_bnns_array_descriptor(dst.layout, &dst_bnns_array)) {
            return false;
        }
        
        BNNSCopier<T1, T2>::copy(&src_bnns_array, &dst_bnns_array);
        return true;
    }
};

#pragma mark - VImageCopier

bool init_vi_Buffer(const MultiArray::MemoryLayout& layout, vImage_Buffer *viBuf, size_t bytesPerScalar) {
    size_t rank = layout.rank();
    const auto& shape = layout.shape();
    const auto& strides = layout.strides();
    
    if (rank < 2) {
        // vImage path requires at least two dimensions.
        return false;
    }
    
    // vImage blitter requires first major and every dimension except row (shape[rank - 2]) is contiguous.
    if (!std::is_sorted(strides.begin(), strides.end(), std::greater())) {
        return false;
    }
    
    if (strides[rank - 1] != 1) {
        return false;
    }
    
    size_t height = std::accumulate(shape.begin(), shape.end() - 1, size_t(1), std::multiplies<size_t>());
    if (height * strides[rank - 2] != strides[0] * shape[0]) {
        return false;
    }
    
    size_t width = shape[rank - 1];
    size_t rowBytes = strides[rank - 2] * bytesPerScalar;
    
    viBuf->data = NULL;
    viBuf->height = height;
    viBuf->width = width;
    viBuf->rowBytes = rowBytes;
    
    return true;
}

template<typename T1, typename T2>
struct VImageCopier {
    static bool supported() noexcept {
        return false;
    }
    
    static void copy(vImage_Buffer *src_vi_buffer, vImage_Buffer *dst_vi_buffer) noexcept {}
};

template<typename T>
struct VImageCopier<T, T> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(vImage_Buffer *src_vi_buffer, vImage_Buffer *dst_vi_buffer) noexcept {
        vImageCopyBuffer(src_vi_buffer, dst_vi_buffer, sizeof(T), kvImageDoNotTile);
    }
};

// float -> _Float16
template <>
struct VImageCopier<float, _Float16> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(vImage_Buffer *src_vi_buffer, vImage_Buffer *dst_vi_buffer) noexcept {
        vImageConvert_PlanarFtoPlanar16F(src_vi_buffer, dst_vi_buffer, kvImageDoNotTile);
    }
};

// _Float16 -> float
template <>
struct VImageCopier<_Float16, float> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(vImage_Buffer *src_vi_buffer, vImage_Buffer *dst_vi_buffer) noexcept {
        vImageConvert_Planar16FtoPlanarF(src_vi_buffer, dst_vi_buffer, kvImageDoNotTile);
    }
};

template<typename T1, typename T2>
struct MultiArrayVImageCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        if (!VImageCopier<T1, T2>::supported()) {
            return false;
        }
        
        vImage_Buffer src_vi_buffer;
        vImage_Buffer dst_vi_buffer;
        if (!init_vi_Buffer(src.layout, &src_vi_buffer, sizeof(T1))) {
            return false;
        }
        
        if (!init_vi_Buffer(dst.layout, &dst_vi_buffer, sizeof(T2))) {
            return false;
        }
        
        VImageCopier<T1, T2>::copy(&src_vi_buffer, &dst_vi_buffer);
        return true;
    }
};

#pragma mark - VDSPCopier

template<typename T1, typename T2>
struct VDSPCopier {
    static bool supported() noexcept {
        return false;
    }
    
    static void copy(const T1 *src_data, T2 *dst_data, size_t num_elements) noexcept {}
};

// Double -> Float
template<>
struct VDSPCopier<double, float> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const double *src_data, float *dst_data, size_t num_elements) noexcept {
        vDSP_vdpsp(src_data, 1, dst_data, 1, num_elements);
    }
};

// Float -> Double
template<>
struct VDSPCopier<float, double> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const float *src_data, double *dst_data, size_t num_elements) noexcept {
        vDSP_vspdp(src_data, 1, dst_data, 1, num_elements);
    }
};

// Float -> Int32
template<>
struct VDSPCopier<float, int32_t> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const float *src_data, int32_t *dst_data, size_t num_elements) noexcept {
        vDSP_vfix32(src_data, 1, dst_data, 1, num_elements);
    }
};

// Int32 -> Double
template<>
struct VDSPCopier<int32_t, double> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const int32_t *src_data, double *dst_data, size_t num_elements) noexcept {
        vDSP_vflt32D(src_data, 1, dst_data, 1, num_elements);
    }
};

// Int32 -> Float
template<>
struct VDSPCopier<int32_t, float> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const int32_t *src_data, float *dst_data, size_t num_elements) noexcept {
        vDSP_vflt32(src_data, 1, dst_data, 1, num_elements);
    }
};

template<typename T1, typename T2>
struct MultiArrayVDSPCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        if (!VDSPCopier<T1, T2>::supported()) {
            return false;
        }
        
        if (!src.layout.is_packed() || !dst.layout.is_packed()) {
            return false;
        }
        
        VDSPCopier<T1, T2>::copy(src.data, dst.data, src.layout.get_num_elements());
        return true;
    }
};

#pragma mark - MemCopy

template<typename T1, typename T2>
struct MemCopier {
    static bool supported() noexcept {
        return false;
    }
    
    static void copy(const T1 *src_data, T2 *dst_data, size_t num_elements) noexcept {}
};

template<typename T>
struct MemCopier<T, T> {
    static bool supported() noexcept {
        return true;
    }
    
    static void copy(const T *src_data, T *dst_data, size_t num_elements) noexcept {
        std::memcpy(dst_data, src_data, num_elements);
    }
};

template<typename T1, typename T2>
struct MultiArrayMemCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        if (!MemCopier<T1, T2>::supported()) {
            return false;
        }
        
        if (!src.layout.is_packed() || !dst.layout.is_packed()) {
            return false;
        }
        
        MemCopier<T1, T2>::copy(src.data, dst.data, src.layout.get_num_elements());
        return true;
    }
};

#pragma mark - MultiArrayIterator
/// TODO - remove recursion and coalesce contiguous dimensions.
template <typename T1, typename T2>
struct MultiArrayIterator {
    explicit MultiArrayIterator(TypedMultiArray<T1>& array1, TypedMultiArray<T2>& array2)
    :array1(array1), array2(array2)
    {}
    
    template<typename FN>
    void loop(FN&& fn, T1 *data1, T2 *data2, size_t dim) {
        const size_t index = dim - 1;
        const auto& layout1 = array1.layout;
        const auto& layout2 = array2.layout;
        const ssize_t stride1 = layout1.strides()[index];
        const ssize_t stride2 = layout2.strides()[index];
        const size_t bound = layout1.shape()[index];
        
        if (index == 0) {
            for (size_t i = 0; i < bound; i++) {
                if (fn(data1 + stride1 * i, data2 + stride2 * i)) {
                    break;
                }
            }
            return;
        }
        
        for (size_t i = 0; i < bound; i++) {
            loop(fn, data1 + stride1 * i, data2 + stride2 * i, dim - 1);
        }
    }
    
    template<typename FN>
    void loop(FN&& fn) {
        loop(fn, array1.data, array2.data, array1.layout.rank());
    }
    
    TypedMultiArray<T1> array1;
    TypedMultiArray<T2> array2;
};

template<typename T1, typename T2>
struct MultiArrayLoopingCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        auto looper  = MultiArrayIterator<T1, T2>(src, dst);
        looper.loop([](T1 *src, T2 *dst){
            *dst = static_cast<T2>(*src);
            return true;
        });
        
        return true;
    }
};

template <typename T1, typename T2>
struct MultiArrayCopier {
    static bool copy(TypedMultiArray<T1>& src, TypedMultiArray<T2>& dst) {
        if (src.layout.shape() != dst.layout.shape()) {
            return false;
        }
        
        if (src.layout.get_num_elements() == 0) {
            return true;
        }
        
        if (MultiArrayBNNSCopier<T1, T2>::copy(src, dst)) {
            return true;
        }
        
        if (MultiArrayVImageCopier<T1, T2>::copy(src, dst)) {
            return true;
        }
        
        if (MultiArrayVDSPCopier<T1, T2>::copy(src, dst)) {
            return true;
        }
        
        if (MultiArrayMemCopier<T1, T2>::copy(src, dst)) {
            return true;
        }
        
        return MultiArrayLoopingCopier<T1, T2>::copy(src, dst);
    }
};

template <typename T>
bool copy(TypedMultiArray<T>& src, MultiArray& dst) {
    const auto& dstLayout = dst.layout();
    switch (dstLayout.dataType()) {
        case MultiArray::DataType::Int: {
            auto dst_array = TypedMultiArray<int32_t>(reinterpret_cast<int32_t *>(dst.data()), dstLayout);
            return MultiArrayCopier<T, int32_t>::copy(src, dst_array);
        }
            
        case MultiArray::DataType::Float16: {
            auto dst_array = TypedMultiArray<_Float16>(reinterpret_cast<_Float16 *>(dst.data()), dstLayout);
            return MultiArrayCopier<T, _Float16>::copy(src, dst_array);
        }
            
        case MultiArray::DataType::Float: {
            auto dst_array = TypedMultiArray<float>(reinterpret_cast<float *>(dst.data()), dstLayout);
            return MultiArrayCopier<T, float>::copy(src, dst_array);
        }
            
        case MultiArray::DataType::Double: {
            auto dst_array = TypedMultiArray<double>(reinterpret_cast<double *>(dst.data()), dstLayout);
            return MultiArrayCopier<T, double>::copy(src, dst_array);
        }
    }
}
} //namespace

namespace executorchcoreml {

size_t MultiArray::MemoryLayout::get_num_elements() const noexcept {
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
    }
    
    return true;
}

bool MultiArray::copy(MultiArray& dst) const noexcept {
    switch (layout().dataType()) {
        case MultiArray::DataType::Int: {
            auto src = TypedMultiArray<int32_t>(reinterpret_cast<int32_t *>(data()), layout());
            return ::copy(src, dst);
        }
            
        case MultiArray::DataType::Float16: {
            auto src = TypedMultiArray<_Float16>(reinterpret_cast<_Float16 *>(data()), layout());
            return ::copy(src, dst);
        }
            
        case MultiArray::DataType::Float: {
            auto src = TypedMultiArray<float>(reinterpret_cast<float *>(data()), layout());
            return ::copy(src, dst);
        }
            
        case MultiArray::DataType::Double: {
            auto src = TypedMultiArray<double>(reinterpret_cast<double *>(data()), layout());
            return ::copy(src, dst);
        }
    }
}
} // namespace executorchcoreml
