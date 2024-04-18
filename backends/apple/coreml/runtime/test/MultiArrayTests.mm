//
// MultiArrayTests.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <multiarray.h>
#import <objc_array_util.h>
#import <vector>

#import <XCTest/XCTest.h>

using namespace executorchcoreml;

namespace {
size_t get_buffer_size(const std::vector<size_t>& shape, const std::vector<ssize_t>& srides) {
    auto max_stride_it = std::max_element(srides.begin(), srides.end());
    size_t max_stride_axis = static_cast<size_t>(std::distance(srides.begin(), max_stride_it));
    size_t dimension_with_max_stride = shape[max_stride_axis];
    return dimension_with_max_stride * (*max_stride_it);
}

template<typename T>
MultiArray::DataType get_multiarray_data_type();

template<> MultiArray::DataType get_multiarray_data_type<float>() {
    return MultiArray::DataType::Float32;
}

template<> MultiArray::DataType get_multiarray_data_type<double>() {
    return MultiArray::DataType::Float64;
}

template<> MultiArray::DataType get_multiarray_data_type<int64_t>() {
    return MultiArray::DataType::Int64;
}

template<> MultiArray::DataType get_multiarray_data_type<int32_t>() {
    return MultiArray::DataType::Int32;
}

template<> MultiArray::DataType get_multiarray_data_type<int16_t>() {
    return MultiArray::DataType::Short;
}

template<> MultiArray::DataType get_multiarray_data_type<_Float16>() {
    return MultiArray::DataType::Float16;
}

template<typename T1, typename T2>
void verify_values(const MultiArray& multiarray1, const MultiArray& multiarray2) {
    for (size_t i = 0;  i < multiarray1.layout().num_elements(); ++i) {
        XCTAssertEqual(multiarray1.value<T1>(i), multiarray2.value<T2>(i));
    }
}

template<typename T>
MultiArray make_multi_array(const std::vector<size_t>& shape, const std::vector<ssize_t>& strides, std::vector<uint8_t>& storage) {
    storage.resize(get_buffer_size(shape, strides) * sizeof(T), 0);
    MultiArray::MemoryLayout layout(get_multiarray_data_type<T>(), shape, strides);
    return MultiArray(storage.data(), std::move(layout));
}

template<typename T>
MultiArray make_multi_array_and_fill(const std::vector<size_t>& shape, const std::vector<ssize_t>& strides, std::vector<uint8_t>& storage) {
    auto result = make_multi_array<T>(shape, strides, storage);
    for (size_t i = 0;  i < result.layout().num_elements(); ++i) {
        T value = static_cast<T>(i);
        result.set_value(i, value);
    }
    
    return result;
}

template<typename T1, typename T2>
void verify_copy_(const std::vector<size_t>& shape,
                  const std::vector<ssize_t>& src_strides,
                  const std::vector<ssize_t>& dst_strides) {
    std::vector<uint8_t> src_storage;
    auto src_multiarray = make_multi_array_and_fill<T1>(shape, src_strides, src_storage);
    
    std::vector<uint8_t> dst_storage;
    auto dst_multiarray = make_multi_array<T2>(shape, dst_strides, dst_storage);
    src_multiarray.copy(dst_multiarray, MultiArray::CopyOptions(true, false));
    verify_values<T1, T2>(src_multiarray, dst_multiarray);
    
    dst_storage.clear();
    dst_storage.resize(get_buffer_size(shape, dst_strides) * sizeof(T2), 0);
    src_multiarray.copy(dst_multiarray, MultiArray::CopyOptions(false, false));
    verify_values<T1, T2>(src_multiarray, dst_multiarray);
}

template<typename T1, typename T2>
void verify_copy(const std::vector<size_t>& shape,
                 const std::vector<ssize_t>& src_strides,
                 const std::vector<ssize_t>& dst_strides) {
    verify_copy_<T1, T2>(shape, src_strides, dst_strides);
    verify_copy_<T2, T1>(shape, src_strides, dst_strides);
}
} //namespace

@interface MultiArrayTests : XCTestCase

@end

@implementation MultiArrayTests

- (void)verifyDataCopyWithShape:(const std::vector<size_t>&)shape
                     srcStrides:(const std::vector<ssize_t>&)srcStrides
                     dstStrides:(const std::vector<ssize_t>&)dstStrides {
    verify_copy<int16_t, int32_t>(shape, srcStrides, dstStrides);
    verify_copy<int16_t, int64_t>(shape, srcStrides, dstStrides);
    verify_copy<int32_t, int64_t>(shape, srcStrides, dstStrides);
    verify_copy<float, double>(shape, srcStrides, srcStrides);
    verify_copy<float, _Float16>(shape, srcStrides, dstStrides);
    verify_copy<double, _Float16>(shape, srcStrides, srcStrides);
}

- (void)testAdjacentDataCopy {
    std::vector<size_t> shape = {1, 3, 10, 10};
    std::vector<ssize_t> strides = {3 * 10 * 10, 10 * 10, 10, 1};
    [self verifyDataCopyWithShape:shape srcStrides:strides dstStrides:strides];
}

- (void)testNonAdjacentDataCopy {
    std::vector<size_t> shape = {1, 3, 10, 10};
    std::vector<ssize_t> srcStrides = {3 * 10 * 64, 10 * 64, 64, 1};
    std::vector<ssize_t> dstStrides = {3 * 10 * 10 * 10, 10 * 10 * 10, 100, 10};
    [self verifyDataCopyWithShape:shape srcStrides:srcStrides dstStrides:dstStrides];
}

@end
