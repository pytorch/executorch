//
// MLMultiArray+Copy.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <MLMultiArray_Copy.h>

#import <multiarray.h>

namespace {
using namespace executorchcoreml;

template<typename T>
T toValue(NSNumber *value);

template<> size_t toValue(NSNumber *value) {
    return value.unsignedLongValue;
}

template<> ssize_t toValue(NSNumber *value) {
    return value.longLongValue;
}

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
std::vector<T> to_vector(NSArray<NSNumber *> *numbers) {
    std::vector<T> result;
    result.reserve(numbers.count);
    for (NSNumber *number in numbers) {
        result.emplace_back(toValue<T>(number));
    }
    
    return result;
}

MultiArray::DataType to_multi_array_data_type(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeInt32: {
            return MultiArray::DataType::Int;
        }
        case MLMultiArrayDataTypeFloat: {
            return MultiArray::DataType::Float;
        }
        case MLMultiArrayDataTypeFloat16: {
            return MultiArray::DataType::Float16;
        }
        case MLMultiArrayDataTypeDouble: {
            return MultiArray::DataType::Double;
        }
    }
}

MultiArray to_multi_array(void *data,
                          MLMultiArrayDataType dataType,
                          NSArray<NSNumber *> *shape,
                          NSArray<NSNumber *> *strides) {
    auto layout = MultiArray::MemoryLayout(to_multi_array_data_type(dataType),
                                           to_vector<size_t>(shape),
                                           to_vector<ssize_t>(strides));
    return MultiArray(data, std::move(layout));
}
} //namespace

@implementation MLMultiArray (Copy)

- (void)copyInto:(MLMultiArray *)dstMultiArray {
    [self getBytesWithHandler:^(const void *srcBytes, __unused NSInteger srcSize) {
        [dstMultiArray getMutableBytesWithHandler:^(void *dstBytes, __unused NSInteger size, NSArray<NSNumber *> * strides) {
            auto src = ::to_multi_array(const_cast<void *>(srcBytes), self.dataType, self.shape, self.strides);
            auto dst = ::to_multi_array(dstBytes, dstMultiArray.dataType, dstMultiArray.shape, strides);
            src.copy(dst);
        }];
    }];
}

@end
