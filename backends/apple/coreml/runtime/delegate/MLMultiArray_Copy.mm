//
// MLMultiArray+Copy.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <MLMultiArray_Copy.h>

#import <objc_array_util.h>
#import <multiarray.h>

namespace {
using namespace executorchcoreml;

MultiArray to_multi_array(void *data,
                          MLMultiArrayDataType dataType,
                          NSArray<NSNumber *> *shape,
                          NSArray<NSNumber *> *strides) {
    auto layout = MultiArray::MemoryLayout(to_multiarray_data_type(dataType).value(),
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
