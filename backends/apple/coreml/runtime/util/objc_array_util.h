//
//  objc_array_util.h
//  util
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>
#import <type_traits>
#import <vector>

namespace executorchcoreml {

template <typename T> T to_value(NSNumber* value);

template <> inline size_t to_value(NSNumber* value) { return value.unsignedLongValue; }

template <> inline ssize_t to_value(NSNumber* value) { return value.longLongValue; }

template <> inline int to_value(NSNumber* value) { return value.intValue; }

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline NSArray<NSNumber*>* to_array(const std::vector<T>& array) {
    NSMutableArray<NSNumber*>* result = [NSMutableArray arrayWithCapacity:array.size()];
    for (T value: array) {
        [result addObject:@(value)];
    }

    return result;
}

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline std::vector<T> to_vector(NSArray<NSNumber*>* numbers) {
    std::vector<T> result;
    result.reserve(numbers.count);
    for (NSNumber* number in numbers) {
        result.emplace_back(to_value<T>(number));
    }

    return result;
}

}
