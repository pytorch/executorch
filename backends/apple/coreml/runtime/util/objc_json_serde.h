//
//  objc_json_serde.h
//  util
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <Foundation/Foundation.h>

#import <map>
#import <string>
#import <unordered_map>
#import <vector>

#import <objc_safe_cast.h>

namespace executorchcoreml {
namespace serde {
namespace json {

inline NSString* to_string(std::string_view view) { return @(std::string(view).c_str()); }

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
T to_scalar(NSNumber* value);

template <> inline size_t to_scalar(NSNumber* value) { return value.unsignedLongLongValue; }

template <> inline int64_t to_scalar(NSNumber* value) { return value.longLongValue; }

id to_json_object(const std::string& json_string);

id to_json_object(NSData* data);

std::string to_json_string(id json_object);

template <typename T, typename Enable = void> struct Converter { };

template <typename T>
using is_vector =
    std::is_same<std::decay_t<T>,
                 std::vector<typename std::decay_t<T>::value_type, typename std::decay_t<T>::allocator_type>>;

template <typename T>
using is_unordered_map = std::is_same<std::decay_t<T>,
                                      std::unordered_map<typename std::decay_t<T>::key_type,
                                                         typename std::decay_t<T>::mapped_type,
                                                         typename std::decay_t<T>::hasher,
                                                         typename std::decay_t<T>::key_equal,
                                                         typename std::decay_t<T>::allocator_type>>;

template <typename T> struct Converter<T, typename std::enable_if<std::is_arithmetic_v<T>>::type> {
    template <typename U = T> static id to_json(U&& value) { return @(value); }

    static void from_json(id json_value, T& value) {
        NSNumber* json_number = SAFE_CAST(json_value, NSNumber);
        if (json_number) {
            value = to_scalar<T>(json_number);
        }
    }
};

template <typename T> struct Converter<T, typename std::enable_if<std::is_same_v<T, std::string>>::type> {
    template <typename U = T> static id to_json(U&& value) { return @(value.c_str()); }

    static void from_json(id json_value, T& value) {
        NSString* json_string = SAFE_CAST(json_value, NSString);
        if (json_string) {
            value = json_string.UTF8String;
        }
    }
};

template <typename T> struct Converter<T, typename std::enable_if<is_vector<T>::value>::type> {
    using value_type = typename T::value_type;

    template <typename U = T> static id to_json(U&& values) {
        NSMutableArray<id>* result = [NSMutableArray arrayWithCapacity:values.size()];
        for (auto it = values.begin(); it != values.end(); ++it) {
            [result addObject:Converter<value_type>::to_json(*it)];
        }

        return result;
    }

    static void from_json(id json_value, T& values) {
        NSArray<id>* json_values = SAFE_CAST(json_value, NSArray);
        for (id json_object in json_values) {
            value_type value;
            Converter<value_type>::from_json(json_object, value);
            values.emplace_back(std::move(value));
        }
    }
};

template <typename T> struct Converter<T, std::enable_if_t<is_unordered_map<T>::value>> {
    using value_type = typename T::mapped_type;
    using key_type = typename T::key_type;

    static_assert(std::is_same<key_type, std::string>::value, "key_type must be string");

    template <typename U = T> static id to_json(U&& values) {
        NSMutableDictionary<NSString*, id>* result = [NSMutableDictionary dictionaryWithCapacity:values.size()];
        for (auto it = values.begin(); it != values.end(); ++it) {
            result[@(it->first.c_str())] = Converter<value_type>::to_json(it->second);
        }

        return result;
    }

    static void from_json(id json_value, T& values) {
        NSDictionary<NSString*, id>* json_values = SAFE_CAST(json_value, NSDictionary);
        for (NSString* key in json_values) {
            value_type value;
            Converter<value_type>::from_json(json_values[key], value);
            values.emplace(std::string(key.UTF8String), std::move(value));
        }
    }
};

template <typename T> inline id to_json_value(T&& value) {
    return Converter<typename std::decay_t<T>>::to_json(std::forward<T>(value));
}

template <typename T> void from_json_value(id json_value, T& value) {
    return Converter<T>::from_json(json_value, value);
}

} // namespace serde
} // namespace json
} // namespace executorchcoreml
