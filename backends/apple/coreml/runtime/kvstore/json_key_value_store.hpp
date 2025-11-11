//
// JSONKeyValueStore.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <key_value_store.hpp>

#include <iostream>
#include <sstream>

namespace executorchcoreml {
namespace sqlite {

/// Returns string from `sqlite::UnOwnedValue`
///
/// The method fails if the `value` does not contain `UnOwnedString`.
///
/// @param value The unowned value.
/// @retval The string.
std::string string_from_unowned_value(const sqlite::UnOwnedValue& value);

/// JSON converter for a type `T`.
template <typename T>
struct JSONConverter {
    static constexpr StorageType storage_type = StorageType::Text;
    
    /// Converts a value of type `T` to a `sqlite::Value`.
    inline static sqlite::Value to_sqlite_value(const T& value) noexcept {
        return value.to_json_string();
    }
    
    /// Converts a `sqlite::UnOwnedValue` to a value of type `T`.
    inline static T from_sqlite_value(const sqlite::UnOwnedValue& value) noexcept {
        T result;
        result.from_json_string(string_from_unowned_value(value));
        return result;
    }
};

/// Type representing a JSON KeyValue store, the `Value` type uses a JSON converter.
///
/// The `Value` type must implement `to_json(nlohmann::json& j, const T& value)` to convert
/// the value to json and `from_json(const nlohmann::json& j, T& value)` to convert the value from json.
/// The functions should be in the same namespace where the value is defined.
template<typename Key, typename Value, typename KeyConverter = Converter<Key>>
using JSONKeyValueStore = KeyValueStore<Key, Value, JSONConverter<Value>, KeyConverter>;

} // namespace sqlite
} // namespace executorchcoreml
