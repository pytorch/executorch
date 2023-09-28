//
// JSONKeyValueStore.h
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <key_value_store.hpp>

#include <iostream>
#include <sstream>

#include <json.hpp>

namespace executorchcoreml {
namespace sqlite {

using json = nlohmann::json;

/// JSON converter for a type `T`.
template <typename T>
struct JSONConverter {
    static constexpr StorageType storage_type = StorageType::Text;
    
    /// Converts a value of type `T` to a `sqlite::Value`.
    inline static sqlite::Value to_sqlite_value(const T& value) {
        json j;
        to_json(j, value);
        std::stringstream ss;
        ss << j;
        return ss.str();
    }
    
    /// Converts a `sqlite::UnOwnedValue` to a value of type `T`.
    inline static T from_sqlite_value(const sqlite::UnOwnedValue& value) {
        auto text = std::get<sqlite::UnOwnedString>(value);
        json j = json::parse(text.data, text.data + text.size);
        T result;
        from_json(j, result);
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
