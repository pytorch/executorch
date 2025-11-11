//
//  json_key_value_store.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include <json_key_value_store.hpp>

namespace executorchcoreml {
namespace sqlite {

std::string string_from_unowned_value(const sqlite::UnOwnedValue& value) {
    auto unowned_string = std::get<sqlite::UnOwnedString>(value);
    std::string result;
    result.assign(unowned_string.data, unowned_string.size);
    return result;
}


} // namespace sqlite
} // namespace executorchcoreml
