//
// Serdes.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <string>

namespace executorchcoreml {
struct Asset;
struct ModelMetadata;

namespace serde {
namespace json {

/// Serializes `Asset` to json.
///
/// @param asset The asset value.
/// @retval Serialized json.
std::string to_json_string(const executorchcoreml::Asset& asset);

/// Populates `Asset` from serialized json.
///
/// @param json_string  A json string.
/// @param asset The asset value to be populated from the json string.
void from_json_string(const std::string& json_string, executorchcoreml::Asset& asset);

/// Serializes `ModelMetadata` to json.
///
/// @param metdata The metadata value.
/// @retval Serialized json.
std::string to_json_string(const executorchcoreml::ModelMetadata& metdata);

/// Populates `ModelMetadata` from serialized json.
///
/// @param json_string  A json string.
/// @param metadata The metadata value to be populated from the json string.
void from_json_string(const std::string& json_string, executorchcoreml::ModelMetadata& metadata);

} // namespace json
} // namespace serde
} // namespace executorchcoreml
