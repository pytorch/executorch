//
// Serdes.hpp
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <string>
#import <vector>

#import <json.hpp>

#import <asset.h>
#import <metadata.h>

namespace executorchcoreml {

/// Populates `nlohmann::json` from `ModelAsset`.
///
/// @param json The json value to be populated from the asset value.
/// @param asset The asset value.
void to_json(nlohmann::json& json, const Asset& asset);

/// Populates `ModelAsset` from `nlohmann::json`.
///
/// @param json The json value.
/// @param asset The asset value to be populated from the json value.
void from_json(const nlohmann::json& json, Asset& asset);

/// Populates nlohmann::json`from `ModelMetadata`.
///
/// @param json The json value to be populated from the metadata value.
/// @param metadata The metadata value.
void to_json(nlohmann::json& json, const ModelMetadata& metadata);

/// Populates `ModelMetadata` from `nlohmann::json`.
///
/// @param json The json value.
/// @param metadata The metadata value to be populated from the json value.
void from_json(const nlohmann::json& json, ModelMetadata& metadata);

} // namespace executorchcoreml
