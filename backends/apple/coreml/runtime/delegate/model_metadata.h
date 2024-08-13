//
// model_metadata.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <string>
#import <vector>

#import <serde_json.h>

namespace executorchcoreml {

/// A struct representing a model's metadata.
struct ModelMetadata {
    /// Constructs a `ModelMetada` instance.
    /// @param identifier The unique identifier.
    /// @param input_names The input names for the model.
    /// @param output_names   The output names for the model.
    inline ModelMetadata(std::string identifier,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names) noexcept
        : identifier(std::move(identifier)), input_names(std::move(input_names)),
          output_names(std::move(output_names)) { }

    inline ModelMetadata() noexcept { }

    /// Returns `true` if the metadata is valid otherwise `false`.
    inline bool is_valid() const noexcept {
        return !identifier.empty() && !input_names.empty() && !output_names.empty();
    }

    inline std::string to_json_string() const noexcept { return executorchcoreml::serde::json::to_json_string(*this); }

    inline void from_json_string(const std::string& json_string) noexcept {
        executorchcoreml::serde::json::from_json_string(json_string, *this);
    }

    /// Unique identifier.
    std::string identifier;
    /// Input names of the model.
    std::vector<std::string> input_names;
    /// Output names of the model.
    std::vector<std::string> output_names;
};
} // namespace executorchcoreml
