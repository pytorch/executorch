//
// model_metadata.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <map>
#import <string>
#import <vector>

#import "serde_json.h"

namespace executorchcoreml {

/// A struct representing per-method metadata (for multifunction models).
struct MethodMetadata {
    /// Constructs a `MethodMetadata` instance.
    /// @param input_names The input names for the method.
    /// @param output_names The output names for the method.
    inline MethodMetadata(std::vector<std::string> input_names, std::vector<std::string> output_names) noexcept
        : input_names(std::move(input_names)), output_names(std::move(output_names)) { }

    inline MethodMetadata() noexcept { }

    /// Input names of the method.
    std::vector<std::string> input_names;
    /// Output names of the method.
    std::vector<std::string> output_names;
};

/// A struct representing a model's metadata.
struct ModelMetadata {
    /// Constructs a `ModelMetada` instance (for single-method models).
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
        if (identifier.empty()) {
            return false;
        }

        if (is_multifunction()) {
            // For multifunction models, every method must have outputs
            for (const auto& [name, method]: methods) {
                if (method.output_names.empty()) {
                    return false;
                }
            }
            return true;
        }

        // For single-method models, check output_names directly
        return !output_names.empty();
    }

    /// Returns `true` if this is multifunction metadata (has methods).
    inline bool is_multifunction() const noexcept { return !methods.empty(); }

    /// Get metadata for a specific method. Returns nullptr if method not found.
    /// For single-method models, returns nullptr (use input_names/output_names directly).
    inline const MethodMetadata* get_method_metadata(const std::string& method_name) const noexcept {
        auto it = methods.find(method_name);
        if (it != methods.end()) {
            return &it->second;
        }
        return nullptr;
    }

    inline std::string to_json_string() const noexcept { return executorchcoreml::serde::json::to_json_string(*this); }

    inline void from_json_string(const std::string& json_string) noexcept {
        executorchcoreml::serde::json::from_json_string(json_string, *this);
    }

    /// Unique identifier.
    std::string identifier;
    /// Input names of the model (for single-method models).
    std::vector<std::string> input_names;
    /// Output names of the model (for single-method models).
    std::vector<std::string> output_names;
    /// Per-method metadata (for multifunction models).
    std::map<std::string, MethodMetadata> methods;
};
} // namespace executorchcoreml
