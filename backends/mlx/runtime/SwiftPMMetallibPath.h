// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace executorch::backends::mlx {

// Finds the current platform's metallib under the SwiftPM resource bundle in
// one of the supplied container directories. Exposed for focused path tests.
std::optional<std::string> find_swiftpm_metallib_path(
    const std::vector<std::string>& container_paths);

// Resolves the current platform's metallib from loaded Apple bundles.
std::optional<std::string> resolve_swiftpm_metallib_path();

} // namespace executorch::backends::mlx
