//
// metadata_keys.hpp
// inmemoryfs
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <string>

namespace inmemoryfs {

struct InMemoryNodeMetadataKeys {
    constexpr static std::string_view kName = "name";
    constexpr static std::string_view kDataRegion = "dataRegion";
    constexpr static std::string_view kChildIndices = "children";
    constexpr static std::string_view kKind = "kind";
};

struct InMemoryFileSystemMetadataKeys {
    constexpr static std::string_view kNodes = "nodes";
};

struct RangeKeys {
    constexpr static std::string_view kOffset = "offset";
    constexpr static std::string_view kSize = "size";
};

} // namespace inmemoryfs
