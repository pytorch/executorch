//
// inmemory_filesystem_metadata.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "memory_buffer.hpp"
#include "range.hpp"

namespace inmemoryfs {

struct InMemoryNodeMetadata {
    std::string name;
    size_t kind;
    Range data_region;
    std::unordered_map<std::string, size_t> child_name_to_indices_map;
};

struct InMemoryFileSystemMetadata {
    std::vector<InMemoryNodeMetadata> nodes;
};

} // namespace inmemoryfs
