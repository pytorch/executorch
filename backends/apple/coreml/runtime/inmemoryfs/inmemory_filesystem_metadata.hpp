//
// inmemory_filesystem_metadata.hpp
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <memory_buffer.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace inmemoryfs {

struct InMemoryNodeMetadata {
    std::string name;
    size_t kind;
    MemoryRegion data_region;
    std::unordered_map<std::string, size_t> child_name_to_indices_map;
};

struct InMemoryFileSystemMetadata {
    std::vector<InMemoryNodeMetadata> nodes;
};

} // namespace inmemoryfs

