//
// inmemory_filesystem_utils.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "inmemory_filesystem_utils.hpp"

#include <iostream>
#include <json.hpp>
#include <sstream>

#include <inmemory_filesystem_metadata.hpp>
#include <inmemory_filesystem_metadata_keys.hpp>

namespace inmemoryfs {

using json = nlohmann::json;

void to_json(json& j, const MemoryRegion& region) {
    j = json { { MemoryRegionKeys::kOffset, region.offset }, { MemoryRegionKeys::kSize, region.size } };
}

void from_json(const json& j, MemoryRegion& region) {
    if (j.contains(MemoryRegionKeys::kOffset)) {
        j.at(MemoryRegionKeys::kOffset).get_to(region.offset);
    }
    if (j.contains(MemoryRegionKeys::kSize)) {
        j.at(MemoryRegionKeys::kSize).get_to(region.size);
    }
}

void to_json(json& j, const InMemoryNodeMetadata& node) {
    j = json { { InMemoryNodeMetadataKeys::kName, node.name },
               { InMemoryNodeMetadataKeys::kDataRegion, node.data_region },
               { InMemoryNodeMetadataKeys::kChildIndices, node.child_name_to_indices_map },
               { InMemoryNodeMetadataKeys::kKind, static_cast<int>(node.kind) } };
}

void from_json(const json& j, InMemoryNodeMetadata& node) {
    if (j.contains(InMemoryNodeMetadataKeys::kName)) {
        j.at(InMemoryNodeMetadataKeys::kName).get_to(node.name);
    }
    if (j.contains(InMemoryNodeMetadataKeys::kDataRegion)) {
        j.at(InMemoryNodeMetadataKeys::kDataRegion).get_to(node.data_region);
    }
    if (j.contains(InMemoryNodeMetadataKeys::kChildIndices)) {
        j.at(InMemoryNodeMetadataKeys::kChildIndices).get_to(node.child_name_to_indices_map);
    }
    if (j.contains(InMemoryNodeMetadataKeys::kKind)) {
        j.at(InMemoryNodeMetadataKeys::kKind).get_to(node.kind);
    }
}

void to_json(json& j, const InMemoryFileSystemMetadata& fs) {
    j = json { { InMemoryFileSystemMetadataKeys::kNodes, fs.nodes } };
}

void from_json(const json& j, InMemoryFileSystemMetadata& fs) {
    if (j.contains(InMemoryFileSystemMetadataKeys::kNodes)) {
        j.at(InMemoryFileSystemMetadataKeys::kNodes).get_to(fs.nodes);
    }
}

static void write_metadata_to_stream(const InMemoryFileSystemMetadata& fs_metadata, std::ostream& stream) {
    std::stringstream ss;
    json value;
    to_json(value, fs_metadata);
    ss << value;
    std::string metadata_json = ss.str();
    // reverse it for writing
    std::reverse(metadata_json.begin(), metadata_json.end());
    stream << metadata_json;
}

void serialize(const InMemoryFileSystem& file_system,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               std::ostream& ostream) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        write_metadata_to_stream(fs_metadata, stream);
    };

    file_system.serialize(canonical_path, alignment, metadata_writer, ostream);
}

size_t get_serialization_size(const InMemoryFileSystem& file_system,
                              const std::vector<std::string>& canonical_path,
                              size_t alignment) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        write_metadata_to_stream(fs_metadata, stream);
    };

    return file_system.get_serialization_size(canonical_path, alignment, metadata_writer);
}

std::unique_ptr<InMemoryFileSystem> make(const std::shared_ptr<MemoryBuffer>& buffer) noexcept {
    InMemoryFileSystem::MetadataReader metadata_reader = [](std::istream& stream) {
        json metadata_json;
        nlohmann::detail::json_sax_dom_parser<json> sdp(metadata_json, true);
        if (!json::sax_parse(stream, &sdp, nlohmann::detail::input_format_t::json, false)) {
            return std::optional<InMemoryFileSystemMetadata>();
        }

        InMemoryFileSystemMetadata metadata;
        from_json(metadata_json, metadata);
        return std::optional<InMemoryFileSystemMetadata>(std::move(metadata));
    };

    return InMemoryFileSystem::make(buffer, metadata_reader);
}
} // namespace inmemoryfs
