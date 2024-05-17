//
// inmemory_filesystem_utils.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "inmemory_filesystem_utils.hpp"

#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "inmemory_filesystem_metadata.hpp"
#include "inmemory_filesystem_metadata_keys.hpp"
#include "json_util.hpp"

namespace inmemoryfs {

using json = nlohmann::json;

void to_json(json& j, const Range& range) {
    j = json { { RangeKeys::kOffset, range.offset }, { RangeKeys::kSize, range.size } };
}

void from_json(const json& j, Range& range) {
    if (j.contains(RangeKeys::kOffset)) {
        j.at(RangeKeys::kOffset).get_to(range.offset);
    }
    if (j.contains(RangeKeys::kSize)) {
        j.at(RangeKeys::kSize).get_to(range.size);
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

static std::string serialize_metadata(const InMemoryFileSystemMetadata& fs_metadata) {
    std::stringstream ss;
    json value;
    to_json(value, fs_metadata);
    ss << value;
    std::string result = ss.str();
    // reverse it for writing
    std::reverse(result.begin(), result.end());
    return result;
}

static bool write_metadata_to_stream(const InMemoryFileSystemMetadata& fs_metadata, std::ostream& stream) {
    auto content = serialize_metadata(fs_metadata);
    stream << content;
    return stream.good();
}

static size_t write_metadata_to_buffer(const InMemoryFileSystemMetadata& fs_metadata, void* dst) {
    auto content = serialize_metadata(fs_metadata);
    std::memcpy(dst, content.data(), content.length());
    return content.length();
}

bool serialize(const InMemoryFileSystem& file_system,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               std::ostream& ostream,
               std::error_code& ec) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        write_metadata_to_stream(fs_metadata, stream);
        return true;
    };

    return file_system.serialize(canonical_path, alignment, metadata_writer, ostream, ec);
}

bool serialize(const InMemoryFileSystem& file_system,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               void* dst,
               std::error_code& ec) noexcept {
    InMemoryFileSystem::MetadataWriterInMemory metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                                    void* metadata_dst) {
        return write_metadata_to_buffer(fs_metadata, metadata_dst);
    };

    return file_system.serialize(canonical_path, alignment, metadata_writer, dst, ec);
}

size_t get_buffer_size_for_serialization(const InMemoryFileSystem& file_system,
                                         const std::vector<std::string>& canonical_path,
                                         size_t alignment) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        return write_metadata_to_stream(fs_metadata, stream);
    };

    return file_system.get_buffer_size_for_serialization(canonical_path, alignment, metadata_writer);
}

std::unique_ptr<InMemoryFileSystem> make_from_buffer(const std::shared_ptr<MemoryBuffer>& buffer) noexcept {
    InMemoryFileSystem::MetadataReader metadata_reader = [](std::istream& stream) {
        auto json_object = executorchcoreml::json::read_object_from_stream(stream);
        if (!json_object) {
            return std::optional<InMemoryFileSystemMetadata>();
        }

        json metadata_json = json::parse(json_object.value());
        InMemoryFileSystemMetadata metadata;
        from_json(metadata_json, metadata);
        return std::optional<InMemoryFileSystemMetadata>(std::move(metadata));
    };

    return InMemoryFileSystem::make_from_buffer(buffer, metadata_reader);
}
} // namespace inmemoryfs
