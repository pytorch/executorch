//
// inmemory_filesystem_utils.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "inmemory_filesystem_utils.hpp"

#import <iostream>
#import <sstream>
#import <unordered_map>

#import <Foundation/Foundation.h>

#import "inmemory_filesystem_metadata.hpp"
#import "inmemory_filesystem_metadata_keys.hpp"
#import "json_util.hpp"
#import "objc_json_serde.h"

namespace executorchcoreml {
namespace serde {
namespace json {

using namespace inmemoryfs;

template <>
struct Converter<Range> {
    static id to_json(const Range& range) {
        return @{
            to_string(RangeKeys::kOffset) : to_json_value(range.offset),
            to_string(RangeKeys::kSize) : to_json_value(range.size)
        };
    }

    static void from_json(id json, Range& range) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }

        from_json_value(json_dict[to_string(RangeKeys::kOffset)], range.offset);
        from_json_value(json_dict[to_string(RangeKeys::kSize)], range.size);
    }
};

template <>
struct Converter<InMemoryNodeMetadata> {
    static id to_json(const InMemoryNodeMetadata& node) {
        return @{
            to_string(InMemoryNodeMetadataKeys::kName) : to_json_value(node.name),
            to_string(InMemoryNodeMetadataKeys::kDataRegion) : to_json_value(node.data_region),
            to_string(InMemoryNodeMetadataKeys::kChildIndices) : to_json_value(node.child_name_to_indices_map),
            to_string(InMemoryNodeMetadataKeys::kKind) : to_json_value(node.kind)
        };
    }

    static void from_json(id json, InMemoryNodeMetadata& node) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }

        from_json_value(json_dict[to_string(InMemoryNodeMetadataKeys::kName)], node.name);
        from_json_value(json_dict[to_string(InMemoryNodeMetadataKeys::kDataRegion)], node.data_region);
        from_json_value(json_dict[to_string(InMemoryNodeMetadataKeys::kChildIndices)], node.child_name_to_indices_map);
        from_json_value(json_dict[to_string(InMemoryNodeMetadataKeys::kKind)], node.kind);
    }
};

template <>
struct Converter<InMemoryFileSystemMetadata> {
    static id to_json(const InMemoryFileSystemMetadata& fs) {
        return @{
            to_string(InMemoryFileSystemMetadataKeys::kNodes) : to_json_value(fs.nodes)
        };
    }

    static void from_json(id json, InMemoryFileSystemMetadata& fs) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }

        from_json_value(json_dict[to_string(InMemoryFileSystemMetadataKeys::kNodes)], fs.nodes);
    }
};

} // namespace json
} // namespace serde
} // namespace executorchcoreml

namespace {
using namespace inmemoryfs;

std::string serialize_metadata(const InMemoryFileSystemMetadata& metadata) {
    using namespace executorchcoreml::serde::json;
    std::string result = to_json_string(Converter<InMemoryFileSystemMetadata>::to_json(metadata));
    std::reverse(result.begin(), result.end());
    return result;
}

bool write_metadata_to_stream(const InMemoryFileSystemMetadata& metadata, std::ostream& stream) {
    auto content = serialize_metadata(metadata);
    return stream.write(content.data(), content.length()).good();
}

size_t write_metadata_to_buffer(const InMemoryFileSystemMetadata& metadata, void *dst) {
    auto content = serialize_metadata(metadata);
    std::memcpy(dst, content.data(), content.length());
    return content.length();
}

std::optional<InMemoryFileSystemMetadata> read_metadata_from_stream(std::istream& stream) {
    using namespace executorchcoreml::serde::json;
    auto json_object = executorchcoreml::json::read_object_from_stream(stream);
    if (!json_object) {
        return std::optional<InMemoryFileSystemMetadata>();
    }

    InMemoryFileSystemMetadata metadata;
    Converter<InMemoryFileSystemMetadata>::from_json(to_json_object(json_object.value()), metadata);
    return metadata;
}
} // namespace

namespace inmemoryfs {
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
               void *dst,
               std::error_code& ec) noexcept {
    InMemoryFileSystem::MetadataWriterInMemory metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                                    void *metadata_dst) {
        return ::write_metadata_to_buffer(fs_metadata, metadata_dst);
    };

    return file_system.serialize(canonical_path, alignment, metadata_writer, dst, ec);
}

size_t get_buffer_size_for_serialization(const InMemoryFileSystem& file_system,
                                         const std::vector<std::string>& canonical_path,
                                         size_t alignment) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        return ::write_metadata_to_stream(fs_metadata, stream);
    };

    return file_system.get_buffer_size_for_serialization(canonical_path, alignment, metadata_writer);
}

std::unique_ptr<InMemoryFileSystem> make_from_buffer(const std::shared_ptr<MemoryBuffer>& buffer) noexcept {
    InMemoryFileSystem::MetadataReader metadata_reader = [](std::istream& stream) {
        return ::read_metadata_from_stream(stream);
    };

    return InMemoryFileSystem::make_from_buffer(buffer, metadata_reader);
}
} // namespace inmemoryfs
