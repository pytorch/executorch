//
// inmemory_filesystem_utils.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import "inmemory_filesystem_utils.hpp"

#import <iostream>
#import <sstream>
#import <unordered_map>

#import <inmemory_filesystem_metadata.hpp>
#import <inmemory_filesystem_metadata_keys.hpp>

#import <objc_json_serde.h>
#import <json_util.hpp>

namespace executorchcoreml {
namespace serde {
namespace json {

using namespace inmemoryfs;

template <>
struct Converter<MemoryRegion> {
    static id to_json(const MemoryRegion& region) {
        return @{
            to_string(MemoryRegionKeys::kOffset) : to_json_value(region.offset),
            to_string(MemoryRegionKeys::kSize) : to_json_value(region.size)
        };
    }
    
    static void from_json(id json, MemoryRegion& region) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(MemoryRegionKeys::kOffset)], region.offset);
        from_json_value(json_dict[to_string(MemoryRegionKeys::kSize)], region.size);
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
using namespace::inmemoryfs;

void write_metadata_to_stream(const InMemoryFileSystemMetadata& metadata, std::ostream& stream) {
    using namespace executorchcoreml::serde::json;
    std::string json_string = to_json_string(Converter<InMemoryFileSystemMetadata>::to_json(metadata));
    std::reverse(json_string.begin(), json_string.end());
    stream << json_string;
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

void serialize(const InMemoryFileSystem& file_system,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               std::ostream& ostream) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        ::write_metadata_to_stream(fs_metadata, stream);
    };
    
    file_system.serialize(canonical_path, alignment, metadata_writer, ostream);
}

size_t get_serialization_size(const InMemoryFileSystem& file_system,
                              const std::vector<std::string>& canonical_path,
                              size_t alignment) noexcept {
    InMemoryFileSystem::MetadataWriter metadata_writer = [](const InMemoryFileSystemMetadata& fs_metadata,
                                                            std::ostream& stream) {
        ::write_metadata_to_stream(fs_metadata, stream);
    };
    
    return file_system.get_serialization_size(canonical_path, alignment, metadata_writer);
}

std::unique_ptr<InMemoryFileSystem> make(const std::shared_ptr<MemoryBuffer>& buffer) noexcept {
    InMemoryFileSystem::MetadataReader metadata_reader = [](std::istream& stream) {
        return ::read_metadata_from_stream(stream);
    };
    
    return InMemoryFileSystem::make(buffer, metadata_reader);
}
} // namespace inmemoryfs
