//
// inmemory_filesystem.cpp
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "inmemory_filesystem.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

#include <json.hpp>
#include <reversed_memory_stream.hpp>

namespace {
using namespace inmemoryfs;

class InMemoryFileNode : public InMemoryFileSystem::InMemoryNode {
public:
    InMemoryFileNode(std::string name,
                     InMemoryFileSystem::Attributes attributes,
                     std::shared_ptr<MemoryBuffer> buffer) noexcept
        : InMemoryNode(std::move(name), std::move(attributes), InMemoryNode::Kind::File), buffer_(std::move(buffer)) { }

    InMemoryFileNode(InMemoryFileNode const&) = delete;
    InMemoryFileNode& operator=(InMemoryFileNode const&) = delete;

    inline std::shared_ptr<MemoryBuffer> getBuffer() const noexcept { return buffer_; }

private:
    const std::shared_ptr<MemoryBuffer> buffer_;
};

class InMemoryDirectoryNode : public InMemoryFileSystem::InMemoryNode {
public:
    using ItemsType = std::unordered_map<std::string, std::unique_ptr<InMemoryNode>>;

    InMemoryDirectoryNode(std::string name, InMemoryFileSystem::Attributes attributes, ItemsType items) noexcept
        : InMemoryNode(std::move(name), std::move(attributes), InMemoryNode::Kind::Directory),
          items_(std::move(items)) { }

    InMemoryDirectoryNode(std::string name, InMemoryFileSystem::Attributes attributes) noexcept
        : InMemoryNode(std::move(name), std::move(attributes), InMemoryNode::Kind::Directory), items_(ItemsType()) { }

    InMemoryDirectoryNode(InMemoryDirectoryNode const&) = delete;
    InMemoryDirectoryNode& operator=(InMemoryDirectoryNode const&) = delete;

    inline void add_item(const std::string& name, std::unique_ptr<InMemoryNode> node) noexcept {
        items_[name] = std::move(node);
    }

    inline void remove_item(const std::string& name) noexcept { items_.erase(name); }

    inline const ItemsType& get_items() const noexcept { return items_; }

    inline bool contains(const std::string& key) const noexcept { return items_.find(key) != items_.end(); }

    InMemoryNode* get_item(const std::string& key) noexcept {
        auto it = items_.find(key);
        if (it == items_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

private:
    ItemsType items_;
};

template <typename Iter, typename Container> inline bool is_last(Iter it, const Container& container) {
    return (it != container.end()) && (next(it) == container.end());
}

inline size_t align(size_t length, size_t alignment) { return ((length + (alignment - 1)) / alignment) * alignment; }

InMemoryFileSystem::InMemoryNode* get_node(InMemoryFileSystem::InMemoryNode* node,
                                           std::vector<std::string>::const_iterator path_start,
                                           std::vector<std::string>::const_iterator path_end) {
    for (auto it = path_start; it != path_end; ++it) {
        if (!node) {
            return nullptr;
        }
        const std::string& component = *it;
        InMemoryDirectoryNode* directory_node = static_cast<InMemoryDirectoryNode*>(node);
        node = directory_node->get_item(component);
    }

    return node;
}

std::string toString(time_t time) {
    constexpr auto format = "%Y-%m-%dT%TZ";
    std::stringstream stream;
    stream << std::put_time(gmtime(&time), format);
    return stream.str();
}

time_t toTime(const std::string& str) {
    constexpr auto format = "%Y-%m-%dT%TZ";
    time_t time = (time_t)(-1);
    std::stringstream stream(str);
    stream >> std::get_time(gmtime(&time), format);
    return time;
}

template <typename TP> std::time_t toTime(TP tp) {
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now() + system_clock::now());
    return system_clock::to_time_t(sctp);
}

InMemoryFileSystem::Attributes get_file_attributes(const std::filesystem::path& path) {
    auto attributes = InMemoryFileSystem::Attributes();
    auto modificationTime = toTime(std::filesystem::last_write_time(path));
    attributes.modificationTime = modificationTime;

    return attributes;
}

std::unique_ptr<InMemoryFileSystem::InMemoryNode> make_file_node(const std::filesystem::path& path,
                                                                 std::error_code& error) {
    error.clear();
    auto name = path.filename().string();
    auto file_path = path.string();
    auto buffer = MemoryBuffer::read_file_content(file_path, MemoryBuffer::ReadOption::Any, error);
    if (error) {
        return nullptr;
    }

    auto attributes = get_file_attributes(path);
    return std::make_unique<InMemoryFileNode>(std::move(name), std::move(attributes), std::move(buffer));
}

std::unique_ptr<InMemoryFileSystem::InMemoryNode> make_directory_node(const std::filesystem::path& path,
                                                                      std::error_code& error) {
    auto name = path.filename();
    std::unordered_map<std::string, std::unique_ptr<InMemoryFileSystem::InMemoryNode>> items = {};
    for (const std::filesystem::directory_entry& entry: std::filesystem::directory_iterator(path)) {
        if (!entry.exists()) {
            continue;
        }

        auto itemPath = std::filesystem::canonical(entry.path());
        auto itemName = itemPath.filename().string();
        std::unique_ptr<InMemoryFileSystem::InMemoryNode> node;
        if (entry.is_directory()) {
            node = make_directory_node(itemPath, error);
        } else if (!entry.is_directory()) {
            node = make_file_node(itemPath, error);
        }

        if (node) {
            items[itemName] = std::move(node);
        }
    }
    auto attributes = get_file_attributes(path);
    return std::make_unique<InMemoryDirectoryNode>(std::move(name), std::move(attributes), std::move(items));
}

std::unique_ptr<InMemoryFileSystem::InMemoryNode> make_node(const std::filesystem::path& path, std::error_code& error) {
    error.clear();
    auto status = std::filesystem::exists(path, error);
    if (!status || error) {
        return nullptr;
    }

    auto name = path.filename();
    bool isDirectory = std::filesystem::is_directory(path, error);
    if (error) {
        return nullptr;
    }

    if (isDirectory) {
        return make_directory_node(path, error);
    }

    return make_file_node(path, error);
}

bool write_node(InMemoryFileSystem::InMemoryNode* node,
                const std::filesystem::path& dst_path,
                bool recursive,
                std::error_code& error);

bool write_file_node(InMemoryFileNode* node, const std::filesystem::path& dst_path, std::error_code& error) {
    error.clear();
    std::filesystem::path file_path = dst_path;
    file_path.append(node->name());
    std::ofstream stream;
    stream.open(file_path, std::ofstream::out);
    if (!stream.good()) {
        error = std::error_code(errno, std::system_category());
        return false;
    }
    auto buffer = node->getBuffer().get();
    if (buffer->size() > 0) {
        char* bufferPtr = static_cast<char*>(buffer->data());
        stream.write(bufferPtr, static_cast<std::streamsize>(buffer->size()));
    }
    if (!stream.good()) {
        error = std::error_code(errno, std::system_category());
    }
    stream.close();

    return !error;
}

bool write_directory_node(InMemoryDirectoryNode* node,
                          const std::filesystem::path& dst_path,
                          bool recursive,
                          std::error_code& error) {
    error.clear();
    std::filesystem::path dir_path = dst_path;
    dir_path.append(node->name());
    if (!std::filesystem::create_directory(dir_path, error)) {
        return false;
    }

    for (const auto& [_, node]: node->get_items()) {
        if (node.get()->isDirectory() && !recursive) {
            continue;
        }
        if (!write_node(node.get(), dir_path, recursive, error)) {
            return false;
        }
    }

    return true;
}

bool write_node(InMemoryFileSystem::InMemoryNode* node,
                const std::filesystem::path& dst_path,
                bool recursive,
                std::error_code& error) {
    switch (node->kind()) {
        case InMemoryFileSystem::InMemoryNode::Kind::Directory:
            return write_directory_node(static_cast<InMemoryDirectoryNode*>(node), dst_path, recursive, error);

        case InMemoryFileSystem::InMemoryNode::Kind::File:
            return write_file_node(static_cast<InMemoryFileNode*>(node), dst_path, error);
    }
}
} // namespace

namespace inmemoryfs {
using json = nlohmann::json;

void to_json(json& j, const MemoryRegion& region) { j = json { { "offset", region.offset }, { "size", region.size } }; }

void from_json(const json& j, MemoryRegion& region) {
    j.at("offset").get_to(region.offset);
    j.at("size").get_to(region.size);
}

void to_json(json& j, const InMemoryFileSystem::Attributes& attributes) {
    j = json { { "modificationTime", toString(attributes.modificationTime) } };
}

void from_json(const json& j, InMemoryFileSystem::Attributes& attributes) {
    const std::string& modificationTime = j.at("modificationTime");
    attributes.modificationTime = toTime(modificationTime);
}

struct Attributes {
    time_t creation_time;
    time_t modification_time;

    inline Attributes() noexcept : creation_time(time(0)), modification_time(time(0)) { }
};
} // namespace inmemoryfs

namespace serdes {
using json = nlohmann::json;

struct FlattenedInMemoryNodeKeys {
    constexpr static std::string_view kName = "name";
    constexpr static std::string_view kDataRegion = "dataRegion";
    constexpr static std::string_view kChildren = "children";
    constexpr static std::string_view kKind = "kind";
};

struct FlattenedInMemoryNode {
    std::string name;
    InMemoryFileSystem::InMemoryNode::Kind kind;
    InMemoryFileSystem::Attributes attributes;
    std::unordered_map<std::string, size_t> children;
    MemoryRegion data_region;
    InMemoryFileNode* file_node = nullptr;

    static std::vector<FlattenedInMemoryNode> flatten(InMemoryFileSystem::InMemoryNode* node,
                                                      size_t alignment) noexcept;

    static std::unique_ptr<InMemoryFileSystem::InMemoryNode>
    unflatten(const std::vector<FlattenedInMemoryNode>& nodes, const std::shared_ptr<MemoryBuffer>& data) noexcept;

private:
    static void populate(InMemoryFileSystem::InMemoryNode* node,
                         std::unordered_map<InMemoryFileSystem::InMemoryNode*, size_t>& node_to_index_map,
                         size_t alignment,
                         std::vector<FlattenedInMemoryNode>& result) noexcept;

    static size_t get_next_offset(const std::vector<FlattenedInMemoryNode>& nodes) noexcept;
};

size_t FlattenedInMemoryNode::get_next_offset(const std::vector<FlattenedInMemoryNode>& nodes) noexcept {
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        if (it->kind == InMemoryFileSystem::InMemoryNode::Kind::File) {
            return it->data_region.get_length() + 1;
        }
    }

    return 0;
}

void FlattenedInMemoryNode::populate(InMemoryFileSystem::InMemoryNode* node,
                                     std::unordered_map<InMemoryFileSystem::InMemoryNode*, size_t>& node_to_index_map,
                                     size_t alignment,
                                     std::vector<FlattenedInMemoryNode>& result) noexcept {
    FlattenedInMemoryNode flattened_node;
    flattened_node.name = node->name();
    flattened_node.kind = node->kind();
    switch (node->kind()) {
        case InMemoryFileSystem::InMemoryNode::Kind::File: {
            size_t index = result.size();
            InMemoryFileNode* file_node = static_cast<InMemoryFileNode*>(node);
            size_t offset = align(FlattenedInMemoryNode::get_next_offset(result), alignment);
            auto buffer = file_node->getBuffer();
            size_t size = buffer->size();
            flattened_node.data_region = MemoryRegion(offset, size);
            flattened_node.file_node = file_node;
            result.emplace_back(std::move(flattened_node));
            node_to_index_map[node] = index;
            break;
        }
        case InMemoryFileSystem::InMemoryNode::Kind::Directory: {
            InMemoryDirectoryNode* directory_node = static_cast<InMemoryDirectoryNode*>(node);
            for (const auto& [key, item]: directory_node->get_items()) {
                populate(item.get(), node_to_index_map, alignment, result);
                flattened_node.children[key] = node_to_index_map[item.get()];
            }
            node_to_index_map[node] = result.size();
            result.emplace_back(std::move(flattened_node));
            break;
        }
    }
}

std::vector<FlattenedInMemoryNode> FlattenedInMemoryNode::flatten(InMemoryFileSystem::InMemoryNode* node,
                                                                  size_t alignment) noexcept {
    std::unordered_map<InMemoryFileSystem::InMemoryNode*, size_t> node_to_index_map;
    std::vector<FlattenedInMemoryNode> result;
    populate(node, node_to_index_map, alignment, result);

    return result;
}

std::unique_ptr<InMemoryFileSystem::InMemoryNode>
FlattenedInMemoryNode::unflatten(const std::vector<FlattenedInMemoryNode>& flattened_nodes,
                                 const std::shared_ptr<MemoryBuffer>& buffer) noexcept {
    if (flattened_nodes.size() == 0) {
        return nullptr;
    }

    std::vector<std::unique_ptr<InMemoryFileSystem::InMemoryNode>> nodes;
    nodes.reserve(flattened_nodes.size());
    for (size_t index = 0; index < flattened_nodes.size(); index++) {
        const FlattenedInMemoryNode& flattened_node = flattened_nodes[index];
        auto name = flattened_node.name;
        auto attributes = flattened_node.attributes;
        switch (flattened_node.kind) {
            case InMemoryFileSystem::InMemoryNode::Kind::File: {
                auto region = flattened_node.data_region;
                std::shared_ptr sliced_buffer = buffer->slice(region);
                if (!sliced_buffer) {
                    return nullptr;
                }
                auto file_node = std::make_unique<InMemoryFileNode>(
                    std::move(name), std::move(attributes), std::move(sliced_buffer));
                nodes.emplace_back(std::move(file_node));
                break;
            }
            case InMemoryFileSystem::InMemoryNode::Kind::Directory: {
                std::unordered_map<std::string, std::unique_ptr<InMemoryFileSystem::InMemoryNode>> items;
                items.reserve(flattened_node.children.size());
                for (const auto& [name, index]: flattened_node.children) {
                    auto moveIt = std::make_move_iterator(nodes.begin() + index);
                    items[name] = *moveIt;
                }
                auto directory_node =
                    std::make_unique<InMemoryDirectoryNode>(std::move(name), std::move(attributes), std::move(items));
                nodes.emplace_back(std::move(directory_node));
                break;
            }
        }
    }

    return std::move(nodes.back());
}

void to_json(json& j, const FlattenedInMemoryNode& flattened_node) {
    j = json { { FlattenedInMemoryNodeKeys::kName, flattened_node.name },
               { FlattenedInMemoryNodeKeys::kDataRegion, flattened_node.data_region },
               { FlattenedInMemoryNodeKeys::kChildren, flattened_node.children },
               { FlattenedInMemoryNodeKeys::kKind, static_cast<int>(flattened_node.kind) } };
}

void from_json(const json& j, FlattenedInMemoryNode& flattened_node) {
    if (j.contains(FlattenedInMemoryNodeKeys::kName)) {
        j.at(FlattenedInMemoryNodeKeys::kName).get_to(flattened_node.name);
    }
    if (j.contains(FlattenedInMemoryNodeKeys::kDataRegion)) {
        j.at(FlattenedInMemoryNodeKeys::kDataRegion).get_to(flattened_node.data_region);
    }
    if (j.contains(FlattenedInMemoryNodeKeys::kChildren)) {
        j.at(FlattenedInMemoryNodeKeys::kChildren).get_to(flattened_node.children);
    }
    if (j.contains(FlattenedInMemoryNodeKeys::kKind)) {
        j.at(FlattenedInMemoryNodeKeys::kKind).get_to(flattened_node.kind);
    }
}

struct InMemoryNodeMetaDataKeys {
    constexpr static std::string_view kNodes = "nodes";
};

struct InMemoryNodeMetaData {
    std::vector<FlattenedInMemoryNode> nodes;

    std::string json_string();
    void writeToStream(std::ostream& ostream);
};

void to_json(json& j, const InMemoryNodeMetaData& metadata) {
    j = json { { InMemoryNodeMetaDataKeys::kNodes, metadata.nodes } };
}

void from_json(const json& j, InMemoryNodeMetaData& metadata) {
    if (j.contains(InMemoryNodeMetaDataKeys::kNodes)) {
        j.at(InMemoryNodeMetaDataKeys::kNodes).get_to(metadata.nodes);
    }
}

std::string InMemoryNodeMetaData::json_string() {
    json value;
    to_json(value, *this);
    std::stringstream ss;
    ss << value;
    std::string metadataStr = ss.str();
    return metadataStr;
}

void InMemoryNodeMetaData::writeToStream(std::ostream& ostream) {
    std::string jsonStr = json_string();
    // reverse it for writing
    std::reverse(jsonStr.begin(), jsonStr.end());
    ostream << jsonStr;
}
} // namespace serdes

namespace inmemoryfs {

std::string InMemoryFileSystem::ErrorCategory::message(int code) const {
    switch (static_cast<ErrorCode>(code)) {
        case ErrorCode::DirectoryExists:
            return "The item at the path is a directory.";
        case ErrorCode::ItemNotFound:
            return "Path does not exist.";
        case ErrorCode::ItemExists:
            return "Item already exists at the path.";
        case ErrorCode::DirectoryExpected:
            return "The item at the path is not a directory";
        case ErrorCode::FileExpected:
            return "The item at the path is not a file";
    }
}

InMemoryFileSystem::InMemoryFileSystem(std::string name) noexcept
    : root_(std::make_unique<InMemoryDirectoryNode>(std::move(name), Attributes())) { }

bool InMemoryFileSystem::is_directory(const std::vector<std::string>& canonical_path) noexcept {
    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    return node && node->isDirectory();
}

bool InMemoryFileSystem::is_file(const std::vector<std::string>& canonical_path) noexcept {
    auto node = get_node(root_.get(), canonical_path.begin(), canonical_path.end());
    return node && node->isFile();
}

bool InMemoryFileSystem::exists(const std::vector<std::string>& canonical_path) noexcept {
    auto node = get_node(root_.get(), canonical_path.begin(), canonical_path.end());
    return node != nullptr;
}

std::vector<std::vector<std::string>> InMemoryFileSystem::get_item_paths(const std::vector<std::string>& canonical_path,
                                                                         std::error_code& error) const noexcept {
    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (node == nullptr) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return {};
    }

    if (node->isFile()) {
        error = InMemoryFileSystem::ErrorCode::DirectoryExpected;
        return {};
    }

    auto directory_node = static_cast<InMemoryDirectoryNode*>(node);
    const auto& items = directory_node->get_items();
    std::vector<std::vector<std::string>> result;
    result.reserve(items.size());
    for (const auto& [component, _]: items) {
        auto components = canonical_path;
        components.emplace_back(component);
        result.emplace_back(std::move(components));
    }

    return result;
}

std::optional<InMemoryFileSystem::Attributes>
InMemoryFileSystem::get_attributes(const std::vector<std::string>& canonical_path,
                                   std::error_code& error) const noexcept {
    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (node == nullptr) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return std::nullopt;
    }

    return node->attributes();
}

std::shared_ptr<MemoryBuffer> InMemoryFileSystem::get_file_content(const std::vector<std::string>& canonical_path,
                                                                   std::error_code& error) const noexcept {
    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (node == nullptr) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return nullptr;
    }
    if (node->isDirectory()) {
        error = InMemoryFileSystem::ErrorCode::FileExpected;
        return nullptr;
    }

    InMemoryFileNode* file_node = static_cast<InMemoryFileNode*>(node);
    return file_node->getBuffer();
}

bool InMemoryFileSystem::make_directory(const std::vector<std::string>& canonical_path,
                                        Attributes attributes,
                                        bool create_intermediate_directories,
                                        std::error_code& error) noexcept {
    if (canonical_path.size() == 0) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    auto directory_node = static_cast<InMemoryDirectoryNode*>(root());
    for (auto it = canonical_path.begin(); it != canonical_path.end(); ++it) {
        auto node = directory_node->get_item(*it);
        bool createDirectory = create_intermediate_directories || is_last(it, canonical_path);
        if (node == nullptr && createDirectory) {
            auto child_directory_node = std::make_unique<InMemoryDirectoryNode>(*it, attributes);
            directory_node->add_item(*it, std::move(child_directory_node));
            directory_node = static_cast<InMemoryDirectoryNode*>(directory_node->get_item(*it));
        } else if (node != nullptr && node->isDirectory()) {
            directory_node = static_cast<InMemoryDirectoryNode*>(node);
        } else {
            if (node == nullptr) {
                error = InMemoryFileSystem::ErrorCode::ItemNotFound;
            } else if (node->isFile()) {
                error = InMemoryFileSystem::ErrorCode::DirectoryExpected;
            }
            return false;
        }
    }

    return true;
}

bool InMemoryFileSystem::make_file(const std::vector<std::string>& canonical_path,
                                   std::shared_ptr<MemoryBuffer> buffer,
                                   Attributes attributes,
                                   bool overwrite,
                                   std::error_code& error) noexcept {
    if (canonical_path.size() == 0) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    auto node = get_node(root(), canonical_path.begin(), std::prev(canonical_path.end()));
    if (!node) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    if (!node->isDirectory()) {
        error = InMemoryFileSystem::ErrorCode::DirectoryExpected;
        return false;
    }

    InMemoryDirectoryNode* directory_node = static_cast<InMemoryDirectoryNode*>(node);
    if (directory_node->contains(canonical_path.back()) && !overwrite) {
        error = InMemoryFileSystem::ErrorCode::ItemExists;
        return false;
    }

    auto name = canonical_path.back();
    auto file_node = std::make_unique<InMemoryFileNode>(std::move(name), std::move(attributes), std::move(buffer));
    directory_node->add_item(canonical_path.back(), std::move(file_node));

    return true;
}

bool InMemoryFileSystem::remove_item(const std::vector<std::string>& canonical_path, std::error_code& error) noexcept {
    if (canonical_path.size() == 0) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    auto node = get_node(root(), canonical_path.begin(), std::prev(canonical_path.end()));
    if (!node->isDirectory()) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    InMemoryDirectoryNode* directory_node = static_cast<InMemoryDirectoryNode*>(node);
    if (!directory_node->contains(canonical_path.back())) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }
    directory_node->remove_item(canonical_path.back());

    return true;
}

bool InMemoryFileSystem::set_attributes(const std::vector<std::string>& canonical_path,
                                        Attributes attributes,
                                        std::error_code& error) noexcept {
    if (canonical_path.size() == 0) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (!node) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    node->setAttributes(std::move(attributes));
    return true;
}

bool InMemoryFileSystem::write_item_to_disk(const std::vector<std::string>& canonical_path,
                                            const std::string& dst_path,
                                            bool recursive,
                                            std::error_code& error) const noexcept {
    std::filesystem::path dst_file_path(dst_path);
    auto status = std::filesystem::exists(dst_file_path, error);
    if (!status || error) {
        return false;
    }

    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (!node) {
        error = InMemoryFileSystem::ErrorCode::ItemNotFound;
        return false;
    }

    bool result = write_node(node, dst_file_path, recursive, error);
    if (!result) {
        auto rootPath = dst_path;
        rootPath.append(root()->name());
        std::filesystem::remove_all(rootPath, error);
    }

    return result;
}

std::unique_ptr<InMemoryFileSystem> InMemoryFileSystem::make(const std::string& path, std::error_code& error) {
    std::filesystem::path file_path(path);
    auto status = std::filesystem::exists(file_path, error);
    if (!status || error) {
        return nullptr;
    }

    auto node = make_node(file_path, error);
    if (!node) {
        return nullptr;
    }

    switch (node->kind()) {
        case InMemoryFileSystem::InMemoryNode::Kind::Directory: {
            auto fs = std::make_unique<InMemoryFileSystem>(std::move(node));
            return fs;
        }
        case InMemoryFileSystem::InMemoryNode::Kind::File: {
            auto fs = std::make_unique<InMemoryFileSystem>();
            auto rootNode = static_cast<InMemoryDirectoryNode*>(fs->root());
            rootNode->add_item(file_path.filename().string(), std::move(node));
            return fs;
        }
    }
}

void InMemoryFileSystem::serialize(const std::vector<std::string>& canonical_path,
                                   size_t alignment,
                                   std::ostream& stream) {
    using namespace serdes;
    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (!node) {
        return;
    }

    auto nodes = FlattenedInMemoryNode::flatten(node, alignment);
    for (const auto& node: nodes) {
        if (node.file_node == nullptr) {
            continue;
        }
        auto region = node.data_region;
        auto buffer = node.file_node->getBuffer();
        auto start = static_cast<char*>(buffer->data());
        stream.seekp(region.offset);
        stream.write(start, region.size);
    }

    auto metadata = InMemoryNodeMetaData { .nodes = std::move(nodes) };
    metadata.writeToStream(stream);
}

size_t InMemoryFileSystem::get_serialization_size(const std::vector<std::string>& canonical_path, size_t alignment) {
    using namespace serdes;

    auto node = get_node(root(), canonical_path.begin(), canonical_path.end());
    if (!node) {
        return 0;
    }

    auto nodes = FlattenedInMemoryNode::flatten(node, alignment);
    size_t length = 0;
    for (const auto& node: nodes) {
        auto region = node.data_region;
        length = std::max(region.get_length(), length);
    }
    auto metadata = InMemoryNodeMetaData { .nodes = std::move(nodes) };
    const auto& str = metadata.json_string();
    length += str.length();

    return length;
}

std::unique_ptr<InMemoryFileSystem> InMemoryFileSystem::make(const std::shared_ptr<MemoryBuffer>& buffer) {
    using namespace serdes;

    // read metadata from the end of the stream
    auto istream = ReversedIMemoryStream(buffer);
    json metadata_json;
    nlohmann::detail::json_sax_dom_parser<json> sdp(metadata_json, true);
    if (!json::sax_parse(istream, &sdp, nlohmann::detail::input_format_t::json, false)) {
        return nullptr;
    }

    InMemoryNodeMetaData metadata;
    from_json(metadata_json, metadata);
    auto rootNode = FlattenedInMemoryNode::unflatten(metadata.nodes, buffer);
    if (!rootNode) {
        return nullptr;
    }

    return std::make_unique<InMemoryFileSystem>(std::move(rootNode));
}

} // namespace inmemoryfs
