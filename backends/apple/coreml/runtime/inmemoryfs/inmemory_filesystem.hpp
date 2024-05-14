//
// inmemory_filesystem.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <stdio.h>
#include <string>
#include <system_error>

#include "inmemory_filesystem_metadata.hpp"
#include "memory_buffer.hpp"

namespace inmemoryfs {

/// A class representing an in-memory file system.
class InMemoryFileSystem final {
public:
    /// Error codes for `InMemoryFileSystem`.
    enum class ErrorCode: int8_t {
        DirectoryExists = 1,   // If the path already exists.
        ItemNotFound,          // If the path does not exist.
        ItemExists,            // If an item at the path already exists.
        DirectoryExpected,     // If path is not a directory.
        FileExpected,          // If the path is not a file.
    };

    /// Options for loading file content.
    enum class FileLoadOption: int8_t {
        Malloc = 1,   // Copy file contents into memory.
        MMap,         // Memory map file contents.
        LazyMMap      // Memory map file contents but lazily.
    };

    /// The error category for `InMemoryFileSystem`.
    struct ErrorCategory final: public std::error_category {
    public:
        inline const char* name() const noexcept override {
            return "InMemoryFileSystem";
        }

        std::string message(int code) const override;
    };

    struct Attributes {
        time_t modificationTime;

        inline Attributes() noexcept:
        modificationTime(time(0))
        {}
    };

    using MetadataWriter = std::function<bool(const InMemoryFileSystemMetadata&, std::ostream&)>;
    using MetadataWriterInMemory = std::function<size_t(const InMemoryFileSystemMetadata&, void *)>;
    using MetadataReader = std::function<std::optional<InMemoryFileSystemMetadata>(std::istream&)>;

    /// A class representing an in-memory node. This could either be a file node or a directory node.
    class InMemoryNode {
    public:
        /// The node kind.
        enum class Kind: uint8_t {
            File = 0,   /// Node is a File.
            Directory   /// Node is a Directory.
        };

        /// Constructs an in-memory node instance.
        ///
        /// @param name The name of the Node. It must be unique in the enclosing Directory.
        /// @param attributes   The node attributes.
        /// @param kind   The node kind.
        inline InMemoryNode(std::string name, Attributes attributes, Kind kind) noexcept:
        name_(std::move(name)),
        attributes_(std::move(attributes)),
        kind_(kind)
        {}

        InMemoryNode(InMemoryNode const&) = delete;
        InMemoryNode& operator=(InMemoryNode const&) = delete;

        inline virtual ~InMemoryNode() {}

        /// Returns the node attributes.
        inline Attributes attributes() const noexcept {
            return attributes_;
        }

        /// Sets the node attributes.
        ///
        /// @param attributes The node attributes.
        inline void set_attributes(Attributes attributes) noexcept {
            attributes_ = std::move(attributes);
        }

        /// Returns the node kind, possible values are `File` and `Directory`.
        inline Kind kind() const noexcept {
            return kind_;
        }

        /// Returns the name of the node.
        inline const std::string& name() const noexcept {
            return name_;
        }

        inline void set_name(std::string name) noexcept {
            std::swap(name_, name);
        }

        /// Returns `true` if the node is a directory otherwise `false`.
        inline bool isDirectory() const noexcept {
            switch (kind_) {
                case InMemoryFileSystem::InMemoryNode::Kind::Directory:
                    return true;
                default:
                    return false;
            }
        }

        /// Returns `true` if the node is a file otherwise `false`.
        inline bool isFile() const noexcept {
            return !isDirectory();
        }

    private:
        std::string name_;
        InMemoryFileSystem::Attributes attributes_;
        const Kind kind_;
    };

    /// Constructs an`InMemoryFileSystem` instance with an empty root and the specified name.
    ///
    /// @param rootName The name of the root node.
    explicit InMemoryFileSystem(std::string rootName = "root") noexcept;

    /// Constructs an`InMemoryFileSystem` instance with the specified root.
    ///
    /// @param root The root node.
    explicit InMemoryFileSystem(std::unique_ptr<InMemoryNode> root) noexcept
    :root_(std::move(root))
    {}

    InMemoryFileSystem(InMemoryFileSystem const&) = delete;
    InMemoryFileSystem& operator=(InMemoryFileSystem const&) = delete;

    virtual ~InMemoryFileSystem() {}

    /// Returns the root.
    InMemoryNode *root() const noexcept {
        return root_.get();
    }

    /// Checks if the node at the specified path is a directory.
    ///
    /// @param canonical_path   The path components from the root.
    /// @retval `true` if the node at the specified path is a directory otherwise `false`.
    bool is_directory(const std::vector<std::string>& canonical_path) noexcept;

    /// Checks if the node at the specified path is a file.
    ///
    /// @param canonical_path   The path components from the root.
    /// @retval `true` if the node at the specified path is a file otherwise `false`.
    bool is_file(const std::vector<std::string>& canonical_path) noexcept;

    /// Checks if the node at the specified path exists.
    ///
    /// @param canonical_path   The path components from the root.
    /// @retval `true` if the node at the specified path exists.
    bool exists(const std::vector<std::string>& canonical_path) const noexcept;

    /// Retrieves the canonical path of all the child nodes at the specified path. The node
    /// at the specified path must be a directory otherwise it returns an empty vector with the `error`
    /// populated.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval paths to all the items at the specified path.
    std::vector<std::vector<std::string>> get_item_paths(const std::vector<std::string>& canonical_path,
                                                         std::error_code& error) const noexcept;

    /// Retrieves the attributes of the item at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The item attributes at the specified path.
    std::optional<Attributes> get_attributes(const std::vector<std::string>& canonical_path,
                                             std::error_code& error) const noexcept;

    /// Retrieves the contents of the file at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The file contents or `nullptr` if the item at the specified path is not a file.
    std::shared_ptr<MemoryBuffer> get_file_content(const std::vector<std::string>& canonical_path,
                                                   std::error_code& error) const noexcept;

    /// Creates an in-memory directory at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param attributes  The directory attributes.
    /// @param create_intermediate_directories   If this is `true` then the method will also create intermediate directories if not present.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the directory is created otherwise `false`.
    bool make_directory(const std::vector<std::string>& canonical_path,
                        Attributes attributes,
                        bool create_intermediate_directories,
                        std::error_code& error) noexcept;

    /// Creates an in-memory file at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param buffer  The file contents.
    /// @param attributes  The file attributes.
    /// @param overwrite   If this is `true` then the the method will overwrite the contents at the specified path.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the file is created otherwise `false`.
    bool make_file(const std::vector<std::string>& canonical_path,
                   std::shared_ptr<MemoryBuffer> buffer,
                   Attributes attributes,
                   bool overwrite,
                   std::error_code& error) noexcept;

    /// Removes the item at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the item is removed otherwise `false`.
    bool remove_item(const std::vector<std::string>& canonical_path,
                     std::error_code& error) noexcept;

    /// Sets the attributes at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the attributes are updated otherwise `false`.
    bool set_attributes(const std::vector<std::string>& canonical_path,
                        Attributes attributes,
                        std::error_code& error) noexcept;

    /// Writes the item at the specified path to the filesystem.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param dst_path  The filesystem path where the item contents will be saved.
    /// @param recursive   If this is `true` then the the method will recursively write the contents of nested directory items.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the write succeeded otherwise `false`.
    bool write_item_to_disk(const std::vector<std::string>& canonical_path,
                            const std::string& dst_path,
                            bool recursive,
                            std::error_code& error) const noexcept;

    /// Renames the item at the specified path, if there is already an item with the same name then
    /// the rename would fail.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param name The new name,
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the write succeeded otherwise `false`.
    bool rename_item(const std::vector<std::string>& canonical_path,
                     const std::string& name,
                     std::error_code& error) noexcept;

    /// Creates  an`InMemoryFileSystem` from the filesystem path.
    ///
    /// The structure of the `InMemoryFileSystem` is identical to the structure of the filesystem at the
    /// specified path.
    ///
    /// @param path  The filesystem path.
    /// @param option The loading option.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The `InMemoryFileSystem` instance if the construction succeeded otherwise `nullptr`.
    static std::unique_ptr<InMemoryFileSystem> make_from_directory(const std::string& path,
                                                                   FileLoadOption option,
                                                                   std::error_code& error) noexcept;

    /// Serializes the item at the specified path and writes it to the stream.
    ///
    /// The structure of the `InMemoryFileSystem` is identical to the structure of the filesystem at the
    /// specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param alignment  The alignment of the offset where an item is written to the stream.
    /// @param metadata_writer The function to use when serializing the filesystem metadata.
    /// @param ostream   The output stream.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the serialized bytes were written to `ostream` otherwise `false`.
    bool serialize(const std::vector<std::string>& canonical_path,
                   size_t alignment,
                   const MetadataWriter& metadata_writer,
                   std::ostream& ostream,
                   std::error_code& error) const noexcept;

    /// Serializes the item at the specified path and writes it to the stream.
    ///
    /// The structure of the `InMemoryFileSystem` is identical to the structure of the filesystem at the
    /// specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param alignment  The alignment of the offset where an item is written to the stream.
    /// @param metadata_writer The function to use when serializing the filesystem metadata.
    /// @param dst   The destination pointer.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the serialized bytes were written to `ostream` otherwise `false`.
    bool serialize(const std::vector<std::string>& canonical_path,
                   size_t alignment,
                   const MetadataWriterInMemory& metadata_writer,
                   void *dst,
                   std::error_code& error) const noexcept;

    /// Computes the size of the buffer that would be needed to serialized the item at the specified path.
    ///
    /// @param canonical_path  The path components from the root.
    /// @param alignment  The offset alignment where an item is written to the stream.
    /// @param metadata_writer The function to use when serializing the filesystem metadata.
    /// @retval The size of the buffer that will be needed to write the item at the specified path.
    size_t get_buffer_size_for_serialization(const std::vector<std::string>& canonical_path,
                                             size_t alignment,
                                             const MetadataWriter& metadata_writer) const noexcept;

    /// Constructs an `InMemoryFileSystem` instance from the buffer contents.
    ///
    /// @param buffer  The memory buffer.
    /// @param metadata_reader The function to use when deserializing the filesystem metadata.
    /// @retval The constructed `InMemoryFileSystem` or `nullptr` if the deserialization failed.
    static std::unique_ptr<InMemoryFileSystem> make_from_buffer(const std::shared_ptr<MemoryBuffer>& buffer,
                                                                const MetadataReader& metadata_reader) noexcept;

private:
    const std::unique_ptr<InMemoryNode> root_;
};

/// Constructs an `error_code` from a `InMemoryFileSystem::ErrorCode`.
inline std::error_code make_error_code(InMemoryFileSystem::ErrorCode code) {
    static InMemoryFileSystem::ErrorCategory errorCategory;
    return {static_cast<int>(code), errorCategory};
}

}; // namespace inmemoryfs

namespace std {

template <> struct is_error_code_enum<inmemoryfs::InMemoryFileSystem::ErrorCode> : true_type {};

} // namespace std
