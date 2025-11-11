//
// inmemory_filesystem_utils.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include "inmemory_filesystem.hpp"

namespace inmemoryfs {

/// Serializes the item at the specified path and writes it to the stream.
///
/// The structure of the `InMemoryFileSystem` is identical to the structure of the filesystem at the
/// specified path.
///
/// @param fs  The in-memory filesystem.
/// @param canonical_path  The path components from the root.
/// @param alignment  The alignment of the offset where an item is written to the stream.
/// @param ostream   The output stream.
/// @param error   On failure, error is populated with the failure reason.
/// @retval `true` if the serialized bytes were written to `ostream` otherwise `false`.
bool serialize(const InMemoryFileSystem& fs,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               std::ostream& ostream,
               std::error_code& error) noexcept;

/// Serializes the item at the specified path and writes it to the stream.
///
/// The structure of the `InMemoryFileSystem` is identical to the structure of the filesystem at the
/// specified path.
///
/// @param fs  The in-memory filesystem.
/// @param canonical_path  The path components from the root.
/// @param alignment  The alignment of the offset where an item is written to the stream.
/// @param dst   The destination pointer, the buffer size must be >= the size returned by `get_buffer_size_for_serialization`.
/// @param error   On failure, error is populated with the failure reason.
/// @retval `true` if the serialized bytes were written to `ostream` otherwise `false`.
bool serialize(const InMemoryFileSystem& fs,
               const std::vector<std::string>& canonical_path,
               size_t alignment,
               void *dst,
               std::error_code& error) noexcept;

/// Computes the size of the buffer that would be needed to serialized the item at the specified path.
///
/// @param fs  The in-memory filesystem.
/// @param canonical_path  The path components from the root.
/// @param alignment  The alignment of the offset where an item is written to the stream.
/// @retval The size of the buffer that will be needed to write the item at the specified path.
size_t get_buffer_size_for_serialization(const InMemoryFileSystem& fs,
                                         const std::vector<std::string>& canonical_path,
                                         size_t alignment) noexcept;

/// Constructs an `InMemoryFileSystem` instance from the buffer contents.
///
/// @param buffer  The memory buffer.
/// @retval The constructed `InMemoryFileSystem` or `nullptr` if the deserialization fail
std::unique_ptr<InMemoryFileSystem> make_from_buffer(const std::shared_ptr<MemoryBuffer>& buffer) noexcept;


} // namespace inmemoryfs
