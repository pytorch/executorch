//
// memory_buffer.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <memory>
#include <stdio.h>
#include <string>
#include <system_error>
#include <vector>

#include "range.hpp"

namespace inmemoryfs {
/// A class representing a memory buffer.
class MemoryBuffer: public std::enable_shared_from_this<MemoryBuffer> {
public:
    /// The kind of buffer.
    enum class Kind: uint8_t {
        MMap = 0,  // If the buffer is memory mapped.
        Malloc ,   // If the buffer is heap allocated.
    };

    enum class ReadOption: uint8_t {
        Malloc = 0,
        MMap,
        LazyMMap
    };

    inline MemoryBuffer(void *data,
                        size_t size,
                        Kind kind = Kind::Malloc,
                        std::shared_ptr<MemoryBuffer> parent = nullptr) noexcept:
    data_(data),
    size_(size),
    kind_(kind),
    parent_(parent)
    {}

    MemoryBuffer(const MemoryBuffer &) = delete;
    MemoryBuffer &operator=(const MemoryBuffer &) = delete;

    virtual ~MemoryBuffer() noexcept {}

    /// Returns the underlying data.
    virtual inline void *data() noexcept {
        return data_;
    }

    /// Returns the size of the buffer.
    inline const size_t size() const noexcept {
        return size_;
    }

    /// Loads the contents of the buffer.
    ///
    /// - For a malloced buffer, the method is a no op, content is loaded at the initialization time.
    /// - For a memory mapped buffer, the method can result in memory mapping the contents of the backed file.
    ///
    /// @param error  On failure, error is populated with the failure reason.
    /// @retval `true` if the copy succeeded otherwise `false`.
    inline virtual bool load(std::error_code& error) noexcept {
        return true;
    }

    /// Returns the kind of the buffer.
    inline const Kind kind() const noexcept {
        return kind_;
    }

    /// Returns the offset range that would be used when writing the buffer content.
    ///
    /// @param proposed_offset The proposed offset.
    /// @retval The  offset range that would be used when writing the buffer content.
    inline virtual std::pair<size_t, size_t> get_offset_range(size_t proposed_offset) const noexcept {
        return {proposed_offset, proposed_offset};
    }

    /// Returns the revised range that must be used for writing.
    ///
    /// @param dst  The destination pointer.
    /// @param proposed_range The proposed offset and size for writing the buffer content.
    /// @retval The revised offset and size that must be used to write the buffer content.
    inline virtual Range get_revised_range_for_writing(void *dst, Range proposed_range) const noexcept {
        return proposed_range;
    }

    /// Writes the contents of the buffer to the destination buffer at the given offset.
    ///
    /// @param dst The destination pointer.
    /// @param offset The offset.
    /// @param error  On failure, error is populated with the failure reason.
    /// @retval `true` if the write succeeded otherwise `false`.
    virtual bool write(void *dst,
                       size_t offset,
                       std::error_code& error) noexcept;

    /// Slices a buffer.
    ///
    /// @param range The memory range.
    /// @retval The sliced buffer if the region is inside the buffer otherwise `nullptr`.
    virtual std::shared_ptr<MemoryBuffer> slice(Range range) noexcept;

    /// Reads the file content at the specified path.
    ///
    /// @param file_path The file path.
    /// @param ranges The ranges to be read.
    /// @param option The read option.
    /// @param error  On failure, error is populated with the failure reason.
    /// @retval The read buffers or an empty vector if the read failed.
    static std::vector<std::shared_ptr<MemoryBuffer>>
    read_file_content(const std::string& file_path,
                      const std::vector<Range>& ranges,
                      ReadOption option,
                      std::error_code& error);

    /// Reads the whole file content at the specified path.
    ///
    /// @param file_path The file path.
    /// @param option The read option.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The read buffer or `nullptr` if the read failed.
    static std::shared_ptr<MemoryBuffer>
    read_file_content(const std::string& file_path,
                      ReadOption option,
                      std::error_code& error);

    /// Constructs a `MemoryBuffer`.
    ///
    /// @param size The size of the buffer.
    /// @param alignment The address alignment.
    static std::unique_ptr<MemoryBuffer>
    make_using_malloc(size_t size, size_t alignment = 1);


    /// Constructs a `MemoryBuffer` from memory allocated using `mmap`.
    ///
    /// @param size The size of the buffer.
    static std::unique_ptr<MemoryBuffer>
    make_using_mmap(size_t size);

    /// Constructs a `MemoryBuffer` without copying data.
    ///
    /// @param data The buffer content.
    /// @param size The size of the buffer.
    static std::unique_ptr<MemoryBuffer>
    make_unowned(void *data, size_t size);

    /// Constructs a `MemoryBuffer` with copying data.
    ///
    /// @param data The buffer content.
    /// @param size The size of the buffer.
    static std::unique_ptr<MemoryBuffer>
    make_copy(void *data, size_t size);
private:
    void *data_;
    const size_t size_;
    Kind kind_;
    const std::shared_ptr<MemoryBuffer> parent_;
};
}
