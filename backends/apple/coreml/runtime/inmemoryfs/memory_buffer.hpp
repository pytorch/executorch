//
// memory_buffer.hpp
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <stdio.h>
#include <string>
#include <memory>
#include <vector>

#include <system_error>

namespace inmemoryfs {

/// A struct representing a memory region.
struct MemoryRegion {
    inline MemoryRegion(size_t offset, size_t size) noexcept:
    offset(offset), size(size)
    {}
    
    inline MemoryRegion() noexcept:
    offset(0), size(0)
    {}
    
    /// Returns the length of the region.
    inline size_t get_length() const noexcept {
        return offset + size;
    }
    
    size_t offset = 0;
    size_t size = 0;
};

/// A class representing a memory buffer.
class MemoryBuffer: public std::enable_shared_from_this<MemoryBuffer> {
public:
    /// The kind of buffer.
    enum class Kind: uint8_t {
        MMap = 0,  // If the buffer is memory mapped.
        Malloc ,   // If the buffer is heap allocated.
    };
    
    enum class ReadOption: uint8_t {
        MMap = 0,
        Malloc,
        Any
    };
    
    inline MemoryBuffer(void *data,
                        size_t size,
                        Kind kind = Kind::Malloc,
                        std::shared_ptr<MemoryBuffer> parent = nullptr) noexcept:
    data_(data), size_(size), kind_(kind), parent_(parent)
    {}
    
    MemoryBuffer(const MemoryBuffer &) = delete;
    MemoryBuffer &operator=(const MemoryBuffer &) = delete;
    
    virtual ~MemoryBuffer() noexcept {}
    
    /// Returns the underlying data.
    inline void *data() const noexcept {
        return data_;
    }
    
    /// Returns the size of the buffer.
    inline const size_t size() const noexcept {
        return size_;
    }
    
    /// Returns the kind of the buffer.
    inline const Kind kind() const noexcept {
        return kind_;
    }
    
    /// Slices a buffer.
    ///
    /// @param region The memory region.
    /// @retval The sliced buffer if the region is inside the buffer otherwise `nullptr`.
    virtual std::shared_ptr<MemoryBuffer> slice(MemoryRegion region) noexcept;
    
    /// Reads the file content at the specified path.
    ///
    /// @param file_path The file path.
    /// @param regions The regions to be read.
    /// @param option The read option.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The read buffers or an empty vector if the read failed.
    static std::vector<std::shared_ptr<MemoryBuffer>>
    read_file_content(const std::string& file_path,
                      const std::vector<MemoryRegion>& regions,
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
    
    /// Constructs a `MemoryBuffer` by copying data.
    ///
    /// @param data The data pointer.
    /// @param size The size of the buffer.
    static std::shared_ptr<MemoryBuffer>
    make(const void *data, size_t size);
    
    /// Constructs a `MemoryBuffer` from a bytes vector.
    ///
    /// @param bytes A bytes vector.
    static std::shared_ptr<MemoryBuffer>
    make(std::vector<uint8_t> bytes);
    
    /// Constructs a `MemoryBuffer` without copying data.
    ///
    /// @param data The data pointer.
    /// @param size The size of the buffer.
    static std::shared_ptr<MemoryBuffer>
    make_unowned(void *data, size_t size);
    
private:
    void *data_;
    const size_t size_;
    Kind kind_;
    const std::shared_ptr<MemoryBuffer> parent_;
};
}
