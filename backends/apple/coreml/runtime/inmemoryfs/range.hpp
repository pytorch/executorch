//
//  range.hpp
//
// Copyright © 2023 Apple Inc. All rights reserved.

#pragma once

namespace inmemoryfs {
/// A struct representing a memory region.
struct Range {
    inline Range(size_t offset, size_t size) noexcept:
    offset(offset), size(size)
    {}
    
    inline Range() noexcept:
    offset(0), size(0)
    {}
    
    /// Returns the length of the region.
    inline size_t length() const noexcept {
        return offset + size;
    }
    
    size_t offset = 0;
    size_t size = 0;
};
} // namespace inmemoryfs
