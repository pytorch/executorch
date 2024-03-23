//
// multiarray.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <vector>

namespace executorchcoreml {

/// A class representing an unowned buffer.
class Buffer {
public:
    /// Constructs a buffer from data and size.
    explicit Buffer(const void* data, size_t size) noexcept : data_(data), size_(size) { }

    /// Returns the data pointer.
    inline const void* data() const noexcept { return data_; }

    /// Returns the size of the buffer.
    inline size_t size() const noexcept { return size_; }

private:
    const void* data_;
    size_t size_;
};

/// A class representing a MultiArray.
class MultiArray {
public:
    /// The MultiArray datatype.
    enum class DataType : uint8_t { Int = 0, Double, Float, Float16 };

    /// A class describing the memory layout of a MultiArray.
    class MemoryLayout {
    public:
        MemoryLayout(DataType dataType, std::vector<size_t> shape, std::vector<ssize_t> strides)
            : dataType_(dataType), shape_(std::move(shape)), strides_(std::move(strides)) { }

        /// Returns the datatype of the MultiArray.
        inline DataType dataType() const noexcept { return dataType_; }

        /// Returns the shape of the MultiArray.
        inline const std::vector<size_t>& shape() const noexcept { return shape_; }

        /// Returns the strides of the MultiArray.
        inline const std::vector<ssize_t>& strides() const noexcept { return strides_; }

        /// Returns the MultiArray rank.
        inline size_t rank() const noexcept { return shape_.size(); }

        /// Returns the number of elements in the MultiArray.
        size_t get_num_elements() const noexcept;

        /// Returns `true` if the memory layout is packed otherwise `false`.
        bool is_packed() const noexcept;

    private:
        DataType dataType_;
        std::vector<size_t> shape_;
        std::vector<ssize_t> strides_;
    };

    /// Constructs a `MultiArray` from data and it's memory layout.
    ///
    /// The data is not owned by the `MultiArray`.
    MultiArray(void* data, MemoryLayout layout) : data_(data), layout_(std::move(layout)) { }

    /// Returns the data pointer.
    inline void* data() const noexcept { return data_; }

    /// Returns the layout of the MultiArray.
    inline const MemoryLayout& layout() const noexcept { return layout_; }

    /// Copies this into another `MultiArray`.
    ///
    /// @param dst The destination `MultiArray`.
    bool copy(MultiArray& dst) const noexcept;

private:
    void* data_;
    MemoryLayout layout_;
};

} // namespace executorchcoreml
