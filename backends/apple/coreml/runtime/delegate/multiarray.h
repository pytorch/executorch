//
// multiarray.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <CoreML/CoreML.h>
#import <iostream>
#import <optional>
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
class MultiArray final {
public:
    /// The MultiArray datatype.
    enum class DataType : uint8_t {
        Bool = 0,
        Byte,
        Char,
        Short,
        Int32,
        Int64,
        Float16,
        Float32,
        Float64,
    };

    /// Options for copying.
    struct CopyOptions {
        inline CopyOptions() noexcept : use_bnns(true), use_memcpy(true) { }

        inline CopyOptions(bool use_bnns, bool use_memcpy) noexcept : use_bnns(use_bnns), use_memcpy(use_memcpy) { }

        bool use_bnns = true;
        bool use_memcpy = true;
    };

    /// A class describing the memory layout of a MultiArray.
    class MemoryLayout final {
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
        size_t num_elements() const noexcept;

        /// Returns the byte size of an element.
        size_t num_bytes() const noexcept;

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
    void copy(MultiArray& dst, CopyOptions options = CopyOptions()) const noexcept;

    /// Get the value at `indices`.
    template <typename T> inline T value(const std::vector<size_t>& indices) const noexcept {
        return *(static_cast<T*>(data(indices)));
    }

    /// Set the value at `indices`.
    template <typename T> inline void set_value(const std::vector<size_t>& indices, T value) const noexcept {
        T* ptr = static_cast<T*>(data(indices));
        *ptr = value;
    }

    /// Get the value at `index`.
    template <typename T> inline T value(size_t index) const noexcept { return *(static_cast<T*>(data(index))); }

    /// Set the value at `index`.
    template <typename T> inline void set_value(size_t index, T value) const noexcept {
        T* ptr = static_cast<T*>(data(index));
        *ptr = value;
    }

private:
    void* data(const std::vector<size_t>& indices) const noexcept;

    void* data(size_t index) const noexcept;

    void* data_;
    MemoryLayout layout_;
};

/// Converts `MultiArray::DataType` to `MLMultiArrayDataType`.
std::optional<MLMultiArrayDataType> to_ml_multiarray_data_type(MultiArray::DataType data_type);

/// Converts `MLMultiArrayDataType` to `MultiArray::DataType`.
std::optional<MultiArray::DataType> to_multiarray_data_type(MLMultiArrayDataType data_type);


} // namespace executorchcoreml
