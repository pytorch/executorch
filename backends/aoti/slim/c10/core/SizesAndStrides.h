/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/platform/assert.h>

/// Maximum number of dimensions that can be stored inline.
/// For tensors with more dimensions, out-of-line storage is used.
#define SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5

namespace executorch::backends::aoti::slim::c10 {

using ::executorch::runtime::IntArrayRef;

/**
 * SizesAndStrides - Packed container for tensor sizes and strides.
 *
 * This class efficiently stores tensor dimension sizes and strides together.
 * For tensors with up to 5 dimensions, storage is inline (no heap allocation).
 * For larger tensors, heap storage is used.
 *
 * Memory layout:
 * - Inline: 5 int64_t for sizes + 5 int64_t for strides
 * - Out-of-line: pointer to heap array [sizes..., strides...]
 */
class SizesAndStrides {
 public:
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  /// Default constructor - creates a 1-dimensional tensor with size 0.
  SizesAndStrides() {
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (SLIMTENSOR_UNLIKELY(!isInline())) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(outOfLineStorage_);
    }
  }

  SizesAndStrides(const SizesAndStrides& rhs) : size_(rhs.size_) {
    if (SLIMTENSOR_LIKELY(rhs.isInline())) {
      copyDataInline(rhs);
    } else {
      allocateOutOfLineStorage(size_);
      copyDataOutline(rhs);
    }
  }

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (SLIMTENSOR_LIKELY(rhs.isInline())) {
      if (SLIMTENSOR_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      copyDataInline(rhs);
    } else {
      if (isInline()) {
        allocateOutOfLineStorage(rhs.size_);
      } else {
        resizeOutOfLineStorage(rhs.size_);
      }
      copyDataOutline(rhs);
    }
    size_ = rhs.size_;
    return *this;
  }

  SizesAndStrides(SizesAndStrides&& rhs) noexcept : size_(rhs.size_) {
    if (SLIMTENSOR_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }
    rhs.size_ = 0;
  }

  SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    if (SLIMTENSOR_LIKELY(rhs.isInline())) {
      if (SLIMTENSOR_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      copyDataInline(rhs);
    } else {
      if (!isInline()) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }
    size_ = rhs.size_;
    rhs.size_ = 0;
    return *this;
  }

  bool operator==(const SizesAndStrides& other) const {
    if (size_ != other.size_) {
      return false;
    }
    return !(
        isInline()
            ? std::memcmp(
                  inlineStorage_, other.inlineStorage_, sizeof(inlineStorage_))
            : std::memcmp(
                  outOfLineStorage_,
                  other.outOfLineStorage_,
                  storageBytes(size_)));
  }

  /// Returns the number of dimensions.
  size_t size() const noexcept {
    return size_;
  }

  // Size accessors

  const int64_t* sizes_data() const noexcept {
    if (SLIMTENSOR_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  int64_t* sizes_data() noexcept {
    if (SLIMTENSOR_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  sizes_const_iterator sizes_begin() const noexcept {
    return sizes_data();
  }

  sizes_iterator sizes_begin() noexcept {
    return sizes_data();
  }

  sizes_const_iterator sizes_end() const noexcept {
    return sizes_begin() + size();
  }

  sizes_iterator sizes_end() noexcept {
    return sizes_begin() + size();
  }

  IntArrayRef sizes_arrayref() const noexcept {
    return IntArrayRef{sizes_data(), size()};
  }

  void set_sizes(IntArrayRef newSizes) {
    resize(newSizes.size());
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  void set_sizes(std::initializer_list<int64_t> newSizes) {
    resize(newSizes.size());
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  int64_t size_at(size_t idx) const noexcept {
    ET_DCHECK_MSG(idx < size(), "Index out of bounds");
    return sizes_data()[idx];
  }

  int64_t& size_at(size_t idx) noexcept {
    ET_DCHECK_MSG(idx < size(), "Index out of bounds");
    return sizes_data()[idx];
  }

  int64_t size_at_unchecked(size_t idx) const noexcept {
    return sizes_data()[idx];
  }

  int64_t& size_at_unchecked(size_t idx) noexcept {
    return sizes_data()[idx];
  }

  // Stride accessors

  const int64_t* strides_data() const noexcept {
    if (SLIMTENSOR_LIKELY(isInline())) {
      return &inlineStorage_[SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  int64_t* strides_data() noexcept {
    if (SLIMTENSOR_LIKELY(isInline())) {
      return &inlineStorage_[SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_begin() const noexcept {
    return strides_data();
  }

  strides_iterator strides_begin() noexcept {
    return strides_data();
  }

  strides_const_iterator strides_end() const noexcept {
    return strides_begin() + size();
  }

  strides_iterator strides_end() noexcept {
    return strides_begin() + size();
  }

  IntArrayRef strides_arrayref() const noexcept {
    return IntArrayRef{strides_data(), size()};
  }

  void set_strides(IntArrayRef strides) {
    ET_DCHECK_MSG(
        strides.size() == size(),
        "strides size (%zu) must match size (%zu)",
        strides.size(),
        size());
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  void set_strides(std::initializer_list<int64_t> strides) {
    ET_DCHECK_MSG(
        strides.size() == size(),
        "strides size (%zu) must match size (%zu)",
        strides.size(),
        size());
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  int64_t stride_at(size_t idx) const noexcept {
    ET_DCHECK_MSG(idx < size(), "Index out of bounds");
    return strides_data()[idx];
  }

  int64_t& stride_at(size_t idx) noexcept {
    ET_DCHECK_MSG(idx < size(), "Index out of bounds");
    return strides_data()[idx];
  }

  int64_t stride_at_unchecked(size_t idx) const noexcept {
    return strides_data()[idx];
  }

  int64_t& stride_at_unchecked(size_t idx) noexcept {
    return strides_data()[idx];
  }

  /// Resizes to a new number of dimensions.
  void resize(size_t newSize) {
    const auto oldSize = size();
    if (newSize == oldSize) {
      return;
    }
    if (SLIMTENSOR_LIKELY(
            newSize <= SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE &&
            isInline())) {
      if (oldSize < newSize) {
        const auto bytesToZero =
            (newSize - oldSize) * sizeof(inlineStorage_[0]);
        memset(&inlineStorage_[oldSize], 0, bytesToZero);
        memset(
            &inlineStorage_
                [SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
            0,
            bytesToZero);
      }
      size_ = newSize;
    } else {
      resizeSlowPath(newSize, oldSize);
    }
  }

 private:
  bool isInline() const noexcept {
    return size_ <= SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    ET_DCHECK_MSG(rhs.isInline(), "rhs must be inline");
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
  }

  void copyDataOutline(const SizesAndStrides& rhs) noexcept {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
  }

  static size_t storageBytes(size_t size) noexcept {
    return size * 2 * sizeof(int64_t);
  }

  void allocateOutOfLineStorage(size_t size) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    outOfLineStorage_ = static_cast<int64_t*>(malloc(storageBytes(size)));
    ET_CHECK_MSG(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  void resizeOutOfLineStorage(size_t newSize) {
    ET_DCHECK_MSG(!isInline(), "must not be inline");
    outOfLineStorage_ = static_cast<int64_t*>(
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        realloc(outOfLineStorage_, storageBytes(newSize)));
    ET_CHECK_MSG(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  void resizeSlowPath(size_t newSize, size_t oldSize) {
    if (newSize <= SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {
      ET_DCHECK_MSG(
          !isInline(),
          "resizeSlowPath called when fast path should have been hit!");
      int64_t* tempStorage = outOfLineStorage_;
      memcpy(
          &inlineStorage_[0],
          &tempStorage[0],
          SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE *
              sizeof(inlineStorage_[0]));
      memcpy(
          &inlineStorage_[SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
          &tempStorage[oldSize],
          SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE *
              sizeof(inlineStorage_[0]));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(tempStorage);
    } else {
      if (isInline()) {
        int64_t* tempStorage =
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            static_cast<int64_t*>(malloc(storageBytes(newSize)));
        ET_CHECK_MSG(
            tempStorage,
            "Could not allocate memory to change Tensor SizesAndStrides!");
        const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
        const auto bytesToZero = (newSize > oldSize)
            ? (newSize - oldSize) * sizeof(tempStorage[0])
            : 0;
        memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);
        if (bytesToZero) {
          memset(&tempStorage[oldSize], 0, bytesToZero);
        }
        memcpy(
            &tempStorage[newSize],
            &inlineStorage_[SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
            bytesToCopy);
        if (bytesToZero) {
          memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
        }
        outOfLineStorage_ = tempStorage;
      } else {
        const bool isGrowing = oldSize < newSize;
        if (isGrowing) {
          resizeOutOfLineStorage(newSize);
        }
        memmove(
            outOfLineStorage_ + newSize,
            outOfLineStorage_ + oldSize,
            std::min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));
        if (!isGrowing) {
          resizeOutOfLineStorage(newSize);
        } else {
          const auto bytesToZero =
              (newSize - oldSize) * sizeof(outOfLineStorage_[0]);
          memset(&outOfLineStorage_[oldSize], 0, bytesToZero);
          memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);
        }
      }
    }
    size_ = newSize;
  }

  size_t size_{1};
  union {
    int64_t* outOfLineStorage_;
    // NOLINTNEXTLINE(*c-array*)
    int64_t inlineStorage_[SLIMTENSOR_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
  };
};

} // namespace executorch::backends::aoti::slim::c10
