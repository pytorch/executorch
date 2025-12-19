#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/backends/aoti/slim/c10/util/ArrayRef.h>
#include <executorch/runtime/platform/assert.h>

#define STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5

namespace executorch::backends::aoti::slim::c10 {

// Packed container for TensorImpl sizes and strides.
// This design improves on the previous approach of using a pair of
// c10::SmallVector<int64_t, 5> by specializing for the operations we
// actually use and enforcing that the number of sizes is the same as
// the number of strides. The memory layout is as follows:
//
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
class SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  SizesAndStrides() {
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (STANDALONE_UNLIKELY(!isInline())) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(outOfLineStorage_);
    }
  }

  SizesAndStrides(const SizesAndStrides& rhs) : size_(rhs.size_) {
    if (STANDALONE_LIKELY(rhs.isInline())) {
      copyDataInline(rhs);
    } else {
      allocateOutOfLineStorage(size_);
      copyDataOutline(rhs);
    }
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

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (STANDALONE_LIKELY(rhs.isInline())) {
      if (STANDALONE_UNLIKELY(!isInline())) {
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

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides(SizesAndStrides&& rhs) noexcept : size_(rhs.size_) {
    if (STANDALONE_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }

    rhs.size_ = 0;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    if (STANDALONE_LIKELY(rhs.isInline())) {
      if (STANDALONE_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      copyDataInline(rhs);
    } else {
      // They're outline. We're going to steal their vector.
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

  size_t size() const noexcept {
    return size_;
  }

  const int64_t* sizes_data() const noexcept {
    if (STANDALONE_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  int64_t* sizes_data() noexcept {
    if (STANDALONE_LIKELY(isInline())) {
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

  void set_strides(IntArrayRef strides) {
    ET_DCHECK_MSG(
        strides.size() == size(),
        "strides size (%zu) must match size (%zu)",
        strides.size(),
        size());
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  const int64_t* strides_data() const noexcept {
    if (STANDALONE_LIKELY(isInline())) {
      return &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  int64_t* strides_data() noexcept {
    if (STANDALONE_LIKELY(isInline())) {
      return &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_begin() const noexcept {
    if (STANDALONE_LIKELY(isInline())) {
      return &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_iterator strides_begin() noexcept {
    if (STANDALONE_LIKELY(isInline())) {
      return &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
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

  // Size accessors.
  int64_t size_at(size_t idx) const noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t& size_at(size_t idx) noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t size_at_unchecked(size_t idx) const noexcept {
    return sizes_data()[idx];
  }

  int64_t& size_at_unchecked(size_t idx) noexcept {
    return sizes_data()[idx];
  }

  // Size accessors.
  int64_t stride_at(size_t idx) const noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t& stride_at(size_t idx) noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t stride_at_unchecked(size_t idx) const noexcept {
    return strides_data()[idx];
  }

  int64_t& stride_at_unchecked(size_t idx) noexcept {
    return strides_data()[idx];
  }

  void resize(size_t newSize) {
    const auto oldSize = size();
    if (newSize == oldSize) {
      return;
    }
    if (STANDALONE_LIKELY(
            newSize <= STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE &&
            isInline())) {
      if (oldSize < newSize) {
        const auto bytesToZero =
            (newSize - oldSize) * sizeof(inlineStorage_[0]);
        memset(&inlineStorage_[oldSize], 0, bytesToZero);
        memset(
            &inlineStorage_
                [STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
            0,
            bytesToZero);
      }
      size_ = newSize;
    } else {
      resizeSlowPath(newSize, oldSize);
    }
  }

 private:
  void resizeSlowPath(size_t newSize, size_t oldSize) {
    if (newSize <= STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {
      ET_DCHECK_MSG(
          !isInline(),
          "resizeSlowPath called when fast path should have been hit!");
      int64_t* tempStorage = outOfLineStorage_;
      memcpy(
          &inlineStorage_[0],
          &tempStorage[0],
          STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE *
              sizeof(inlineStorage_[0]));
      memcpy(
          &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
          &tempStorage[oldSize],
          STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE *
              sizeof(inlineStorage_[0]));
      // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
      // HAS BEEN OVERWRITTEN!
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(tempStorage);
    } else {
      if (isInline()) {
        // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
        // OVERWRITE inlineStorage_!
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
            &inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
            bytesToCopy);
        if (bytesToZero) {
          memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
        }
        outOfLineStorage_ = tempStorage;
      } else {
        const bool isGrowing = oldSize < newSize;
        if (isGrowing) {
          // Resize before shifting so that we have room.
          resizeOutOfLineStorage(newSize);
        }
        // Shift the old strides to their new starting point. Note
        // that this does not occur in the inline path above because
        // the stride starting point is not moving.
        memmove(
            outOfLineStorage_ + newSize,
            outOfLineStorage_ + oldSize,
            std::min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));
        if (!isGrowing) {
          // Resize after shifting so that we don't lose data.
          resizeOutOfLineStorage(newSize);
        } else {
          // Zero the end of the sizes portion.
          const auto bytesToZero =
              (newSize - oldSize) * sizeof(outOfLineStorage_[0]);
          memset(&outOfLineStorage_[oldSize], 0, bytesToZero);
          memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);
        }
      }
    }
    size_ = newSize;
  }

  bool isInline() const noexcept {
    return size_ <= STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    ET_DCHECK_MSG(rhs.isInline(), "rhs must be inline");
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
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

  void copyDataOutline(const SizesAndStrides& rhs) noexcept {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
  }

  size_t size_{1};
  union {
    int64_t* outOfLineStorage_;
    // NOLINTNEXTLINE(*c-array*)
    int64_t inlineStorage_[STANDALONE_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
  };
};

} // namespace executorch::backends::aoti::slim::c10
