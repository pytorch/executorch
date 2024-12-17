/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

namespace vkcompute::utils {

/**
 * @class This class is designed to improving memory allocation for unique
 *pointer allocations for a class or derivatives of a class. Allocation is done
 *in blocks and every unique pointer allocation takes chunk from the block.
 *
 * This class is designed to be a fast alternative to calling new and delete for
 *every unique pointer. Its speed is mainly due to the fact that it allocates
 *memory in successive location in a block and does not try to reuse indivudual
 *memory location after its deallocated.
 **/
template <class BaseType, const uint32_t kInitBlockSize = 4 * 1024, const uint32_t kAllocAlignment = 16>
class MemoryPool final {
 public:
  MemoryPool() = default;
  virtual ~MemoryPool() = default;

  struct Deleter final // deleter
  {
    inline void operator()(BaseType* obj) const {
      if (obj) {
        obj->~BaseType();
      }
    }
  };

  struct alignas(kAllocAlignment) AlignedStruct final {
    uint8_t data[kAllocAlignment];
  };

  /**
   * @brief Allocates a new object of Type type by allocating memory from an
   *allocation block and returns a unique pointer to it.
   * @param ref - reference object to construct object from.
   **/
  template <class Type>
  std::unique_ptr<BaseType, MemoryPool::Deleter> make_unique(Type&& ref) {
    static_assert(
        kAllocAlignment >= alignof(Type),
        "Memory pool can only safely allocate object of alignment kAllocAlignment or less");
    auto ptr = std::unique_ptr<BaseType, MemoryPool::Deleter>(
        new (alloc(sizeof(Type))) Type(std::move(ref)), MemoryPool::Deleter());
    return ptr;
  }

  /**
   * @brief Reset all block allocation info, so it can be reused.
   * Call this function only after all the unique pointers allcated by this
   * block have been deallocated. Else it may result in memory leaks.
   **/
  void reset() noexcept(false) {
    for (auto& block : blocks_) {
      block.reset();
    }
    currentBlockIndex_ = 0;
  }

 private:
  struct BlockInfo {
    uint32_t totalCapacity_ = 0;
    uint32_t currentCapacity_ = 0;
    std::unique_ptr<AlignedStruct[]> data_{};

    BlockInfo() = default;
    ~BlockInfo() = default;

    BlockInfo(BlockInfo&& other) noexcept {
      *this = other;
    }

    BlockInfo& operator=(BlockInfo&& other) noexcept {
      totalCapacity_ = other.totalCapacity_;
      currentCapacity_ = other.currentCapacity_;
      data_ = std::move(other.data_);
      return *this;
    }

    BlockInfo& operator=(BlockInfo& other) {
      totalCapacity_ = other.totalCapacity_;
      currentCapacity_ = other.currentCapacity_;
      data_ = std::move(other.data_);
      return *this;
    }

    void reset() noexcept(false) {
      currentCapacity_ = totalCapacity_;
    }
  };

  std::vector<BlockInfo> blocks_{};
  uint32_t currentBlockIndex_ = 0;

  void* alloc(size_t size) {
    size =
        ((size + kAllocAlignment - 1) / kAllocAlignment) *
        kAllocAlignment;
    if (blocks_.empty() ||
        blocks_[currentBlockIndex_].currentCapacity_ < size) {
      //  find a block that has enough space
      bool found = false;
      for (uint32_t i = currentBlockIndex_ + 1; i < blocks_.size(); ++i) {
        if (blocks_[i].currentCapacity_ >= size) {
          currentBlockIndex_ = i;
          found = true;
          break;
        }
      }

      if (!found) { // no block has enough space, create a new one
        auto& blockInfo = blocks_.emplace_back(BlockInfo{});
        blockInfo.totalCapacity_ = std::max(size, kInitBlockSize);
        blockInfo.currentCapacity_ = blockInfo.totalCapacity_;
        blockInfo.data_ = std::make_unique<AlignedStruct[]>((blockInfo.totalCapacity_ + sizeof(AlignedStruct) - 1) / sizeof(AlignedStruct));
        currentBlockIndex_ = blocks_.size() - 1;
      }
    }

    auto& blockInfo = blocks_[currentBlockIndex_];
    auto ret = blockInfo.data_.get()->data + (blockInfo.totalCapacity_ - blockInfo.currentCapacity_);
    blockInfo.currentCapacity_ -= size;
    return ret;
  }
};

} // namespace vkcompute::utils
