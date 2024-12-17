/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/utils/MemoryPool.h>

namespace vkcompute::utils {

// constexpr const size_t kUniquePtrAllocatorInitBlockSize = 16 * 1024;
// constexpr const size_t kUniquePtrAllocatorAlignment = 8;

// void MemoryPool::reset() noexcept(false) {
//   for (auto& block : blocks_) {
//     block.reset();
//   }
//   currentBlockIndex_ = 0;
// }

// MemoryPool::BlockInfo::BlockInfo(BlockInfo&& other) noexcept {
//   *this = other;
// }

// MemoryPool::BlockInfo& MemoryPool::BlockInfo::operator=(
//     MemoryPool::BlockInfo&& other) noexcept {
//   totalCapacity_ = other.totalCapacity_;
//   currentCapacity_ = other.currentCapacity_;
//   data_ = std::move(other.data_);
//   return *this;
// }

// MemoryPool::BlockInfo& MemoryPool::BlockInfo::operator=(
//     MemoryPool::BlockInfo& other) {
//   totalCapacity_ = other.totalCapacity_;
//   currentCapacity_ = other.currentCapacity_;
//   data_ = std::move(other.data_);
//   return *this;
// }

// void MemoryPool::BlockInfo::reset() noexcept(false) {
//   currentCapacity_ = totalCapacity_;
// }

// void* MemoryPool::alloc(size_t size) {
//   size = ((size + kUniquePtrAllocatorAlignment - 1) /
//           kUniquePtrAllocatorAlignment) *
//       kUniquePtrAllocatorAlignment;
//   if (blocks_.empty() || blocks_[currentBlockIndex_].currentCapacity_ < size)
//   {
//     //  find a block that has enough space
//     bool found = false;
//     for (uint32_t i = currentBlockIndex_ + 1; i < blocks_.size(); ++i) {
//       if (blocks_[i].currentCapacity_ >= size) {
//         currentBlockIndex_ = i;
//         found = true;
//         break;
//       }
//     }

//     if (!found) { // no block has enough space, create a new one
//       auto& blockInfo = blocks_.emplace_back(BlockInfo{});
//       blockInfo.totalCapacity_ =
//           std::max(size, kUniquePtrAllocatorInitBlockSize);
//       blockInfo.currentCapacity_ = blockInfo.totalCapacity_;
//       blockInfo.data_ =
//       std::make_unique<uint8_t[]>(blockInfo.totalCapacity_);
//       currentBlockIndex_ = blocks_.size() - 1;
//     }
//   }

//   auto& blockInfo = blocks_[currentBlockIndex_];
//   auto ret = blockInfo.data_.get() +
//       (blockInfo.totalCapacity_ - blockInfo.currentCapacity_);
//   // blockInfo.allocations_.push_back(reinterpret_cast<AllocatorBase*>(ret));
//   blockInfo.currentCapacity_ -= size;
//   return ret;
// }

} // namespace vkcompute::utils
