/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::MemoryAllocator;
using torch::executor::testing::TensorFactory;

class TempMemoryAllocator final : public MemoryAllocator {
 private:
  // We allocate a little more than requested and use that memory as a node in
  // a linked list, pushing the allocated buffers onto a list that's iterated
  // and freed when the KernelRuntimeContext is destroyed.
  struct AllocationNode {
    void* data;
    AllocationNode* next;
  };

  AllocationNode* head_ = nullptr;

 public:
  TempMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // Allocate enough memory for the node, the data and the alignment bump.
    size_t alloc_size = sizeof(AllocationNode) + size + alignment;
    void* node_memory = malloc(alloc_size);

    // If allocation failed, log message and return nullptr.
    if (node_memory == nullptr) {
      ET_LOG(Error, "Failed to allocate %zu bytes", alloc_size);
      return nullptr;
    }

    // Compute data pointer.
    uint8_t* data_ptr =
        reinterpret_cast<uint8_t*>(node_memory) + sizeof(AllocationNode);

    // Align the data pointer.
    void* aligned_data_ptr = alignPointer(data_ptr, alignment);

    // Assert that the alignment didn't overflow the allocated memory.
    ET_DCHECK_MSG(
        reinterpret_cast<uintptr_t>(aligned_data_ptr) + size <=
            reinterpret_cast<uintptr_t>(node_memory) + alloc_size,
        "aligned_data_ptr %p + size %zu > node_memory %p + alloc_size %zu",
        aligned_data_ptr,
        size,
        node_memory,
        alloc_size);

    // Construct the node.
    AllocationNode* new_node = reinterpret_cast<AllocationNode*>(node_memory);
    new_node->data = aligned_data_ptr;
    new_node->next = head_;
    head_ = new_node;

    // Return the aligned data pointer.
    return head_->data;
  }

  void reset() override {
    AllocationNode* current = head_;
    while (current != nullptr) {
      AllocationNode* next = current->next;
      free(current);
      current = next;
    }
    head_ = nullptr;
  }

  ~TempMemoryAllocator() override {
    reset();
  }
};

std::tuple<Tensor&, Tensor&> op_topk_values(
    const Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  TempMemoryAllocator allocator = TempMemoryAllocator();
  executorch::runtime::KernelRuntimeContext context(nullptr, &allocator);
  return torch::executor::aten::topk_outf(
      context, input, k, dim, largest, sorted, values, indices);
}

class OpTopkValuesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpTopkValuesTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;
  TensorFactory<ScalarType::Long> tfLong;

  Tensor input =
      tfFloat.make({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  int64_t k = 2;
  int64_t dim = 0;
  bool largest = true;
  bool sorted = true;
  Tensor values = tfFloat.zeros({2, 2, 2});
  Tensor indices = tfLong.zeros({2, 2, 2});
  Tensor values_expected = tfFloat.make({2, 2, 2}, {9, 10, 11, 12, 5, 6, 7, 8});
  Tensor indices_expected = tfLong.make({2, 2, 2}, {2, 2, 2, 2, 1, 1, 1, 1});
  op_topk_values(input, k, dim, largest, sorted, values, indices);
  EXPECT_TENSOR_CLOSE(values, values_expected);
  EXPECT_TENSOR_EQ(indices, indices_expected);
}
