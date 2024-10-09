/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/portable/cpu/util/allocate_tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
using ScalarType = exec_aten::ScalarType;

class AllocateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST(AllocateTest, AllocateTensor) {
  uint8_t* temp_allocator_ptr = (uint8_t*)malloc(2048);
  executorch::runtime::MemoryAllocator temp_allocator(2048, temp_allocator_ptr);
  executorch::runtime::KernelRuntimeContext ctx(nullptr, &temp_allocator);

  executorch::aten::SizesType sizes[3] = {1, 2, 3};
  executorch::aten::DimOrderType dim_order[3] = {0, 1, 2};
  executorch::aten::StridesType strides[3] = {3, 3, 1};

  torch::executor::ArrayRef<exec_aten::SizesType> sizes_ref(sizes, 3);
  torch::executor::ArrayRef<exec_aten::StridesType> strides_ref(strides, 3);
  torch::executor::ArrayRef<exec_aten::DimOrderType> dim_orders_ref(
      dim_order, 3);

  torch::executor::allocate_tensor(
      ctx, sizes, dim_order, strides, ScalarType::Float);

  free(temp_allocator_ptr);
}

TEST(AllocateTest, FailAllocateTensor) {
  torch::executor::runtime_init();

  uint8_t* temp_allocator_ptr = (uint8_t*)malloc(16);
  executorch::runtime::MemoryAllocator temp_allocator(16, temp_allocator_ptr);
  executorch::runtime::KernelRuntimeContext ctx(nullptr, &temp_allocator);

  executorch::aten::SizesType sizes[3] = {1, 2, 3};
  executorch::aten::DimOrderType dim_order[3] = {0, 1, 2};
  executorch::aten::StridesType strides[3] = {3, 3, 1};

  torch::executor::ArrayRef<exec_aten::SizesType> sizes_ref(sizes, 3);
  torch::executor::ArrayRef<exec_aten::StridesType> strides_ref(strides, 3);
  torch::executor::ArrayRef<exec_aten::DimOrderType> dim_orders_ref(
      dim_order, 3);

  ET_EXPECT_DEATH(
      torch::executor::allocate_tensor(
          ctx, sizes, dim_order, strides, ScalarType::Float),
      "Failed to malloc");

  free(temp_allocator_ptr);
}
