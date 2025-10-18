/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/tests/utils.h>
#include <executorch/runtime/core/error.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace executorch::backends::aoti;
using namespace executorch::backends::aoti::test;
using namespace executorch::runtime;
using executorch::runtime::etensor::Tensor;

// Test fixture for common shims tests
class CommonShimsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any existing cached metadata before each test
    cleanup_tensor_metadata();
  }

  void TearDown() override {
    // Clean up metadata and free any tensor data
    cleanup_tensor_metadata();
    for (auto& tensor : test_tensors_) {
      free_tensor_data(tensor.get());
    }
    test_tensors_.clear();
  }

  // Helper to create and track test tensors for cleanup
  Tensor* create_tracked_tensor(const std::vector<int64_t>& sizes) {
    auto tensor = create_test_tensor(sizes);
    Tensor* ptr = tensor.get();
    test_tensors_.push_back(tensor);
    return ptr;
  }

 private:
  std::vector<std::shared_ptr<Tensor>> test_tensors_;
};

// Test aoti_torch_get_sizes basic functionality
TEST_F(CommonShimsTest, GetSizesBasicFunctionality) {
  // Test 1D tensor
  auto tensor_1d = create_tracked_tensor({5});
  int64_t* sizes_ptr;
  AOTITorchError error = aoti_torch_get_sizes(tensor_1d, &sizes_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr, nullptr);
  EXPECT_EQ(sizes_ptr[0], 5);

  // Test 2D tensor
  auto tensor_2d = create_tracked_tensor({3, 4});
  error = aoti_torch_get_sizes(tensor_2d, &sizes_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr, nullptr);
  EXPECT_EQ(sizes_ptr[0], 3);
  EXPECT_EQ(sizes_ptr[1], 4);

  // Test 3D tensor
  auto tensor_3d = create_tracked_tensor({2, 3, 4});
  error = aoti_torch_get_sizes(tensor_3d, &sizes_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr, nullptr);
  EXPECT_EQ(sizes_ptr[0], 2);
  EXPECT_EQ(sizes_ptr[1], 3);
  EXPECT_EQ(sizes_ptr[2], 4);
}

// Test aoti_torch_get_strides basic functionality
TEST_F(CommonShimsTest, GetStridesBasicFunctionality) {
  // Test 1D tensor
  auto tensor_1d = create_tracked_tensor({5});
  int64_t* strides_ptr;
  AOTITorchError error = aoti_torch_get_strides(tensor_1d, &strides_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr, nullptr);
  EXPECT_EQ(strides_ptr[0], 1);

  // Test 2D tensor - row major: [3, 4] should have strides [4, 1]
  auto tensor_2d = create_tracked_tensor({3, 4});
  error = aoti_torch_get_strides(tensor_2d, &strides_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr, nullptr);
  EXPECT_EQ(strides_ptr[0], 4);
  EXPECT_EQ(strides_ptr[1], 1);

  // Test 3D tensor - row major: [2, 3, 4] should have strides [12, 4, 1]
  auto tensor_3d = create_tracked_tensor({2, 3, 4});
  error = aoti_torch_get_strides(tensor_3d, &strides_ptr);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr, nullptr);
  EXPECT_EQ(strides_ptr[0], 12);
  EXPECT_EQ(strides_ptr[1], 4);
  EXPECT_EQ(strides_ptr[2], 1);
}

// Test caching logic for sizes
TEST_F(CommonShimsTest, SizesCachingLogic) {
  auto tensor = create_tracked_tensor({2, 3, 4});

  // First call should cache the sizes
  int64_t* sizes_ptr1;
  AOTITorchError error = aoti_torch_get_sizes(tensor, &sizes_ptr1);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr1, nullptr);

  // Second call should return the same cached pointer
  int64_t* sizes_ptr2;
  error = aoti_torch_get_sizes(tensor, &sizes_ptr2);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(sizes_ptr1, sizes_ptr2); // Should be the exact same pointer

  // Values should still be correct
  EXPECT_EQ(sizes_ptr2[0], 2);
  EXPECT_EQ(sizes_ptr2[1], 3);
  EXPECT_EQ(sizes_ptr2[2], 4);
}

// Test caching logic for strides
TEST_F(CommonShimsTest, StridesCachingLogic) {
  auto tensor = create_tracked_tensor({2, 3, 4});

  // First call should cache the strides
  int64_t* strides_ptr1;
  AOTITorchError error = aoti_torch_get_strides(tensor, &strides_ptr1);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr1, nullptr);

  // Second call should return the same cached pointer
  int64_t* strides_ptr2;
  error = aoti_torch_get_strides(tensor, &strides_ptr2);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(strides_ptr1, strides_ptr2); // Should be the exact same pointer

  // Values should still be correct
  EXPECT_EQ(strides_ptr2[0], 12);
  EXPECT_EQ(strides_ptr2[1], 4);
  EXPECT_EQ(strides_ptr2[2], 1);
}

// Test that different tensors have different cached entries
TEST_F(CommonShimsTest, DifferentTensorsCacheSeparately) {
  auto tensor1 = create_tracked_tensor({2, 3});
  auto tensor2 = create_tracked_tensor({4, 5});

  // Get sizes for both tensors
  int64_t* sizes1_ptr;
  int64_t* sizes2_ptr;

  EXPECT_EQ(aoti_torch_get_sizes(tensor1, &sizes1_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_sizes(tensor2, &sizes2_ptr), Error::Ok);

  // Pointers should be different (different cache entries)
  EXPECT_NE(sizes1_ptr, sizes2_ptr);

  // Values should be correct
  EXPECT_EQ(sizes1_ptr[0], 2);
  EXPECT_EQ(sizes1_ptr[1], 3);
  EXPECT_EQ(sizes2_ptr[0], 4);
  EXPECT_EQ(sizes2_ptr[1], 5);

  // Test strides as well
  int64_t* strides1_ptr;
  int64_t* strides2_ptr;

  EXPECT_EQ(aoti_torch_get_strides(tensor1, &strides1_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor2, &strides2_ptr), Error::Ok);

  // Pointers should be different (different cache entries)
  EXPECT_NE(strides1_ptr, strides2_ptr);

  // Values should be correct
  EXPECT_EQ(strides1_ptr[0], 3);
  EXPECT_EQ(strides1_ptr[1], 1);
  EXPECT_EQ(strides2_ptr[0], 5);
  EXPECT_EQ(strides2_ptr[1], 1);
}

// Test cache persistence across multiple calls
TEST_F(CommonShimsTest, CachePersistence) {
  auto tensor = create_tracked_tensor({3, 4, 5});

  // Multiple calls to sizes should all return the same pointer
  int64_t* sizes_ptr1;
  int64_t* sizes_ptr2;
  int64_t* sizes_ptr3;

  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr1), Error::Ok);
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr2), Error::Ok);
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr3), Error::Ok);

  EXPECT_EQ(sizes_ptr1, sizes_ptr2);
  EXPECT_EQ(sizes_ptr2, sizes_ptr3);

  // Multiple calls to strides should all return the same pointer
  int64_t* strides_ptr1;
  int64_t* strides_ptr2;
  int64_t* strides_ptr3;

  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr1), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr2), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr3), Error::Ok);

  EXPECT_EQ(strides_ptr1, strides_ptr2);
  EXPECT_EQ(strides_ptr2, strides_ptr3);
}

// Test 0D tensor (scalar)
TEST_F(CommonShimsTest, ScalarTensor) {
  auto tensor_0d = create_tracked_tensor({});

  // Test sizes for 0D tensor
  int64_t* sizes_ptr;
  AOTITorchError error = aoti_torch_get_sizes(tensor_0d, &sizes_ptr);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr, nullptr);

  // Test strides for 0D tensor
  int64_t* strides_ptr;
  error = aoti_torch_get_strides(tensor_0d, &strides_ptr);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr, nullptr);

  // Cache should work for 0D tensors too
  int64_t* sizes_ptr2;
  error = aoti_torch_get_sizes(tensor_0d, &sizes_ptr2);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(sizes_ptr, sizes_ptr2);
}

// Test large tensor dimensions
TEST_F(CommonShimsTest, LargeTensorDimensions) {
  auto tensor = create_tracked_tensor({100, 200, 300, 400});

  // Test sizes
  int64_t* sizes_ptr;
  AOTITorchError error = aoti_torch_get_sizes(tensor, &sizes_ptr);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(sizes_ptr, nullptr);
  EXPECT_EQ(sizes_ptr[0], 100);
  EXPECT_EQ(sizes_ptr[1], 200);
  EXPECT_EQ(sizes_ptr[2], 300);
  EXPECT_EQ(sizes_ptr[3], 400);

  // Test strides - expected: [24000000, 120000, 400, 1]
  int64_t* strides_ptr;
  error = aoti_torch_get_strides(tensor, &strides_ptr);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(strides_ptr, nullptr);
  EXPECT_EQ(strides_ptr[0], 24000000);
  EXPECT_EQ(strides_ptr[1], 120000);
  EXPECT_EQ(strides_ptr[2], 400);
  EXPECT_EQ(strides_ptr[3], 1);
}

// Test that cleanup_tensor_metadata clears the cache
TEST_F(CommonShimsTest, CleanupFunctionality) {
  auto tensor = create_tracked_tensor({2, 3});

  // Cache some data
  int64_t* sizes_ptr1;
  int64_t* strides_ptr1;
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr1), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr1), Error::Ok);

  // Clear the cache
  cleanup_tensor_metadata();

  // Getting sizes/strides again should create new cache entries
  // (We can't directly test if the pointers are different since that would be
  // implementation-dependent, but we can at least verify the functions still
  // work)
  int64_t* sizes_ptr2;
  int64_t* strides_ptr2;
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr2), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr2), Error::Ok);

  // Values should still be correct
  EXPECT_EQ(sizes_ptr2[0], 2);
  EXPECT_EQ(sizes_ptr2[1], 3);
  EXPECT_EQ(strides_ptr2[0], 3);
  EXPECT_EQ(strides_ptr2[1], 1);
}

// Test mixed operations to ensure caches are independent
TEST_F(CommonShimsTest, IndependentCaches) {
  auto tensor = create_tracked_tensor({2, 3, 4});

  // Get sizes first
  int64_t* sizes_ptr1;
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr1), Error::Ok);

  // Get strides
  int64_t* strides_ptr1;
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr1), Error::Ok);

  // Get sizes again - should be cached
  int64_t* sizes_ptr2;
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr2), Error::Ok);
  EXPECT_EQ(sizes_ptr1, sizes_ptr2);

  // Get strides again - should be cached
  int64_t* strides_ptr2;
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr2), Error::Ok);
  EXPECT_EQ(strides_ptr1, strides_ptr2);

  // Sizes and strides pointers should be different (different caches)
  EXPECT_NE(sizes_ptr1, strides_ptr1);
}
