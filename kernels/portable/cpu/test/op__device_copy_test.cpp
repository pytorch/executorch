/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Tests for et_copy._h2d_copy.out and et_copy._d2h_copy.out runtime kernels.
 *
 * Uses a MockDeviceAllocator to verify that the kernels correctly call
 * copy_host_to_device / copy_device_to_host via the DeviceAllocator interface,
 * and that device type is inferred from tensor metadata.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::DeviceAllocator;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::register_device_allocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::MockCudaAllocator;

using TensorShapeDynamism = executorch::runtime::TensorShapeDynamism;

// Forward declare the kernel functions from op__device_copy.cpp
namespace executorch::runtime::native {
Tensor&
_h2d_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out);
Tensor&
_d2h_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out);
} // namespace executorch::runtime::native

static MockCudaAllocator g_mock_cuda;

class OpDeviceCopyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    register_device_allocator(DeviceType::CUDA, &g_mock_cuda);
  }

  void SetUp() override {
    g_mock_cuda.h2d_count_ = 0;
    g_mock_cuda.d2h_count_ = 0;
    g_mock_cuda.last_h2d_size_ = 0;
    g_mock_cuda.last_d2h_size_ = 0;
    g_mock_cuda.last_h2d_index_ = -1;
    g_mock_cuda.last_d2h_index_ = -1;
  }
};

TEST_F(OpDeviceCopyTest, H2dCopyCopiesDataAndCallsAllocator) {
  // Set up a CPU source tensor with known data.
  float src_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t sizes[] = {4};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};
  TensorImpl src_impl(
      ScalarType::Float,
      1,
      sizes,
      src_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor src(&src_impl);

  // Set up a CUDA destination tensor (simulated with host memory).
  float dst_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
  TensorImpl dst_impl(
      ScalarType::Float,
      1,
      sizes,
      dst_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor dst(&dst_impl);

  KernelRuntimeContext ctx{};
  Tensor& result = executorch::runtime::native::_h2d_copy_out(ctx, src, dst);

  // Verify the allocator was called correctly.
  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_h2d_size_, 4 * sizeof(float));
  EXPECT_EQ(g_mock_cuda.last_h2d_index_, 0);

  // Verify data was copied (mock does a real memcpy).
  EXPECT_EQ(dst_data[0], 1.0f);
  EXPECT_EQ(dst_data[1], 2.0f);
  EXPECT_EQ(dst_data[2], 3.0f);
  EXPECT_EQ(dst_data[3], 4.0f);

  // Verify return value is the out tensor.
  EXPECT_EQ(&result, &dst);
}

TEST_F(OpDeviceCopyTest, D2hCopyCopiesDataAndCallsAllocator) {
  // Set up a CUDA source tensor with known data.
  float src_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
  int32_t sizes[] = {4};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};
  TensorImpl src_impl(
      ScalarType::Float,
      1,
      sizes,
      src_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor src(&src_impl);

  // Set up a CPU destination tensor.
  float dst_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
  TensorImpl dst_impl(
      ScalarType::Float,
      1,
      sizes,
      dst_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor dst(&dst_impl);

  KernelRuntimeContext ctx{};
  Tensor& result = executorch::runtime::native::_d2h_copy_out(ctx, src, dst);

  // Verify the allocator was called correctly.
  EXPECT_EQ(g_mock_cuda.d2h_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_d2h_size_, 4 * sizeof(float));
  EXPECT_EQ(g_mock_cuda.last_d2h_index_, 0);

  // Verify data was copied.
  EXPECT_EQ(dst_data[0], 5.0f);
  EXPECT_EQ(dst_data[1], 6.0f);
  EXPECT_EQ(dst_data[2], 7.0f);
  EXPECT_EQ(dst_data[3], 8.0f);

  EXPECT_EQ(&result, &dst);
}

TEST_F(OpDeviceCopyTest, H2dCopyWithDeviceIndex1) {
  // Verify device_index is correctly forwarded to the allocator.
  float src_data[] = {1.0f};
  float dst_data[] = {0.0f};
  int32_t sizes[] = {1};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};

  TensorImpl src_impl(
      ScalarType::Float,
      1,
      sizes,
      src_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor src(&src_impl);

  // Device index = 1 (e.g., cuda:1)
  TensorImpl dst_impl(
      ScalarType::Float,
      1,
      sizes,
      dst_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      1);
  Tensor dst(&dst_impl);

  KernelRuntimeContext ctx{};
  executorch::runtime::native::_h2d_copy_out(ctx, src, dst);

  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_h2d_index_, 1);
}

TEST_F(OpDeviceCopyTest, H2dCopyMultidimensionalTensor) {
  // Test with a 2D tensor [2, 3].
  float src_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float dst_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int32_t sizes[] = {2, 3};
  uint8_t dim_order[] = {0, 1};
  int32_t strides[] = {3, 1};

  TensorImpl src_impl(
      ScalarType::Float,
      2,
      sizes,
      src_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor src(&src_impl);

  TensorImpl dst_impl(
      ScalarType::Float,
      2,
      sizes,
      dst_data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor dst(&dst_impl);

  KernelRuntimeContext ctx{};
  executorch::runtime::native::_h2d_copy_out(ctx, src, dst);

  EXPECT_EQ(g_mock_cuda.h2d_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_h2d_size_, 6 * sizeof(float));

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(dst_data[i], src_data[i]);
  }
}
