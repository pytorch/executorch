/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * ATen-mode tests for et_copy._h2d_copy.out / et_copy._d2h_copy.out.
 *
 * In ATen mode the memory-planned at::Tensor does not carry the runtime's
 * planned device metadata, so the kernels take the copy direction from the op
 * identity (_h2d vs _d2h) and route through the registered non-CPU
 * DeviceAllocator. These tests use a MockCudaAllocator (host memory) so they
 * exercise direction-from-op-identity, the current-device index (-1), and the
 * resize-before-copy behavior without requiring a GPU.
 *
 * The portable-mode kernels are covered separately in op__device_copy_test.cpp,
 * whose TensorImpl-based construction does not compile in ATen mode.
 */

#include <gtest/gtest.h>

#include <exception>

#include <ATen/ATen.h>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::Tensor;
using executorch::runtime::get_device_allocator;
using executorch::runtime::register_device_allocator;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::MockCudaAllocator;

// The ATen-mode custom-op kernels, as declared by the codegen `kernel_name`
// (torch::executor::_h2d_copy_out -> torch::executor::native::_h2d_copy_out).
// The 2-arg (contextless) overloads are the ones bound in ATen mode.
namespace torch {
namespace executor {
namespace native {
Tensor& _h2d_copy_out(const Tensor& self, Tensor& out);
Tensor& _d2h_copy_out(const Tensor& self, Tensor& out);
} // namespace native
} // namespace executor
} // namespace torch

namespace {

MockCudaAllocator& mock_cuda() {
  static MockCudaAllocator instance;
  return instance;
}

class OpDeviceCopyAtenTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    if (get_device_allocator(DeviceType::CUDA) == nullptr) {
      register_device_allocator(&mock_cuda());
    }
    // The registry has no unregister/replace. If a real CUDA allocator was
    // registered first (e.g. this test is linked into a binary that pulls in
    // the CUDA backend, whose static init registers CudaAllocator::instance()),
    // the mock is not installed and these tests cannot observe the mock
    // counters. Skip rather than fail in that configuration; in this CPU-only
    // target the mock is always the registered allocator.
    if (get_device_allocator(DeviceType::CUDA) != &mock_cuda()) {
      GTEST_SKIP()
          << "a non-mock CUDA allocator is registered; these tests require "
          << "MockCudaAllocator and the registry cannot be replaced";
    }
  }

  void SetUp() override {
    mock_cuda().reset();
  }
};

// H2D takes direction from the op identity: it must call copy_host_to_device
// (not d2h) and forward the current-device index (-1), regardless of the
// at::Tensor device metadata (which is CPU in ATen mode).
TEST_F(OpDeviceCopyAtenTest, H2dCopiesHostToDeviceByOpIdentity) {
  const at::Tensor self = at::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor out = at::zeros({4});

  Tensor& result = torch::executor::native::_h2d_copy_out(self, out);

  EXPECT_EQ(mock_cuda().h2d_count_, 1);
  EXPECT_EQ(mock_cuda().d2h_count_, 0);
  EXPECT_EQ(mock_cuda().last_h2d_size_, 4 * sizeof(float));
  EXPECT_EQ(mock_cuda().last_h2d_index_, -1);
  EXPECT_TRUE(at::equal(out, self));
  EXPECT_EQ(&result, &out);
}

// D2H is the mirror: it must call copy_device_to_host and forward index -1.
TEST_F(OpDeviceCopyAtenTest, D2hCopiesDeviceToHostByOpIdentity) {
  const at::Tensor self = at::tensor({5.0f, 6.0f, 7.0f, 8.0f});
  at::Tensor out = at::zeros({4});

  Tensor& result = torch::executor::native::_d2h_copy_out(self, out);

  EXPECT_EQ(mock_cuda().d2h_count_, 1);
  EXPECT_EQ(mock_cuda().h2d_count_, 0);
  EXPECT_EQ(mock_cuda().last_d2h_size_, 4 * sizeof(float));
  EXPECT_EQ(mock_cuda().last_d2h_index_, -1);
  EXPECT_TRUE(at::equal(out, self));
  EXPECT_EQ(&result, &out);
}

// out is resized to self's shape before the copy: start out at a different
// shape so the resize is actually observable.
TEST_F(OpDeviceCopyAtenTest, H2dResizesOutToSelf) {
  const at::Tensor self = at::tensor({1.0f, 2.0f, 3.0f});
  at::Tensor out = at::zeros({8});
  ASSERT_NE(out.sizes(), self.sizes());

  torch::executor::native::_h2d_copy_out(self, out);

  EXPECT_EQ(out.sizes(), self.sizes());
  EXPECT_EQ(mock_cuda().h2d_count_, 1);
  EXPECT_EQ(mock_cuda().last_h2d_size_, 3 * sizeof(float));
  EXPECT_TRUE(at::equal(out, self));
}

// If self is larger than out's backing storage, the kernel must reject the copy
// rather than overrun out's storage (ATen resize_tensor only updates sizes, it
// does not grow storage). It raises a catchable exception (not an abort) so a
// libtorch host can recover.
TEST_F(OpDeviceCopyAtenTest, H2dRejectsWhenSelfExceedsOutStorage) {
  const at::Tensor self = at::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor out = at::zeros({2}); // storage smaller than self
  EXPECT_THROW(
      torch::executor::native::_h2d_copy_out(self, out), std::exception);
  EXPECT_EQ(mock_cuda().h2d_count_, 0);
}

// Same overrun guard for the d2h direction. Both directions share device_copy,
// but assert d2h explicitly so the guard survives any future un-sharing.
TEST_F(OpDeviceCopyAtenTest, D2hRejectsWhenSelfExceedsOutStorage) {
  const at::Tensor self = at::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor out = at::zeros({2}); // storage smaller than self
  EXPECT_THROW(
      torch::executor::native::_d2h_copy_out(self, out), std::exception);
  EXPECT_EQ(mock_cuda().d2h_count_, 0);
}

// A dtype mismatch would make the raw byte copy meaningless, so the kernel must
// reject rather than copy.
TEST_F(OpDeviceCopyAtenTest, H2dRejectsDtypeMismatch) {
  const at::Tensor self = at::tensor({1.0f, 2.0f});
  at::Tensor out = at::zeros({2}, at::kInt);
  EXPECT_THROW(
      torch::executor::native::_h2d_copy_out(self, out), std::exception);
  EXPECT_EQ(mock_cuda().h2d_count_, 0);
}

// A zero-size copy is a valid no-op: it must not touch the allocator (whose
// null-pointer checks would otherwise reject empty tensors).
TEST_F(OpDeviceCopyAtenTest, H2dZeroSizeIsNoOp) {
  const at::Tensor self = at::zeros({0});
  at::Tensor out = at::zeros({0});

  torch::executor::native::_h2d_copy_out(self, out);

  EXPECT_EQ(mock_cuda().h2d_count_, 0);
  EXPECT_EQ(out.numel(), 0);
}

} // namespace
