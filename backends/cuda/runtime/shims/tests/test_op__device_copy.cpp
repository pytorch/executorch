/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/kernels/portable/Functions.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#if (defined(__has_feature) && __has_feature(address_sanitizer)) || \
    defined(__SANITIZE_ADDRESS__)
#include <sanitizer/lsan_interface.h>
#define EXECUTORCH_CUDA_DEVICE_COPY_HAS_LSAN_INTERFACE 1
#else
#define EXECUTORCH_CUDA_DEVICE_COPY_HAS_LSAN_INTERFACE 0
#endif

#include <cstdint>
#include <memory>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::get_device_allocator;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

namespace {

struct CudaDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

CudaPtr allocate_cuda(size_t nbytes) {
  void* ptr = nullptr;
  const cudaError_t err = cudaMalloc(&ptr, nbytes);
  EXPECT_EQ(err, cudaSuccess) << "cudaMalloc failed";
  return CudaPtr(ptr);
}

bool is_cuda_available() {
#if EXECUTORCH_CUDA_DEVICE_COPY_HAS_LSAN_INTERFACE
  __lsan_disable();
#endif
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
#if EXECUTORCH_CUDA_DEVICE_COPY_HAS_LSAN_INTERFACE
  __lsan_enable();
#endif
  return err == cudaSuccess && device_count > 0;
}

std::vector<float> copy_cuda_to_host(const void* device_ptr, size_t numel) {
  std::vector<float> host(numel);
  const cudaError_t err = cudaMemcpy(
      host.data(), device_ptr, numel * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy D2H failed";
  return host;
}

void copy_host_to_cuda(const std::vector<float>& host, void* device_ptr) {
  const cudaError_t err = cudaMemcpy(
      device_ptr,
      host.data(),
      host.size() * sizeof(float),
      cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << "cudaMemcpy H2D failed";
}

class CudaDeviceCopyOpTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    ASSERT_NE(get_device_allocator(DeviceType::CUDA), nullptr)
        << "Linking cuda_backend should auto-register the CUDA allocator";
  }

  void SetUp() override {
    if (!is_cuda_available()) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA device copy op tests";
    }
  }

  Tensor& op_h2d_copy_out(const Tensor& self, Tensor& out) {
    return torch::executor::et_copy::_h2d_copy_outf(context_, self, out);
  }

  Tensor& op_d2h_copy_out(const Tensor& self, Tensor& out) {
    return torch::executor::et_copy::_d2h_copy_outf(context_, self, out);
  }

  KernelRuntimeContext context_;
};

} // namespace

TEST_F(CudaDeviceCopyOpTest, H2dCopyUsesRegisteredCudaAllocator) {
  std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto device_data = allocate_cuda(src_data.size() * sizeof(float));
  ASSERT_NE(device_data.get(), nullptr);

  int32_t sizes[] = {static_cast<int32_t>(src_data.size())};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};

  TensorImpl src_impl(
      ScalarType::Float,
      1,
      sizes,
      src_data.data(),
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor src(&src_impl);

  TensorImpl dst_impl(
      ScalarType::Float,
      1,
      sizes,
      device_data.get(),
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor dst(&dst_impl);

  Tensor& result = op_h2d_copy_out(src, dst);

  EXPECT_EQ(context_.failure_state(), Error::Ok);
  EXPECT_EQ(&result, &dst);
  EXPECT_EQ(copy_cuda_to_host(device_data.get(), src_data.size()), src_data);
}

TEST_F(CudaDeviceCopyOpTest, D2hCopyUsesRegisteredCudaAllocator) {
  const std::vector<float> expected = {5.0f, 6.0f, 7.0f, 8.0f};
  auto device_data = allocate_cuda(expected.size() * sizeof(float));
  ASSERT_NE(device_data.get(), nullptr);
  copy_host_to_cuda(expected, device_data.get());

  std::vector<float> dst_data(expected.size(), 0.0f);
  int32_t sizes[] = {static_cast<int32_t>(expected.size())};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};

  TensorImpl src_impl(
      ScalarType::Float,
      1,
      sizes,
      device_data.get(),
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor src(&src_impl);

  TensorImpl dst_impl(
      ScalarType::Float,
      1,
      sizes,
      dst_data.data(),
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor dst(&dst_impl);

  Tensor& result = op_d2h_copy_out(src, dst);

  EXPECT_EQ(context_.failure_state(), Error::Ok);
  EXPECT_EQ(&result, &dst);
  EXPECT_EQ(dst_data, expected);
}
