/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/utils.h>

#include <gtest/gtest.h>

namespace slim = ::executorch::backends::aoti::slim;
using namespace ::executorch::runtime;
using ::executorch::backends::cuda::copy_slimtensor_to_device_etensor_async;

namespace {

class CudaOutputCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
    const std::vector<float> host = {0, 1, 2, 3, 4, 5, 6, 7};
    ASSERT_EQ(cudaMalloc(&source_, host.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&destination_, 4 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(
            source_,
            host.data(),
            host.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }

  void TearDown() override {
    cudaFree(source_);
    cudaFree(destination_);
  }

  slim::SlimTensor view(void* data, int64_t storage_offset) {
    return slim::from_blob(
        data,
        {2, 2},
        slim::c10::ScalarType::Float,
        slim::DEFAULT_CUDA_DEVICE,
        storage_offset);
  }

  Error copy(const slim::SlimTensor& source) {
    etensor::TensorImpl::SizesType sizes[] = {2, 2};
    etensor::TensorImpl::DimOrderType dim_order[] = {0, 1};
    etensor::TensorImpl::StridesType strides[] = {2, 1};
    etensor::TensorImpl impl(
        etensor::ScalarType::Float, 2, sizes, destination_, dim_order, strides);
    etensor::Tensor output(&impl);
    return copy_slimtensor_to_device_etensor_async(&source, &output, nullptr);
  }

  std::vector<float> copied() {
    std::vector<float> values(4);
    EXPECT_EQ(
        cudaMemcpy(
            values.data(),
            destination_,
            values.size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return values;
  }

  void* source_ = nullptr;
  void* destination_ = nullptr;
};

// AOTInductor can return a contiguous view that starts partway into a larger
// buffer.
TEST_F(
    CudaOutputCopyTest,
    CopiesAContiguousViewThatStartsPartwayIntoItsBuffer) {
  ASSERT_EQ(copy(view(source_, /*storage_offset=*/4)), Error::Ok);
  EXPECT_EQ(copied(), (std::vector<float>{4, 5, 6, 7}));
}

TEST_F(CudaOutputCopyTest, RejectsAViewOutsideItsStorage) {
  // from_blob sizes the storage to fit its view, so moving the view past the
  // end takes as_strided_. The moved view is still inside the allocation.
  slim::SlimTensor past_end = view(source_, /*storage_offset=*/2);
  past_end.as_strided_({2, 2}, {2, 1}, /*storage_offset=*/4);
  EXPECT_EQ(copy(past_end), Error::InvalidArgument);

  EXPECT_EQ(
      copy(view(static_cast<float*>(source_) + 4, /*storage_offset=*/-4)),
      Error::InvalidArgument);
}

} // namespace
