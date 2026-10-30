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
using namespace ::executorch::backends::cuda;

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

// AOTInductor can return a slice of a larger buffer, which is not contiguous.
TEST_F(CudaOutputCopyTest, CopiesAStridedView) {
  slim::SlimTensor columns = slim::from_blob(
      source_, {2, 4}, slim::c10::ScalarType::Float, slim::DEFAULT_CUDA_DEVICE);
  columns.as_strided_({2, 2}, {4, 1}, /*storage_offset=*/1);
  ASSERT_EQ(copy(columns), Error::Ok);
  EXPECT_EQ(copied(), (std::vector<float>{1, 2, 5, 6}));
}

TEST_F(CudaOutputCopyTest, RejectsAViewOutsideItsStorage) {
  // from_blob sizes the storage to fit its view, so moving the view past the
  // end takes as_strided_. The moved view is still inside the allocation.
  slim::SlimTensor past_end = view(source_, /*storage_offset=*/2);
  past_end.as_strided_({2, 2}, {2, 1}, /*storage_offset=*/4);
  EXPECT_EQ(copy(past_end), Error::InvalidArgument);

  // Its nbytes() would fit after the offset, but its last element would not.
  slim::SlimTensor columns = slim::from_blob(
      source_, {2, 4}, slim::c10::ScalarType::Float, slim::DEFAULT_CUDA_DEVICE);
  columns.as_strided_({2, 2}, {4, 1}, /*storage_offset=*/3);
  EXPECT_EQ(copy(columns), Error::InvalidArgument);

  // Its extent in bytes does not fit in size_t.
  columns.as_strided_({2, 2}, {(int64_t{1} << 62) - 1, 1}, 0);
  EXPECT_EQ(copy(columns), Error::InvalidArgument);

  EXPECT_EQ(
      copy(view(static_cast<float*>(source_) + 4, /*storage_offset=*/-4)),
      Error::InvalidArgument);
}

// Whether an output with these sizes and strides can be raw-copied into an
// ETensor that has this shape, dim order and strides from an earlier call.
bool raw_copyable(
    std::initializer_list<int64_t> output_sizes,
    std::initializer_list<int64_t> output_strides,
    std::vector<etensor::TensorImpl::SizesType> sizes,
    std::vector<etensor::TensorImpl::DimOrderType> dim_order,
    std::vector<etensor::TensorImpl::StridesType> strides) {
  std::vector<float> data(192);
  slim::SlimTensor output = slim::from_blob(
      data.data(), output_sizes, output_strides, slim::c10::ScalarType::Float);
  etensor::TensorImpl impl(
      etensor::ScalarType::Float,
      static_cast<ssize_t>(sizes.size()),
      sizes.data(),
      data.data(),
      dim_order.data(),
      strides.data());
  etensor::Tensor etensor(&impl);
  return output_is_raw_copyable(&output, &etensor);
}

TEST(CudaOutputRawCopyTest, AcceptsAnOutputLaidOutLikeItsETensor) {
  EXPECT_TRUE(raw_copyable(
      {2, 3, 4, 4},
      {48, 1, 12, 3},
      {2, 3, 4, 4},
      {0, 2, 3, 1},
      {48, 1, 12, 3}));
  // The ETensor's strides are stale from a smaller shape.
  EXPECT_TRUE(raw_copyable(
      {2, 3, 4, 4}, {48, 1, 12, 3}, {1, 3, 2, 2}, {0, 2, 3, 1}, {12, 1, 6, 3}));
  EXPECT_TRUE(raw_copyable(
      {2, 3, 4, 4}, {48, 16, 4, 1}, {1, 3, 2, 2}, {0, 1, 2, 3}, {12, 4, 2, 1}));
  // A size-1 dim may have any stride, as with a batch of one.
  EXPECT_TRUE(raw_copyable(
      {1, 3, 4, 4}, {1, 1, 12, 3}, {1, 3, 4, 4}, {0, 2, 3, 1}, {48, 1, 12, 3}));
  // An empty output has nothing to copy, whatever its strides.
  EXPECT_TRUE(raw_copyable(
      {2, 0, 4, 4},
      {16, 16, 4, 1},
      {2, 3, 4, 4},
      {0, 1, 2, 3},
      {48, 16, 4, 1}));
}

TEST(CudaOutputRawCopyTest, RejectsAnOutputLaidOutUnlikeItsETensor) {
  EXPECT_FALSE(raw_copyable(
      {2, 3, 4, 4},
      {48, 16, 4, 1},
      {2, 3, 4, 4},
      {0, 2, 3, 1},
      {48, 1, 12, 3}));
  EXPECT_FALSE(raw_copyable(
      {2, 3, 4, 4},
      {48, 1, 12, 3},
      {2, 3, 4, 4},
      {0, 1, 2, 3},
      {48, 16, 4, 1}));
  // Every row is followed by a gap, as in a slice of a wider tensor.
  EXPECT_FALSE(raw_copyable(
      {2, 3, 4, 4},
      {96, 32, 8, 1},
      {2, 3, 4, 4},
      {0, 1, 2, 3},
      {48, 16, 4, 1}));
  EXPECT_FALSE(raw_copyable(
      {3, 4, 4}, {16, 4, 1}, {1, 3, 4, 4}, {0, 1, 2, 3}, {48, 16, 4, 1}));
}

} // namespace
