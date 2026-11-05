/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/platform.h>

#include <gtest/gtest.h>

namespace aoti = ::executorch::backends::aoti;
using namespace ::executorch::runtime;
using namespace ::executorch::backends::cuda;

namespace {

bool program_ran = false;

Error one_tensor(aoti::AOTInductorModelContainerHandle, size_t* count) {
  *count = 1;
  return Error::Ok;
}

// Stands in for the compiled program. Like the real one, it takes ownership of
// the input handles; it leaves the output handles in place.
Error fake_run(
    aoti::AOTInductorModelContainerHandle,
    aoti::Tensor** inputs,
    size_t num_inputs,
    aoti::Tensor**,
    size_t,
    aoti::AOTInductorStreamHandle,
    aoti::AOTIProxyExecutorHandle) {
  program_ran = true;
  for (size_t i = 0; i < num_inputs; ++i) {
    delete reinterpret_cast<aoti::slim::SlimTensor*>(inputs[i]);
  }
  return Error::Ok;
}

class CudaBackendExecuteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
    backend_ = get_backend_class(kCudaBackendId);
    ASSERT_NE(backend_, nullptr);
    handle_.get_num_inputs = one_tensor;
    handle_.get_num_outputs = one_tensor;
    handle_.run = fake_run;
    ASSERT_EQ(cudaMalloc(&input_, sizeof(float) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_, sizeof(float) * 4), cudaSuccess);
    program_ran = false;
  }

  void TearDown() override {
    cudaFree(input_);
    cudaFree(output_);
  }

  Error execute(void* input_data, void* output_data) {
    etensor::TensorImpl::SizesType sizes[] = {4};
    etensor::TensorImpl::DimOrderType dim_order[] = {0};
    etensor::TensorImpl::StridesType strides[] = {1};
    etensor::TensorImpl input_impl(
        etensor::ScalarType::Float,
        1,
        sizes,
        input_data,
        dim_order,
        strides,
        TensorShapeDynamism::STATIC,
        etensor::DeviceType::CUDA);
    etensor::TensorImpl output_impl(
        etensor::ScalarType::Float,
        1,
        sizes,
        output_data,
        dim_order,
        strides,
        TensorShapeDynamism::STATIC,
        etensor::DeviceType::CUDA);
    EValue input{etensor::Tensor(&input_impl)};
    EValue output{etensor::Tensor(&output_impl)};
    EValue* args[] = {&input, &output};
    BackendExecutionContext context;
    return backend_->execute(context, &handle_, Span<EValue*>(args, 2));
  }

  BackendInterface* backend_ = nullptr;
  CudaDelegateHandle handle_{};
  void* input_ = nullptr;
  void* output_ = nullptr;
};

// An output that is not memory planned has no buffer until the caller sets
// one. Running without it has to fail the call, not abort the process.
TEST_F(CudaBackendExecuteTest, OutputWithoutBufferIsAnError) {
  EXPECT_EQ(execute(input_, nullptr), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
}

TEST_F(CudaBackendExecuteTest, InputWithoutBufferIsAnError) {
  EXPECT_EQ(execute(nullptr, output_), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
}

TEST_F(CudaBackendExecuteTest, InputInHostMemoryIsAnError) {
  float host_input[4] = {};
  EXPECT_EQ(execute(host_input, output_), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
}

TEST_F(CudaBackendExecuteTest, OutputWithBufferRuns) {
  EXPECT_EQ(execute(input_, output_), Error::Ok);
  EXPECT_TRUE(program_ran);
}

} // namespace
