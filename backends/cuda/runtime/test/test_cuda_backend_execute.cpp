/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/platform.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

namespace aoti = ::executorch::backends::aoti;
using namespace ::executorch::runtime;
using namespace ::executorch::backends::cuda;

namespace {

bool program_ran = false;
int run_device = -1;
void* run_stream = nullptr;
// The stream the current-stream slot of the run device held during the run.
void* run_slot_stream = nullptr;

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
    aoti::AOTInductorStreamHandle stream,
    aoti::AOTIProxyExecutorHandle) {
  program_ran = true;
  cudaGetDevice(&run_device);
  run_stream = stream;
  const auto slot = ::executorch::backends::cuda::getCurrentCUDAStream(
      static_cast<int8_t>(run_device));
  run_slot_stream = slot.ok() ? static_cast<void*>(slot.get()) : nullptr;
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
    run_device = -1;
  }

  void TearDown() override {
    cudaFree(input_);
    cudaFree(output_);
    cudaSetDevice(0);
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

// Has no keys, so an init that gets past the target_device check fails on
// the missing library rather than dereferencing a null map.
class EmptyDataMap final : public NamedDataMap {
 public:
  Result<const TensorLayout> get_tensor_layout(
      std::string_view) const override {
    return Error::NotFound;
  }
  Result<FreeableBuffer> get_data(std::string_view) const override {
    return Error::NotFound;
  }
  Error load_data_into(std::string_view, void*, size_t) const override {
    return Error::NotFound;
  }
  Result<uint32_t> get_num_keys() const override {
    return 0;
  }
  Result<const char*> get_key(uint32_t) const override {
    return Error::NotFound;
  }
};

Error init_with_target_device(BackendInterface* backend, const char* device) {
  CompileSpec specs[] = {
      {"method_name", {const_cast<char*>("forward"), 7}},
      {"target_device", {const_cast<char*>(device), std::strlen(device)}}};
  FreeableBuffer processed;
  EmptyDataMap data_map;
  BackendInitContext context(nullptr, nullptr, nullptr, &data_map);
  auto result =
      backend->init(context, &processed, ArrayRef<CompileSpec>(specs, 2));
  return result.error();
}

// The program runs on the caller's stream, which the delegate installs in
// the slot of its own device, so the shims that read that slot order their
// work with the program's. A handle on each device catches a slot written for
// the wrong device.
TEST_F(CudaBackendExecuteTest, CallerStreamIsTheRunDevicesCurrentStream) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  for (int device = 0; device < std::min(count, 2); ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    void* input = nullptr;
    void* output = nullptr;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaMalloc(&input, sizeof(float) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output, sizeof(float) * 4), cudaSuccess);
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    handle_.device_index = device;
    {
      ::executorch::extension::cuda::CallerStreamGuard caller(stream);
      EXPECT_EQ(execute(input, output), Error::Ok) << device;
    }
    EXPECT_EQ(run_device, device);
    EXPECT_EQ(run_stream, static_cast<void*>(stream)) << device;
    EXPECT_EQ(run_slot_stream, static_cast<void*>(stream)) << device;
    cudaStreamDestroy(stream);
    cudaFree(input);
    cudaFree(output);
  }
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
// A caller stream on another device would run the program's kernels against
// memory they cannot reach.
TEST_F(CudaBackendExecuteTest, CallerStreamOnAnotherDeviceIsAnError) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices";
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  {
    ::executorch::extension::cuda::CallerStreamGuard caller(stream);
    EXPECT_EQ(execute(input_, output_), Error::InvalidArgument);
  }
  EXPECT_FALSE(program_ran);
  cudaStreamDestroy(stream);
}
#endif

// A program lowered for a device this process cannot see fails at load time
// rather than with an illegal memory access at run time.
TEST_F(CudaBackendExecuteTest, TargetDeviceOutOfRangeIsAnError) {
  EXPECT_EQ(
      init_with_target_device(backend_, "cuda:99"), Error::InvalidArgument);
}

TEST_F(CudaBackendExecuteTest, MalformedTargetDeviceIsAnError) {
  for (const char* device :
       {"cpu", "cuda:", "cuda:x", "cuda:-1", "cuda:1x", "cuda0"}) {
    EXPECT_EQ(init_with_target_device(backend_, device), Error::InvalidArgument)
        << device;
  }
}

// A valid target device gets past the device checks to the missing library.
TEST_F(CudaBackendExecuteTest, ValidTargetDeviceReachesLibraryLoad) {
  for (const char* device : {"cuda", "cuda:0"}) {
    const Error error = init_with_target_device(backend_, device);
    EXPECT_NE(error, Error::Ok) << device;
    EXPECT_NE(error, Error::InvalidArgument) << device;
  }
}

// The delegate runs on its own device, and puts the caller's back afterwards.
TEST_F(CudaBackendExecuteTest, RunsOnItsDeviceWhenAnotherIsCurrent) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices";
  }
  for (const int device : {0, 1}) {
    void* input = nullptr;
    void* output = nullptr;
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&input, sizeof(float) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output, sizeof(float) * 4), cudaSuccess);
    handle_.device_index = device;
    const int current = 1 - device;
    ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
    run_device = -1;
    EXPECT_EQ(execute(input, output), Error::Ok);
    EXPECT_EQ(run_device, device);
    int after = -1;
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, current);
    cudaFree(input);
    cudaFree(output);
  }
}

TEST_F(CudaBackendExecuteTest, InputOnAnotherDeviceIsAnError) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices";
  }
  void* other = nullptr;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&other, sizeof(float) * 4), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  EXPECT_EQ(execute(other, output_), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
  cudaFree(other);
}

} // namespace
