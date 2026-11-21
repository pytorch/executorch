/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#if defined(EXECUTORCH_BUILD_EXTENSION_LLM)
#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>
#endif
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/platform.h>

#include <gtest/gtest.h>

namespace aoti = ::executorch::backends::aoti;
using namespace ::executorch::runtime;
using namespace ::executorch::backends::cuda;

namespace {

bool program_ran = false;
Error program_error = Error::Ok;

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
  return program_error;
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
    program_error = Error::Ok;
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

TEST_F(
    CudaBackendExecuteTest,
    GraphMetricsCountCapturesAndReplayAcrossRecapture) {
  if (!handle_.cuda_graph_state.start_warmup()) {
    GTEST_SKIP() << "CUDA graph allocator unavailable";
  }
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(execute(input_, output_), Error::Ok);
  }
  const auto first = handle_.cuda_graph_state.metrics;
  EXPECT_EQ(first.captures, 1);
  EXPECT_GT(first.replays, first.captures);
  EXPECT_EQ(first.capture_failures, 0);
  handle_.cuda_graph_state.recapture();
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(execute(input_, output_), Error::Ok);
  }
  EXPECT_EQ(handle_.cuda_graph_state.metrics.captures, 2);
  EXPECT_GT(handle_.cuda_graph_state.metrics.replays, first.replays);
  EXPECT_EQ(handle_.cuda_graph_state.metrics.capture_failures, 0);
}

TEST(CudaGraphStateTest, MetricsSurviveMoveAndResourceRelease) {
  CudaGraphState source;
  source.metrics = {2, 11, 1};
  CudaGraphState moved(std::move(source));
  EXPECT_EQ(source.metrics.captures, 0);
  EXPECT_EQ(moved.metrics.captures, 2);
  moved.release();
  EXPECT_EQ(moved.metrics.replays, 11);
  CudaGraphState assigned;
  assigned = std::move(moved);
  EXPECT_EQ(moved.metrics.replays, 0);
  EXPECT_EQ(assigned.metrics.captures, 2);
  EXPECT_EQ(assigned.metrics.replays, 11);
  EXPECT_EQ(assigned.metrics.capture_failures, 1);
}

#if defined(EXECUTORCH_BUILD_EXTENSION_LLM)
class PreparedCache final : public CudaKVCache {
 public:
  Result<bool> note_handle(CudaDelegateHandle*) override {
    return false;
  }
  void forget_handle(CudaDelegateHandle*) override {}
  Error rebind_for_execute(CudaDelegateHandle*) override {
    return rebind_error;
  }
  Error prepare_step(int64_t width, cudaStream_t) override {
    ++prepares;
    EXPECT_EQ(width, 4);
    if (prepared) {
      return Error::InvalidState;
    }
    prepared = prepare_error == Error::Ok;
    return prepare_error;
  }
  Error commit_step(int64_t width) override {
    ++commits;
    EXPECT_EQ(width, 4);
    if (commit_error == Error::Ok) {
      prepared = false;
    }
    return commit_error;
  }
  Error abort_step() override {
    ++aborts;
    if (abort_error == Error::Ok) {
      prepared = false;
    }
    return abort_error;
  }
  OffGraphKVMetrics metrics() const override {
    return {};
  }

  bool prepared = false;
  int prepares = 0;
  int commits = 0;
  int aborts = 0;
  Error prepare_error = Error::Ok;
  Error rebind_error = Error::Ok;
  Error commit_error = Error::Ok;
  Error abort_error = Error::Ok;
};

TEST_F(CudaBackendExecuteTest, FailedPreparationDoesNotAbort) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  cache.prepare_error = Error::MemoryAllocationFailed;
  EXPECT_EQ(execute(input_, output_), Error::MemoryAllocationFailed);
  EXPECT_FALSE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.prepares, 1);
  EXPECT_EQ(cache.commits, 0);
  EXPECT_EQ(cache.aborts, 0);
}

TEST_F(CudaBackendExecuteTest, FailedRebindAbortsAndAllowsNextForward) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  cache.rebind_error = Error::InvalidArgument;
  EXPECT_EQ(execute(input_, output_), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.aborts, 1);
  EXPECT_EQ(cache.commits, 0);

  cache.rebind_error = Error::Ok;
  EXPECT_EQ(execute(input_, output_), Error::Ok);
  EXPECT_TRUE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.prepares, 2);
  EXPECT_EQ(cache.commits, 1);
  EXPECT_EQ(cache.aborts, 1);
}

TEST_F(CudaBackendExecuteTest, FailedProgramAbortsAndAllowsNextForward) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  program_error = Error::Internal;
  EXPECT_EQ(execute(input_, output_), Error::Internal);
  EXPECT_TRUE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.commits, 0);
  EXPECT_EQ(cache.aborts, 1);

  program_error = Error::Ok;
  EXPECT_EQ(execute(input_, output_), Error::Ok);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.prepares, 2);
  EXPECT_EQ(cache.commits, 1);
  EXPECT_EQ(cache.aborts, 1);
}

TEST_F(CudaBackendExecuteTest, FailedCommitAbortsAndAllowsNextForward) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  cache.commit_error = Error::Internal;
  EXPECT_EQ(execute(input_, output_), Error::Internal);
  EXPECT_TRUE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.commits, 1);
  EXPECT_EQ(cache.aborts, 1);

  cache.commit_error = Error::Ok;
  EXPECT_EQ(execute(input_, output_), Error::Ok);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.prepares, 2);
  EXPECT_EQ(cache.commits, 2);
  EXPECT_EQ(cache.aborts, 1);
}

TEST_F(CudaBackendExecuteTest, SuccessfulForwardCommitsWithoutAbort) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  EXPECT_EQ(execute(input_, output_), Error::Ok);
  EXPECT_TRUE(program_ran);
  EXPECT_FALSE(cache.prepared);
  EXPECT_EQ(cache.prepares, 1);
  EXPECT_EQ(cache.commits, 1);
  EXPECT_EQ(cache.aborts, 0);
}

TEST_F(CudaBackendExecuteTest, FailedAbortPreservesOriginalError) {
  PreparedCache cache;
  handle_.kv_cache = &cache;
  cache.rebind_error = Error::InvalidArgument;
  cache.abort_error = Error::Internal;
  EXPECT_EQ(execute(input_, output_), Error::InvalidArgument);
  EXPECT_FALSE(program_ran);
  EXPECT_TRUE(cache.prepared);
  EXPECT_EQ(cache.aborts, 1);
}
#endif

} // namespace
