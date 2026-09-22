/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/interface.h>

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <vector>

namespace cu = executorch::backends::cuda;
namespace aoti = executorch::backends::aoti;
namespace et = executorch::runtime;
using executorch::aten::ScalarType;

namespace {

et::Error one_tensor(aoti::AOTInductorModelContainerHandle, size_t* count) {
  *count = 1;
  return et::Error::Ok;
}

// Reverse rows using real captured device copies. The host computes their
// offsets, so replaying a graph for a different shape or stride is incorrect.
et::Error reverse_rows(
    aoti::AOTInductorModelContainerHandle container,
    et::etensor::Tensor** inputs,
    size_t,
    et::etensor::Tensor** outputs,
    size_t,
    aoti::AOTInductorStreamHandle stream,
    aoti::AOTIProxyExecutorHandle) {
  ++*reinterpret_cast<int*>(container);
  std::unique_ptr<aoti::slim::SlimTensor> input(
      reinterpret_cast<aoti::slim::SlimTensor*>(inputs[0]));
  auto* output = reinterpret_cast<aoti::slim::SlimTensor*>(outputs[0]);
  for (int64_t row = 0; row < input->size(0); ++row) {
    for (int64_t col = 0; col < input->size(1); ++col) {
      const auto src = (input->size(0) - 1 - row) * input->stride(0) +
          col * input->stride(1);
      const auto dst = row * output->stride(0) + col * output->stride(1);
      if (cudaMemcpyAsync(
              static_cast<int32_t*>(output->data_ptr()) + dst,
              static_cast<int32_t*>(input->data_ptr()) + src,
              sizeof(int32_t),
              cudaMemcpyDeviceToDevice,
              static_cast<cudaStream_t>(stream)) != cudaSuccess) {
        return et::Error::Internal;
      }
    }
  }
  return et::Error::Ok;
}

class CudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "Requires CUDA";
    }
    ASSERT_EQ(cudaMalloc(&input_, 16 * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_, 16 * sizeof(int32_t)), cudaSuccess);
    backend_ = et::get_backend_class("CudaBackend");
    ASSERT_NE(backend_, nullptr);
    handle_.container_handle =
        reinterpret_cast<aoti::AOTInductorModelContainerHandle>(&run_calls_);
    handle_.get_num_inputs = one_tensor;
    handle_.get_num_outputs = one_tensor;
    handle_.run = reverse_rows;
    handle_.run_single_threaded = reverse_rows;
    handle_.method_name = "reverse_rows";
    handle_.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
    handle_.cuda_graph_state.warmup_remaining = 3;
  }

  void TearDown() override {
    handle_.cuda_graph_state = cu::CudaGraphState{};
    (void)cudaFree(input_);
    (void)cudaFree(output_);
  }

  void check(
      int rows,
      int cols,
      bool replay,
      bool transpose_input = false,
      bool transpose_output = false,
      ScalarType input_type = ScalarType::Float,
      ScalarType output_type = ScalarType::Float,
      et::Error expected_error = et::Error::Ok) {
    std::vector<int32_t> values(rows * cols);
    std::iota(values.begin(), values.end(), ++input_version_ * 100);
    ASSERT_EQ(
        cudaMemcpy(
            input_,
            values.data(),
            values.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    const std::vector<int32_t> input_strides = transpose_input
        ? std::vector<int32_t>{1, rows}
        : std::vector<int32_t>{cols, 1};
    const std::vector<int32_t> output_strides = transpose_output
        ? std::vector<int32_t>{1, rows}
        : std::vector<int32_t>{cols, 1};
    const auto device =
        executorch::aten::Device(executorch::aten::DeviceType::CUDA, 0);
    auto input = executorch::extension::make_tensor_ptr(
        {rows, cols}, input_, {}, input_strides, input_type, device);
    auto output = executorch::extension::make_tensor_ptr(
        {rows, cols}, output_, {}, output_strides, output_type, device);
    et::EValue in(*input), out(*output);
    et::EValue* args[] = {&in, &out};
    et::BackendExecutionContext context;
    const int previous_calls = run_calls_;
    ASSERT_EQ(backend_->execute(context, &handle_, {args, 2}), expected_error);
    EXPECT_EQ(run_calls_, previous_calls + (replay ? 0 : 1));
    if (expected_error != et::Error::Ok) {
      return;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int32_t> actual(values.size()), expected(values.size());
    ASSERT_EQ(
        cudaMemcpy(
            actual.data(),
            output_,
            actual.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        expected[row * output_strides[0] + col * output_strides[1]] = values
            [(rows - 1 - row) * input_strides[0] + col * input_strides[1]];
      }
    }
    EXPECT_EQ(actual, expected);
  }

  cu::CudaDelegateHandle handle_{};
  et::BackendInterface* backend_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  int run_calls_ = 0;
  int input_version_ = 0;
};

TEST_F(CudaGraphTest, ReplaySurvivesShapeStrideAndTypeChanges) {
  for (int i = 0; i < 4; ++i) {
    check(2, 4, false);
  }
  ASSERT_EQ(handle_.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  const auto graph = handle_.cuda_graph_state.graph_exec;
  check(2, 4, true);
  for (int repeat = 0; repeat < 3; ++repeat) {
    check(1, 4, false);
    check(4, 4, false);
    check(4, 2, false); // Same byte count as the captured shape.
    check(2, 4, false, true);
    check(2, 4, false, false, true);
    check(2, 4, false, false, false, ScalarType::Int);
    check(2, 4, false, false, false, ScalarType::Float, ScalarType::Int);
    check(2, 4, true);
    EXPECT_EQ(handle_.cuda_graph_state.graph_exec, graph);
  }
}

TEST_F(CudaGraphTest, FailedCaptureFallsBackToOrdinaryExecution) {
  for (int i = 0; i < 3; ++i) {
    check(2, 4, false);
  }
  handle_.run_single_threaded =
      [](aoti::AOTInductorModelContainerHandle container,
         et::etensor::Tensor** inputs,
         size_t,
         et::etensor::Tensor**,
         size_t,
         aoti::AOTInductorStreamHandle,
         aoti::AOTIProxyExecutorHandle) {
        ++*reinterpret_cast<int*>(container);
        delete reinterpret_cast<aoti::slim::SlimTensor*>(inputs[0]);
        return et::Error::Internal;
      };
  check(
      2,
      4,
      false,
      false,
      false,
      ScalarType::Float,
      ScalarType::Float,
      et::Error::Internal);
  EXPECT_EQ(handle_.cuda_graph_state.phase, cu::CudaGraphPhase::Disabled);
  check(2, 4, false);
  check(1, 4, false);
  check(2, 4, false);
}

} // namespace
