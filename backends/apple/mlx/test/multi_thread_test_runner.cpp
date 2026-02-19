/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Multi-threaded inference stress test for the MLX delegate.
 *
 * Loads a .pte model on multiple threads (each with its own Module instance)
 * and runs forward passes in parallel, verifying that all succeed and
 * produce correct outputs.
 *
 * The model accumulates via KV cache: with all-ones input and input_pos=[0],
 * call N produces output = 6.0 * N (all elements). Each thread has its own
 * Module (and cache state), so correctness is verified independently.
 *
 * The test expects a model exported by export_multi_thread_test_model.py.
 *
 * Build:
 *   cmake --preset mlx
 *   cmake --build cmake-out --target multi_thread_test_runner
 *
 * Usage:
 *   ET_TESTING_MODEL_PATH=/tmp/multi_thread_test_model.pte \
 *     ./cmake-out/backends/apple/mlx/test/multi_thread_test_runner
 *
 * Environment variables:
 *   ET_TESTING_MODEL_PATH     Path to .pte model file (required)
 *   ET_TESTING_NUM_THREADS    Number of parallel threads (default: 4)
 *   ET_PREDICTIONS_PER_THREAD Inferences per thread (default: 10)
 */

#include <gtest/gtest.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ::executorch::runtime;
using namespace ::executorch::extension;

const std::string kTestPTEPath = [] {
  if (const char* env_p = std::getenv("ET_TESTING_MODEL_PATH")) {
    return std::string(env_p);
  }
  return std::string("model.pte");
}();

const int kNumThreads = [] {
  if (const char* env_p = std::getenv("ET_TESTING_NUM_THREADS")) {
    try {
      return std::stoi(env_p);
    } catch (...) {
    }
  }
  return 4;
}();

const int kPredictionsPerThread = [] {
  if (const char* env_p = std::getenv("ET_PREDICTIONS_PER_THREAD")) {
    try {
      return std::stoi(env_p);
    } catch (...) {
    }
  }
  return 10;
}();

std::vector<TensorPtr> get_ones_inputs(Module& module) {
  const auto method_meta = module.method_meta("forward");
  const auto num_inputs = method_meta->num_inputs();

  std::vector<TensorPtr> tensors;
  tensors.reserve(num_inputs);

  for (auto index = 0; index < num_inputs; ++index) {
    const auto input_tag = method_meta->input_tag(index);

    switch (*input_tag) {
      case Tag::Tensor: {
        const auto tensor_meta = method_meta->input_tensor_meta(index);
        const auto sizes = tensor_meta->sizes();
        if (tensor_meta->scalar_type() == exec_aten::ScalarType::Long) {
          tensors.emplace_back(
              zeros({sizes.begin(), sizes.end()}, tensor_meta->scalar_type()));
        } else {
          tensors.emplace_back(
              ones({sizes.begin(), sizes.end()}, tensor_meta->scalar_type()));
        }
      } break;
      default:
        throw std::runtime_error(
            "Unsupported input tag at index " + std::to_string(index));
    }
  }
  return tensors;
}

struct ThreadResult {
  size_t success_count{0};
  size_t correctness_failures{0};
  std::string error_message;
};

void run_predict(
    int thread_id,
    const std::string& model_path,
    ThreadResult& result) {
  Module module(model_path);

  for (int pred = 0; pred < kPredictionsPerThread; pred++) {
    auto inputs = get_ones_inputs(module);
    for (int i = 0; i < inputs.size(); i++) {
      if (module.set_input(inputs[i], i) != Error::Ok) {
        std::cerr << "Thread " << thread_id << ", prediction " << pred
                  << ": set_input(" << i << ") failed" << std::endl;
        break;
      }
    }

    const auto forward_result = module.forward();

    if (!forward_result.ok()) {
      std::cerr << "Thread " << thread_id << ", prediction " << pred
                << ": forward() failed with error "
                << static_cast<int>(forward_result.error()) << std::endl;
      continue;
    }

    const auto outputs = forward_result.get();
    if (outputs.empty() || !outputs[0].isTensor()) {
      std::cerr << "Thread " << thread_id << ", prediction " << pred
                << ": no tensor output" << std::endl;
      continue;
    }

    const auto& output_tensor = outputs[0].toTensor();
    const float* data = output_tensor.const_data_ptr<float>();
    const float expected = 6.0f * (pred + 1);
    bool correct = true;
    for (ssize_t j = 0; j < output_tensor.numel(); j++) {
      if (std::fabs(data[j] - expected) > 1e-4f) {
        std::cerr << "Thread " << thread_id << ", prediction " << pred
                  << ": output[" << j << "] = " << data[j] << ", expected "
                  << expected << std::endl;
        correct = false;
        break;
      }
    }
    if (!correct) {
      result.correctness_failures++;
    }

    result.success_count++;
  }
}

TEST(MLXMultiThreadTest, LoadAndRunParallel) {
  ASSERT_FALSE(kTestPTEPath.empty()) << "ET_TESTING_MODEL_PATH must be set";
  ASSERT_GT(kNumThreads, 0) << "ET_TESTING_NUM_THREADS must be > 0";
  ASSERT_GT(kPredictionsPerThread, 0)
      << "ET_PREDICTIONS_PER_THREAD must be > 0";

  std::cout << "Running " << kNumThreads << " threads x "
            << kPredictionsPerThread
            << " predictions with model: " << kTestPTEPath << std::endl;

  std::vector<std::thread> threads(kNumThreads);
  std::vector<ThreadResult> results(kNumThreads);

  for (int i = 0; i < kNumThreads; i++) {
    threads[i] =
        std::thread([&, i]() { run_predict(i, kTestPTEPath, results[i]); });
  }
  for (int i = 0; i < kNumThreads; i++) {
    threads[i].join();
  }

  size_t total_success = 0;
  size_t total_correctness_failures = 0;
  for (int i = 0; i < kNumThreads; i++) {
    total_success += results[i].success_count;
    total_correctness_failures += results[i].correctness_failures;
  }

  const size_t total = kNumThreads * kPredictionsPerThread;
  std::cout << "Success: " << total_success << "/" << total << std::endl;
  std::cout << "Correctness failures: " << total_correctness_failures
            << std::endl;

  ASSERT_EQ(total_success, total) << "Some forward() calls failed";
  ASSERT_EQ(total_correctness_failures, 0) << "Some outputs were incorrect";
}
