/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/coreai/runtime/coreai_cache.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/runtime.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace et = executorch::runtime;
namespace aten = executorch::aten;
namespace fs = std::filesystem;

namespace {
size_t count(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (!value)
    return fallback;
  size_t result = 0;
  const char* end = value + std::strlen(value);
  auto parsed = std::from_chars(value, end, result);
  if (parsed.ec != std::errc() || parsed.ptr != end || result == 0)
    throw std::runtime_error(std::string("Invalid positive count: ") + name);
  return result;
}

void check(et::Error error, const char* operation) {
  if (error != et::Error::Ok)
    throw std::runtime_error(
        std::string(operation) +
        " failed: " + std::to_string(static_cast<int>(error)));
}

// TensorImpl borrows both its storage and shape metadata.
struct Input {
  std::vector<aten::Half> data;
  aten::TensorImpl::SizesType sizes[2] = {1, 32};
  aten::TensorImpl::DimOrderType order[2] = {0, 1};
  aten::TensorImpl::StridesType strides[2] = {32, 1};
  aten::TensorImpl impl;

  explicit Input(const std::vector<aten::Half>& values)
      : data(values),
        impl(aten::ScalarType::Half, 2, sizes, data.data(), order, strides) {}
  Input(const Input&) = delete;
  Input& operator=(const Input&) = delete;
};

void validate(
    const std::vector<et::EValue>& outputs,
    const std::vector<float>& expected) {
  if (outputs.size() != 1 || !outputs[0].isTensor())
    throw std::runtime_error("Expected one tensor output");
  const auto& output = outputs[0].toTensor();
  if (output.scalar_type() != aten::ScalarType::Half || output.dim() != 2 ||
      output.size(0) != 1 || output.size(1) != 16)
    throw std::runtime_error("Unexpected output dtype or shape");
  const auto* data = output.const_data_ptr<aten::Half>();
  for (size_t i = 0; i < expected.size(); ++i) {
    const float actual = static_cast<float>(data[i]);
    if (!std::isfinite(actual) ||
        std::abs(actual - expected[i]) > 0.01f + 0.01f * std::abs(expected[i]))
      throw std::runtime_error(
          "Incorrect output at index " + std::to_string(i));
  }
}

struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  size_t ready = 0;
  bool start = false;
  std::atomic<bool> stop{false};

  void cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    changed.notify_all();
  }
};

struct JoinWorkers {
  Gate& gate;
  std::vector<std::thread>& workers;
  ~JoinWorkers() {
    gate.cancel();
    for (auto& worker : workers)
      if (worker.joinable())
        worker.join();
  }
};

void round(
    const std::string& pte,
    const std::string& cache,
    const std::vector<aten::Half>& sample,
    const std::vector<float>& expected,
    size_t threads,
    size_t iterations) {
  Gate gate;
  std::vector<std::string> errors(threads);
  std::vector<size_t> completed(threads, 0);
  std::vector<std::thread> workers;
  workers.reserve(threads);
  JoinWorkers joiner{gate, workers};
  for (size_t index = 0; index < threads; ++index) {
    workers.emplace_back([&, index] {
      try {
        {
          std::unique_lock<std::mutex> lock(gate.mutex);
          ++gate.ready;
          gate.changed.notify_all();
          gate.changed.wait(
              lock, [&] { return gate.start || gate.stop.load(); });
          if (gate.stop)
            return;
        }
        Input input(sample);
        et::BackendOptions<1> options;
        et::LoadBackendOptionsMap option_map;
        check(
            options.set_option("coreai_assets_dir", cache.c_str()),
            "set option");
        check(
            option_map.set_options("CoreAIBackend", options.view()),
            "set backend options");
        executorch::extension::Module module(pte);
        check(module.load(option_map), "load");
        check(module.load_method("forward"), "load_method");
        for (size_t iteration = 0; iteration < iterations && !gate.stop;
             ++iteration) {
          auto result = module.forward(aten::Tensor(&input.impl));
          check(result.error(), "forward");
          validate(result.get(), expected);
          ++completed[index];
        }
      } catch (const std::exception& error) {
        errors[index] = error.what();
        gate.cancel();
      } catch (...) {
        errors[index] = "Unknown worker exception";
        gate.cancel();
      }
    });
  }
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (!gate.changed.wait_for(lock, std::chrono::seconds(30), [&] {
          return gate.ready == threads || gate.stop.load();
        }))
      throw std::runtime_error("Timed out waiting for worker start");
    gate.start = true;
  }
  gate.changed.notify_all();
  for (auto& worker : workers)
    worker.join();
  size_t total = 0;
  for (size_t index = 0; index < threads; ++index) {
    if (!errors[index].empty())
      throw std::runtime_error(
          "Worker " + std::to_string(index) + ": " + errors[index]);
    total += completed[index];
  }
  if (total != threads * iterations)
    throw std::runtime_error("Incomplete inference count");
  std::cout << "Validated " << total << " inferences" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  et::runtime_init();
  std::string cache;
  try {
    if (argc != 2)
      throw std::runtime_error(
          "Usage: coreai_multithread_test <fixture-directory>");
    const size_t threads = count("COREAI_STRESS_THREADS", 4);
    const size_t iterations = count("COREAI_STRESS_ITERATIONS", 10);
    if (threads > std::numeric_limits<size_t>::max() / iterations)
      throw std::runtime_error("Inference count overflow");
    const fs::path fixture(argv[1]);
    std::ifstream stream(fixture / "fixture.json");
    auto json = nlohmann::json::parse(stream);
    if (json.at("dtype") != "float16" ||
        json.at("input_shape").get<std::vector<int>>() !=
            std::vector<int>({1, 32}) ||
        json.at("output_shape").get<std::vector<int>>() !=
            std::vector<int>({1, 16}))
      throw std::runtime_error("Invalid fixture metadata");
    const auto values = json.at("input").get<std::vector<float>>();
    const auto expected = json.at("expected").get<std::vector<float>>();
    if (values.size() != 32 || expected.size() != 16)
      throw std::runtime_error("Invalid fixture sizes");
    std::vector<aten::Half> sample;
    for (float value : values) {
      if (!std::isfinite(value))
        throw std::runtime_error("Non-finite fixture input");
      sample.emplace_back(value);
    }
    for (float value : expected)
      if (!std::isfinite(value))
        throw std::runtime_error("Non-finite fixture output");
    char pattern[] = "/tmp/coreai-multithread-XXXXXX";
    const char* created = mkdtemp(pattern);
    if (!created)
      throw std::runtime_error("Cannot create isolated cache");
    cache = created;
    std::cout << threads << " threads x " << iterations
              << " iterations per round\n"
              << "Cache: " << cache << std::endl;
    for (const char* phase : {"cold", "warm"}) {
      std::cout << phase << " round" << std::endl;
      round(
          (fixture / "model.pte").string(),
          cache,
          sample,
          expected,
          threads,
          iterations);
    }
    check(
        executorch::backends::coreai::clear_cache(cache.c_str()),
        "clear test cache");
    fs::remove_all(cache);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\nCache retained for diagnosis: " << cache
              << std::endl;
    return 1;
  }
}
