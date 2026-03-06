/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * TensorRT Executor Runner - runs ExecuTorch models with TensorRT delegation.
 *
 * Usage:
 *   tensorrt_executor_runner --model_path=model_tensorrt.pte
 *   tensorrt_executor_runner --model_path=model.pte --num_executions=10
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Tag;

namespace {

struct RunnerArgs {
  std::string model_path = "model.pte";
  uint32_t num_executions = 1;
  bool verbose = false;
};

bool parse_args(int argc, char** argv, RunnerArgs& args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printf(
          "Usage: tensorrt_executor_runner [options]\n"
          "Options:\n"
          "  --model_path=PATH    Path to .pte model (default: model.pte)\n"
          "  --num_executions=N   Number of inference runs (default: 1)\n"
          "  --verbose            Enable verbose output\n"
          "  --help               Show this message\n");
      return false;
    } else if (arg.rfind("--model_path=", 0) == 0) {
      args.model_path = arg.substr(13);
    } else if (arg.rfind("--num_executions=", 0) == 0) {
      args.num_executions = static_cast<uint32_t>(std::stoul(arg.substr(17)));
    } else if (arg == "--verbose" || arg == "-v") {
      args.verbose = true;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  RunnerArgs args;
  if (!parse_args(argc, argv, args)) {
    return 1;
  }

  ET_LOG(Info, "TensorRT Executor Runner");
  ET_LOG(Info, "Model: %s", args.model_path.c_str());
  ET_LOG(Info, "Executions: %u", args.num_executions);

  Module module(args.model_path, Module::LoadMode::File);

  auto meta_result = module.method_meta("forward");
  if (!meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata");
    return 1;
  }
  auto meta = meta_result.get();

  // Create ones-filled inputs matching the model's expected shapes and dtypes.
  std::vector<executorch::extension::TensorPtr> input_tensors;
  std::vector<std::vector<float>> float_data;
  std::vector<std::vector<int64_t>> int64_data;
  std::vector<EValue> inputs;
  for (size_t i = 0; i < meta.num_inputs(); ++i) {
    auto tag = meta.input_tag(i);
    if (tag.ok() && tag.get() == Tag::Tensor) {
      auto tmeta = meta.input_tensor_meta(i);
      if (tmeta.ok()) {
        auto sizes = tmeta.get().sizes();
        auto dtype = tmeta.get().scalar_type();
        auto sizes_vec = std::vector<executorch::aten::SizesType>(
            sizes.begin(), sizes.end());
        int64_t numel = 1;
        for (auto s : sizes_vec) {
          numel *= s;
        }

        executorch::extension::TensorPtr tensor;
        if (dtype == ScalarType::Long || dtype == ScalarType::Int) {
          int64_data.emplace_back(numel, 1);
          tensor = executorch::extension::make_tensor_ptr(
              sizes_vec, int64_data.back());
        } else {
          float_data.emplace_back(numel, 1.0f);
          tensor = executorch::extension::make_tensor_ptr(
              sizes_vec, float_data.back());
        }
        inputs.push_back(EValue(*tensor));
        input_tensors.push_back(std::move(tensor));
      }
    }
  }

  if (args.verbose) {
    ET_LOG(Info, "Inputs prepared: %zu tensors", inputs.size());
  }

  // Warmup
  constexpr uint32_t kWarmupRuns = 3;
  for (uint32_t i = 0; i < kWarmupRuns; ++i) {
    auto r = module.forward(inputs);
    if (!r.ok()) {
      ET_LOG(Error, "Warmup execution failed");
      return 1;
    }
  }

  // Benchmark
  et_timestamp_t total_time = 0;
  for (uint32_t i = 0; i < args.num_executions; ++i) {
    auto start = executorch::runtime::pal_current_ticks();
    auto r = module.forward(inputs);
    auto end = executorch::runtime::pal_current_ticks();
    if (!r.ok()) {
      ET_LOG(Error, "Execution %u failed", i);
      return 1;
    }
    total_time += end - start;
  }

  auto ratio = et_pal_ticks_to_ns_multiplier();
  double total_ms = static_cast<double>(total_time) * ratio.numerator /
      ratio.denominator / 1000000.0;
  double avg_ms = total_ms / args.num_executions;

  ET_LOG(
      Info,
      "Executed %u time(s) in %.3f ms total (%.3f ms avg)",
      args.num_executions,
      total_ms,
      avg_ms);

  // Print output summary
  auto result = module.forward(inputs);
  if (result.ok()) {
    auto outputs = result.get();
    ET_LOG(Info, "Outputs: %zu values", outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].isTensor()) {
        const auto& t = outputs[i].toTensor();
        printf("  Output %zu: shape=[", i);
        for (size_t d = 0; d < t.dim(); ++d) {
          if (d > 0)
            printf(", ");
          printf("%d", static_cast<int>(t.size(d)));
        }
        printf("], dtype=%d\n", static_cast<int>(t.scalar_type()));
      }
    }
  }

  return 0;
}
