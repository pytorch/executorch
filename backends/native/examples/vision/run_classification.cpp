// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Run an image-classification PTN on the Vulkan engine and optionally compare
// all logits with an eager reference. Labels are presentation-only.
// Exit codes: 0 success, 1 invalid arguments, 2 execution error, 3 mismatch.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/native/examples/runner_utils/runner_utils.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>
#include <gflags/gflags.h>

DEFINE_string(package, "", "Path to the *.ptn to run. Required.");
DEFINE_string(input, "", "Path to the input tensor, raw float32. Required.");
DEFINE_string(
    expected,
    "",
    "Reference logits to compare against, raw float32.");
DEFINE_string(categories, "", "Newline-separated class labels, for display.");
DEFINE_string(method, "", "Method to run. Defaults to the first one.");
DEFINE_double(rtol, 1e-3, "Relative tolerance for the logits comparison.");
DEFINE_double(atol, 1e-3, "Absolute tolerance for the logits comparison.");
namespace {

std::string label_of(const std::vector<std::string>& labels, int index) {
  if (index >= 0 && index < static_cast<int>(labels.size())) {
    return labels[index];
  }
  return "class_" + std::to_string(index);
}

void print_top5(
    const std::vector<float>& logits,
    const std::vector<std::string>& labels) {
  const std::vector<int> order = ptn::runner_utils::top_k(logits, 5);
  std::printf("top-5 (class : logit):\n");
  for (const int index : order) {
    std::printf(
        "  %4d  %-28s %10.4f\n",
        index,
        label_of(labels, index).c_str(),
        logits[index]);
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Run an image-classification .ptn on the Vulkan engine and check its "
      "logits against the eager reference when --expected is given.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_package.empty() || FLAGS_input.empty()) {
    std::fprintf(stderr, "error: --package and --input are required\n");
    return 1;
  }

  try {
    const auto package =
        std::make_shared<const ptn::Package>(ptn::Package::load(FLAGS_package));
    const ptn::ByteSpan program_bytes = package->program_bytes();
    const auto program = std::make_shared<const ptn::Program>(
        ptn::Program::load(program_bytes.data(), program_bytes.size()));

    const std::vector<std::string> names = program->method_names();
    if (names.empty()) {
      std::fprintf(stderr, "error: package holds no methods\n");
      return 2;
    }
    const std::string method_name =
        FLAGS_method.empty() ? names[0] : FLAGS_method;
    std::printf(
        "package:   %s (program %zu bytes, %zu constants / %zu bytes)\n",
        FLAGS_package.c_str(),
        program_bytes.size(),
        package->keys().size(),
        package->constant_bytes());

    const std::unique_ptr<ptn::VulkanEngineHost> host =
        ptn::VulkanEngineHost::create();
    const std::unique_ptr<ptn::VulkanEngineContext> context =
        host->create_vulkan_context(program, package);
    std::printf(
        "engine:    %s on %s\n",
        host->name().c_str(),
        host->device_name().c_str());

    const std::unique_ptr<ptn::EngineExecutable> executable =
        context->compile(method_name);
    std::printf(
        "method:    %s (%zu inputs, %zu outputs)\n",
        method_name.c_str(),
        executable->num_inputs(),
        executable->num_outputs());
    std::printf(
        "constants: %zu unique source bytes, %zu materialized\n",
        context->unique_constant_bytes(),
        context->materialized_constant_bytes());

    if (executable->num_inputs() != 1 || executable->num_outputs() != 1) {
      std::fprintf(
          stderr,
          "error: expected 1 input and 1 output, got %zu and %zu\n",
          executable->num_inputs(),
          executable->num_outputs());
      return 2;
    }
    if (executable->input_dtype(0) != ptn::kFloat ||
        executable->output_dtype(0) != ptn::kFloat) {
      std::fprintf(stderr, "error: this runner handles fp32 models only\n");
      return 2;
    }

    const std::vector<int64_t> in_sizes = executable->input_sizes(0);
    const std::vector<int64_t> out_sizes = executable->output_sizes(0);
    std::printf(
        "  input:   %s\n  output:  %s\n",
        ptn::runner_utils::shape_str(in_sizes).c_str(),
        ptn::runner_utils::shape_str(out_sizes).c_str());

    const std::vector<float> input =
        ptn::runner_utils::read_floats(FLAGS_input);
    if (static_cast<int64_t>(input.size()) !=
        ptn::runner_utils::numel_of(in_sizes)) {
      std::fprintf(
          stderr,
          "error: input has %zu floats, the model wants %lld\n",
          input.size(),
          static_cast<long long>(ptn::runner_utils::numel_of(in_sizes)));
      return 2;
    }

    executable->set_input(0, input.data(), input.size(), ptn::kFloat);
    executable->execute();

    std::vector<float> logits(
        static_cast<size_t>(ptn::runner_utils::numel_of(out_sizes)));
    executable->get_output(0, logits.data(), logits.size(), ptn::kFloat);
    if (logits.empty()) {
      std::fprintf(stderr, "error: model produced no logits\n");
      return 2;
    }

    std::vector<std::string> labels;
    if (!FLAGS_categories.empty()) {
      labels = ptn::runner_utils::read_lines(FLAGS_categories);
    }
    print_top5(logits, labels);
    const int argmax = ptn::runner_utils::top_k(logits, 1).front();
    std::printf(
        "argmax:    %d (%s)\n", argmax, label_of(labels, argmax).c_str());

    if (FLAGS_expected.empty()) {
      std::printf("result:    ran (no reference given)\n");
      return 0;
    }
    const std::vector<float> reference =
        ptn::runner_utils::read_floats(FLAGS_expected);
    const bool pass = ptn::runner_utils::compare_floats(
        logits, reference, FLAGS_rtol, FLAGS_atol);
    std::printf("result:    %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
