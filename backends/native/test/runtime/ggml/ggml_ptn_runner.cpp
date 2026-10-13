// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Runs one method of a .ptn on the ggml engine. Inputs and outputs are raw
// little-endian bytes in the method's own dtypes. Semicolons separate steps,
// which run in order on one executable, so mutable buffers such as KV caches
// carry over. Prints one line of space-separated sizes per output per step.
//
// Usage:
//   ggml_ptn_runner --package M.ptn --inputs a.bin,b.bin --outputs out0.bin
//                   [--input_sizes 1x6,6] [--device gpu]
//                   [--warmup 3 --iterations 20]
//   ggml_ptn_runner --package M.ptn --inputs a0.bin,b0.bin;a1.bin,b1.bin
//                   --outputs out0.bin;out1.bin --input_sizes 1x6,6;1x1,1
//
// Exit: 0 success, 1 usage, 2 error.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <gflags/gflags.h>

DEFINE_string(package, "", "Path to the *.ptn to run. Required.");
DEFINE_string(method, "forward", "Method to run.");
DEFINE_string(
    inputs,
    "",
    "Comma-separated raw input files, in graph order; semicolons separate "
    "steps.");
DEFINE_string(
    outputs,
    "",
    "Comma-separated raw output files, in graph order; semicolons separate "
    "steps.");
DEFINE_string(
    input_sizes,
    "",
    "Comma-separated input shapes such as 1x6, in graph order, for methods "
    "with dynamic shapes; semicolons separate steps. Defaults to the "
    "serialized shapes.");
DEFINE_string(
    device,
    "cpu",
    "ggml device: cpu, or gpu for the GPU backend this binary links.");
DEFINE_int32(cpu_threads, 0, "ggml CPU threads; 0 keeps ggml's default.");
DEFINE_int32(warmup, 0, "Untimed executions before the timed ones.");
DEFINE_int32(
    iterations,
    1,
    "Timed executions; above one, prints median and min latency.");

namespace {

std::vector<std::string> split(const std::string& list, char delimiter) {
  std::vector<std::string> out;
  std::stringstream stream(list);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    out.push_back(item);
  }
  return out;
}

size_t numel_of(const std::vector<int64_t>& sizes) {
  size_t numel = 1;
  for (const int64_t size : sizes) {
    numel *= static_cast<size_t>(size);
  }
  return numel;
}

std::vector<int64_t> parse_sizes(const std::string& shape) {
  std::vector<int64_t> sizes;
  std::stringstream stream(shape);
  std::string dim;
  while (std::getline(stream, dim, 'x')) {
    int64_t size = -1;
    const char* end = dim.data() + dim.size();
    const auto [parsed, error] = std::from_chars(dim.data(), end, size);
    if (error != std::errc() || parsed != end || size < 0) {
      throw std::invalid_argument("invalid --input_sizes entry: " + shape);
    }
    sizes.push_back(size);
  }
  return sizes;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  return {std::istreambuf_iterator<char>(file), {}};
}

void write_file(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file) {
    throw std::runtime_error("cannot write " + path);
  }
}

void run_step(
    ptn::EngineExecutable& executable,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& input_sizes) {
  if (inputs.size() != executable.num_inputs() ||
      outputs.size() != executable.num_outputs()) {
    throw std::runtime_error(
        "method has " + std::to_string(executable.num_inputs()) +
        " inputs and " + std::to_string(executable.num_outputs()) +
        " outputs, got " + std::to_string(inputs.size()) + " and " +
        std::to_string(outputs.size()));
  }
  if (!input_sizes.empty() && input_sizes.size() != inputs.size()) {
    throw std::runtime_error("--input_sizes must list every input");
  }
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    executable.resize_input(i, parse_sizes(input_sizes[i]));
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    const ptn::ScalarType dtype = executable.input_dtype(i);
    const size_t numel = numel_of(executable.input_sizes(i));
    const std::vector<char> bytes = read_file(inputs[i]);
    const size_t expected_bytes = numel * ptn::element_size(dtype);
    if (bytes.size() != expected_bytes) {
      throw std::runtime_error(
          "input " + std::to_string(i) + " has " +
          std::to_string(bytes.size()) + " bytes, expected " +
          std::to_string(expected_bytes));
    }
    executable.set_input(i, bytes.data(), numel, dtype);
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    executable.execute();
  }
  std::vector<double> latencies_ms;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    executable.execute();
    latencies_ms.push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count());
  }
  if (latencies_ms.size() > 1) {
    std::ranges::sort(latencies_ms);
    std::printf(
        "latency_ms: median %.3f min %.3f (%zu iterations)\n",
        latencies_ms[latencies_ms.size() / 2],
        latencies_ms.front(),
        latencies_ms.size());
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    const ptn::ScalarType dtype = executable.output_dtype(i);
    const std::vector<int64_t> sizes = executable.output_sizes(i);
    const size_t numel = numel_of(sizes);
    std::vector<char> bytes(numel * ptn::element_size(dtype));
    executable.get_output(i, bytes.data(), numel, dtype);
    write_file(outputs[i], bytes);
    std::string line;
    for (const int64_t size : sizes) {
      line += (line.empty() ? "" : " ") + std::to_string(size);
    }
    std::printf("%s\n", line.c_str());
  }
}

void run() {
  const auto package =
      std::make_shared<const ptn::Package>(ptn::Package::load(FLAGS_package));
  const ptn::ByteSpan program_bytes = package->program_bytes();
  const auto program = std::make_shared<const ptn::Program>(
      ptn::Program::load(program_bytes.data(), program_bytes.size()));
  ptn::GgmlEngineOptions options;
  if (FLAGS_device == "gpu") {
    options.device = ptn::GgmlDevice::Gpu;
  } else if (FLAGS_device != "cpu") {
    throw std::invalid_argument("--device must be cpu or gpu");
  }
  options.cpu_threads = FLAGS_cpu_threads;
  const auto host = ptn::GgmlEngineHost::create(options);
  const auto context = host->create_ggml_context(program, package);
  const auto executable = context->compile(FLAGS_method);

  const std::vector<std::string> input_steps = split(FLAGS_inputs, ';');
  const std::vector<std::string> output_steps = split(FLAGS_outputs, ';');
  const std::vector<std::string> size_steps = split(FLAGS_input_sizes, ';');
  if (output_steps.size() != input_steps.size() ||
      (!size_steps.empty() && size_steps.size() != input_steps.size())) {
    throw std::runtime_error(
        "got " + std::to_string(input_steps.size()) + " input steps, " +
        std::to_string(output_steps.size()) + " output steps and " +
        std::to_string(size_steps.size()) + " size steps");
  }
  // Repeated executions advance mutable buffers such as KV caches.
  if (input_steps.size() > 1 && (FLAGS_warmup > 0 || FLAGS_iterations > 1)) {
    throw std::invalid_argument("--warmup and --iterations need one step");
  }
  for (size_t step = 0; step < input_steps.size(); ++step) {
    run_step(
        *executable,
        split(input_steps[step], ','),
        split(output_steps[step], ','),
        size_steps.empty() ? std::vector<std::string>{}
                           : split(size_steps[step], ','));
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_package.empty()) {
    std::fprintf(stderr, "error: --package is required\n");
    return 1;
  }
  if (FLAGS_warmup < 0 || FLAGS_iterations < 1) {
    std::fprintf(stderr, "error: --warmup must be >= 0, --iterations >= 1\n");
    return 1;
  }
  try {
    run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  return 0;
}
