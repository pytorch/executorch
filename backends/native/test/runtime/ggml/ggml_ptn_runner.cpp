// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Runs one method of a .ptn on the ggml CPU engine. Inputs and outputs are raw
// little-endian bytes in the method's own dtypes and serialized shapes. Prints
// one line of space-separated sizes per output.
//
// Usage:
//   ggml_ptn_runner --package M.ptn --inputs a.bin,b.bin --outputs out0.bin
//
// Exit: 0 success, 1 usage, 2 error.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <gflags/gflags.h>

DEFINE_string(package, "", "Path to the *.ptn to run. Required.");
DEFINE_string(method, "forward", "Method to run.");
DEFINE_string(inputs, "", "Comma-separated raw input files, in graph order.");
DEFINE_string(outputs, "", "Comma-separated raw output files, in graph order.");

namespace {

std::vector<std::string> split(const std::string& list) {
  std::vector<std::string> out;
  std::stringstream stream(list);
  std::string item;
  while (std::getline(stream, item, ',')) {
    out.push_back(item);
  }
  return out;
}

size_t numel_of(const std::vector<int64_t>& sizes) {
  return std::accumulate(
      sizes.begin(), sizes.end(), size_t{1}, [](size_t numel, int64_t size) {
        return numel * static_cast<size_t>(size);
      });
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

void run() {
  const auto package =
      std::make_shared<const ptn::Package>(ptn::Package::load(FLAGS_package));
  const ptn::ByteSpan program_bytes = package->program_bytes();
  const auto program = std::make_shared<const ptn::Program>(
      ptn::Program::load(program_bytes.data(), program_bytes.size()));
  const auto host = ptn::GgmlEngineHost::create();
  const auto context = host->create_ggml_context(program, package);
  const auto executable = context->compile(FLAGS_method);

  const std::vector<std::string> inputs = split(FLAGS_inputs);
  const std::vector<std::string> outputs = split(FLAGS_outputs);
  if (inputs.size() != executable->num_inputs() ||
      outputs.size() != executable->num_outputs()) {
    throw std::runtime_error(
        "method has " + std::to_string(executable->num_inputs()) +
        " inputs and " + std::to_string(executable->num_outputs()) +
        " outputs, got " + std::to_string(inputs.size()) + " and " +
        std::to_string(outputs.size()));
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    const ptn::ScalarType dtype = executable->input_dtype(i);
    const size_t numel = numel_of(executable->input_sizes(i));
    const std::vector<char> bytes = read_file(inputs[i]);
    const size_t expected_bytes = numel * ptn::element_size(dtype);
    if (bytes.size() != expected_bytes) {
      throw std::runtime_error(
          "input " + std::to_string(i) + " has " +
          std::to_string(bytes.size()) + " bytes, expected " +
          std::to_string(expected_bytes));
    }
    executable->set_input(i, bytes.data(), numel, dtype);
  }

  executable->execute();

  for (size_t i = 0; i < outputs.size(); ++i) {
    const ptn::ScalarType dtype = executable->output_dtype(i);
    const std::vector<int64_t> sizes = executable->output_sizes(i);
    const size_t numel = numel_of(sizes);
    std::vector<char> bytes(numel * ptn::element_size(dtype));
    executable->get_output(i, bytes.data(), numel, dtype);
    write_file(outputs[i], bytes);
    std::string line;
    for (const int64_t size : sizes) {
      line += (line.empty() ? "" : " ") + std::to_string(size);
    }
    std::printf("%s\n", line.c_str());
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_package.empty()) {
    std::fprintf(stderr, "error: --package is required\n");
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
