/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Runs one XTS3 graph on Int8Interpreter directly, for
 * backends/espressif/test/test_host_interpreter.py:
 *
 *   espnn_host_interpreter GRAPH OFFSET [--constants-to-psram] [--external-io]
 *                          INPUT... -- OUTPUT...
 *
 * The graph is placed OFFSET bytes past a 16-byte boundary, so that a test can
 * load it from addresses with any alignment. Inputs and outputs are raw
 * device-layout (NHWC) int8 files. --constants-to-psram sets
 * Int8RuntimeOptions::copy_constants_to_psram. When the interpreter copies the
 * graph, the runner frees its own copy, as EspNNBackend does. With
 * --external-io, outputs are bound to 16-byte aligned runner buffers prefilled
 * with 0x5a, after checking that a misaligned buffer is refused.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <espnn/int8_interpreter.h>

using executorch::backends::espressif::Int8Interpreter;
using executorch::backends::espressif::Int8RuntimeOptions;
using executorch::backends::espressif::Int8TensorInfo;

namespace {

std::vector<uint8_t> read_file(const char* path) {
  std::vector<uint8_t> data;
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr)
    return data;
  uint8_t chunk[4096];
  size_t read = 0;
  while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    data.insert(data.end(), chunk, chunk + read);
  std::fclose(file);
  return data;
}

int fail(const std::string& what) {
  std::fprintf(stderr, "espnn_host_interpreter: %s\n", what.c_str());
  return 1;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4)
    return fail("usage: GRAPH OFFSET INPUT... -- OUTPUT...");
  const std::vector<uint8_t> graph = read_file(argv[1]);
  const size_t offset = std::strtoul(argv[2], nullptr, 10);
  std::vector<const char*> inputs;
  std::vector<const char*> outputs;
  bool past_separator = false;
  Int8RuntimeOptions options{};
  for (int a = 3; a < argc; ++a) {
    if (std::strcmp(argv[a], "--") == 0)
      past_separator = true;
    else if (std::strcmp(argv[a], "--constants-to-psram") == 0)
      options.copy_constants_to_psram = true;
    else if (std::strcmp(argv[a], "--external-io") == 0)
      options.external_io = true;
    else
      (past_separator ? outputs : inputs).push_back(argv[a]);
  }

  // A 16-byte aligned buffer, holding the graph at the requested offset.
  std::vector<uint64_t> storage((graph.size() + offset) / 8 + 3);
  uint8_t* base = reinterpret_cast<uint8_t*>(storage.data());
  base += (16 - reinterpret_cast<uintptr_t>(base) % 16) % 16;
  std::memcpy(base + offset, graph.data(), graph.size());

  Int8Interpreter interpreter;
  if (!interpreter.load(base + offset, graph.size(), options))
    return fail("load failed");
  if (interpreter.constants_copied()) {
    storage.clear();
    storage.shrink_to_fit();
  }
  if (inputs.size() != interpreter.input_count() ||
      outputs.size() != interpreter.output_count())
    return fail("wrong number of inputs or outputs");
  std::vector<std::vector<uint64_t>> bound(outputs.size());
  for (size_t o = 0; options.external_io && o < bound.size(); ++o) {
    Int8TensorInfo info{};
    interpreter.output_info(o, info);
    bound[o].resize(info.bytes / 8 + 3);
    int8_t* buffer = reinterpret_cast<int8_t*>(bound[o].data());
    buffer += (16 - reinterpret_cast<uintptr_t>(buffer) % 16) % 16;
    std::memset(buffer, 0x5a, info.bytes);
    if (interpreter.bind_output(o, buffer + 1) ||
        !interpreter.bind_output(o, buffer))
      return fail("bound a misaligned output or refused an aligned one");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    Int8TensorInfo info{};
    interpreter.input_info(i, info);
    const std::vector<uint8_t> data = read_file(inputs[i]);
    if (data.size() != info.bytes)
      return fail(std::string("wrong input size: ") + inputs[i]);
    std::memcpy(interpreter.input_data(i), data.data(), data.size());
  }
  if (!interpreter.invoke())
    return fail("invoke failed");
  for (size_t o = 0; o < outputs.size(); ++o) {
    Int8TensorInfo info{};
    interpreter.output_info(o, info);
    std::FILE* file = std::fopen(outputs[o], "wb");
    if (file == nullptr ||
        std::fwrite(interpreter.output_data(o), 1, info.bytes, file) !=
            info.bytes)
      return fail(std::string("cannot write ") + outputs[o]);
    std::fclose(file);
  }
#ifdef ESPNN_HOST_PSRAM
  const int psram = 1;
#else
  const int psram = 0;
#endif
  std::printf(
      "STATS arena=%zu arena_internal=%d parameters=%zu blob_copied=%d "
      "psram=%d\n",
      interpreter.arena_bytes(),
      interpreter.arena_in_internal_ram() ? 1 : 0,
      interpreter.parameter_bytes(),
      interpreter.constants_copied() ? 1 : 0,
      psram);
  return 0;
}
