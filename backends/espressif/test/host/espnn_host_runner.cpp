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
 * Runs an ExecuTorch program on the host build of the ESP-NN delegate, for
 * backends/espressif/test/test_host_runtime.py.
 *
 *   espnn_host_runner PROGRAM IO_DIR STEPS [--instances N] [--share-io]
 *                     [--alias INPUT:OUTPUT]... [--misalign]
 *                     [--reset-before STEP]...
 *
 * Step s of instance k reads input i from IO_DIR/in_<s>_<i>.bin and writes
 * output o to IO_DIR/out_<k>_<s>_<o>.bin: the raw bytes of the tensor in its
 * dim order. With --instances, the method is loaded N times and each step runs
 * on every instance in turn. With --share-io (the program's inputs and
 * outputs must not be memory planned), the tensors are bound once to runner
 * buffers through set_input and set_output_data_ptr, and --alias makes an
 * input and an output share one buffer: that input is then read from its file
 * at step 0 only and afterwards holds the output of the previous step.
 * --misalign places every runner buffer one byte past a 16-byte boundary.
 * --reset-before asks EspNNBackend, through its reset_states backend option,
 * to restore its delegates' state before that step. At the end, the runner
 * prints the delegates' memory as a STATS line.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <espnn/espnn_backend.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::ScalarType;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

namespace {

[[noreturn]] void fail(const char* what, int code = 0) {
  std::fprintf(stderr, "espnn_host_runner: %s (%d)\n", what, code);
  std::exit(1);
}

std::vector<uint8_t> read_file(const std::string& path, size_t expected) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr)
    fail(("cannot open " + path).c_str());
  std::vector<uint8_t> data(expected + 1);
  const size_t read = std::fread(data.data(), 1, data.size(), file);
  std::fclose(file);
  if (read != expected)
    fail(("wrong size: " + path).c_str(), static_cast<int>(read));
  data.resize(expected);
  return data;
}

void write_file(const std::string& path, const void* data, size_t size) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr || std::fwrite(data, 1, size, file) != size)
    fail(("cannot write " + path).c_str());
  std::fclose(file);
}

uint8_t* aligned_buffer(uint8_t* base, size_t misalign) {
  return base + (16 - reinterpret_cast<uintptr_t>(base) % 16) % 16 + misalign;
}

// A tensor over a runner-owned buffer with the program's shape and dim order.
struct HostTensor {
  HostTensor(const TensorInfo& info, size_t misalign)
      : sizes(info.sizes().begin(), info.sizes().end()),
        dim_order(info.dim_order().begin(), info.dim_order().end()),
        strides(sizes.size()),
        storage(info.nbytes() + 32),
        data(aligned_buffer(storage.data(), misalign)),
        nbytes(info.nbytes()),
        impl(
            info.scalar_type(),
            static_cast<ssize_t>(sizes.size()),
            sizes.data(),
            data,
            dim_order.data(),
            strides.data()) {
    if (executorch::runtime::dim_order_to_stride(
            sizes.data(), dim_order.data(), sizes.size(), strides.data()) !=
        Error::Ok)
      fail("invalid dim order");
  }
  std::vector<TensorImpl::SizesType> sizes;
  std::vector<TensorImpl::DimOrderType> dim_order;
  std::vector<TensorImpl::StridesType> strides;
  std::vector<uint8_t> storage;
  uint8_t* data;
  size_t nbytes;
  TensorImpl impl;
};

struct Instance {
  std::vector<std::vector<uint8_t>> planned;
  std::vector<Span<uint8_t>> spans;
  std::unique_ptr<HierarchicalAllocator> planned_allocator;
  std::vector<uint8_t> method_pool;
  std::unique_ptr<MemoryAllocator> method_allocator;
  std::unique_ptr<MemoryManager> memory;
  std::unique_ptr<Method> method;
  std::vector<std::unique_ptr<HostTensor>> inputs;
  std::vector<std::unique_ptr<HostTensor>> outputs;
};

} // namespace

int main(int argc, char** argv) {
  if (argc < 4)
    fail("usage: espnn_host_runner PROGRAM IO_DIR STEPS [options]");
  const std::string io_dir = argv[2];
  const int steps = std::atoi(argv[3]);
  int instance_count = 1;
  bool share_io = false;
  size_t misalign = 0;
  std::vector<std::pair<size_t, size_t>> aliases;
  std::vector<int> resets;
  for (int a = 4; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "--instances" && a + 1 < argc) {
      instance_count = std::atoi(argv[++a]);
    } else if (arg == "--reset-before" && a + 1 < argc) {
      resets.push_back(std::atoi(argv[++a]));
    } else if (arg == "--share-io") {
      share_io = true;
    } else if (arg == "--misalign") {
      misalign = 1;
    } else if (arg == "--alias" && a + 1 < argc) {
      unsigned input = 0, output = 0;
      if (std::sscanf(argv[++a], "%u:%u", &input, &output) != 2)
        fail("--alias takes INPUT:OUTPUT");
      aliases.emplace_back(input, output);
    } else {
      fail(("unknown argument " + arg).c_str());
    }
  }
  if (!aliases.empty() && !share_io)
    fail("--alias needs --share-io");

  executorch::runtime::runtime_init();
  Result<FileDataLoader> loader = FileDataLoader::from(argv[1]);
  if (!loader.ok())
    fail("cannot load program", static_cast<int>(loader.error()));
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok())
    fail("invalid program", static_cast<int>(program.error()));
  Result<MethodMeta> meta = program->method_meta("forward");
  if (!meta.ok())
    fail("no forward method", static_cast<int>(meta.error()));

  std::vector<Instance> instances(instance_count);
  for (Instance& instance : instances) {
    for (size_t b = 0; b < meta->num_memory_planned_buffers(); ++b) {
      instance.planned.emplace_back(meta->memory_planned_buffer_size(b).get());
    }
    for (auto& buffer : instance.planned)
      instance.spans.emplace_back(buffer.data(), buffer.size());
    instance.planned_allocator = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>(instance.spans.data(), instance.spans.size()));
    instance.method_pool.resize(1 << 20);
    instance.method_allocator = std::make_unique<MemoryAllocator>(
        instance.method_pool.size(), instance.method_pool.data());
    instance.memory = std::make_unique<MemoryManager>(
        instance.method_allocator.get(), instance.planned_allocator.get());
    Result<Method> method =
        program->load_method("forward", instance.memory.get());
    if (!method.ok())
      fail("cannot load method", static_cast<int>(method.error()));
    instance.method = std::make_unique<Method>(std::move(method.get()));

    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      instance.inputs.push_back(std::make_unique<HostTensor>(
          meta->input_tensor_meta(i).get(), misalign));
    }
    for (size_t o = 0; o < meta->num_outputs(); ++o) {
      instance.outputs.push_back(std::make_unique<HostTensor>(
          meta->output_tensor_meta(o).get(), misalign));
    }
    for (const auto& [input, output] : aliases) {
      if (input >= instance.inputs.size() ||
          output >= instance.outputs.size() ||
          instance.inputs[input]->nbytes != instance.outputs[output]->nbytes)
        fail("invalid --alias");
    }
    if (!share_io)
      continue;
    for (size_t i = 0; i < instance.inputs.size(); ++i) {
      Error error = instance.method->set_input(
          EValue(executorch::aten::Tensor(&instance.inputs[i]->impl)), i);
      if (error != Error::Ok)
        fail("set_input", static_cast<int>(error));
    }
    for (size_t o = 0; o < instance.outputs.size(); ++o) {
      void* buffer = instance.outputs[o]->data;
      for (const auto& [input, output] : aliases) {
        if (output == o)
          buffer = instance.inputs[input]->data;
      }
      Error error = instance.method->set_output_data_ptr(
          buffer, instance.outputs[o]->nbytes, o);
      if (error != Error::Ok)
        fail("set_output_data_ptr", static_cast<int>(error));
    }
  }

  for (int step = 0; step < steps; ++step) {
    for (int reset : resets) {
      if (reset != step)
        continue;
      executorch::runtime::BackendOptions<1> options;
      if (options.set_option(
              executorch::backends::espressif::kResetStatesOption, true) !=
          Error::Ok)
        fail("BackendOptions::set_option");
      Error error = executorch::runtime::set_option(
          executorch::backends::espressif::kEspNNBackendId, options.view());
      if (error != Error::Ok)
        fail("set_option", static_cast<int>(error));
    }
    for (int k = 0; k < instance_count; ++k) {
      Instance& instance = instances[k];
      for (size_t i = 0; i < instance.inputs.size(); ++i) {
        bool carried = false;
        for (const auto& alias : aliases)
          carried = carried || (alias.first == i && step > 0);
        if (carried)
          continue;
        HostTensor& tensor = *instance.inputs[i];
        const std::vector<uint8_t> bytes = read_file(
            io_dir + "/in_" + std::to_string(step) + "_" + std::to_string(i) +
                ".bin",
            tensor.nbytes);
        std::memcpy(tensor.data, bytes.data(), bytes.size());
        if (!share_io) {
          Error error = instance.method->set_input(
              EValue(executorch::aten::Tensor(&tensor.impl)), i);
          if (error != Error::Ok)
            fail("set_input", static_cast<int>(error));
        }
      }
      Error error = instance.method->execute();
      if (error != Error::Ok)
        fail("execute", static_cast<int>(error));
      for (size_t o = 0; o < instance.outputs.size(); ++o) {
        const executorch::aten::Tensor output =
            instance.method->get_output(o).toTensor();
        write_file(
            io_dir + "/out_" + std::to_string(k) + "_" + std::to_string(step) +
                "_" + std::to_string(o) + ".bin",
            output.const_data_ptr(),
            output.nbytes());
      }
    }
  }
  const auto stats = executorch::backends::espressif::delegate_stats();
  std::printf(
      "STATS delegates=%u arena=%zu scratch=%zu parameters=%zu tables=%zu "
      "staging=%zu states=%u states_in_place=%u\n",
      static_cast<unsigned>(stats.delegates),
      stats.arena_bytes,
      stats.scratch_bytes,
      stats.parameter_bytes,
      stats.table_bytes,
      stats.staging_bytes,
      static_cast<unsigned>(stats.states),
      static_cast<unsigned>(stats.states_updated_in_place));

  return 0;
}
