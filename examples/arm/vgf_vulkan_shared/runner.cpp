/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/VGFSharedDevice.h>
#include <executorch/backends/vulkan/runtime/ResolveSharedDevice.h>
#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>
#include <executorch/backends/vulkan_shared/runtime/SharedVulkanDeviceRegistry.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using executorch::runtime::Error;
using executorch::runtime::EValue;
using namespace executorch::backends;

void check(Error error, const char* operation) {
  if (error != Error::Ok) {
    throw std::runtime_error(
        std::string(operation) + " failed with error " +
        std::to_string(static_cast<unsigned>(error)));
  }
}

std::vector<float> read_floats(const std::string& path, size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamoff>(count * sizeof(float));
  if (!stream || stream.tellg() != bytes) {
    throw std::runtime_error("Wrong file size or unreadable file: " + path);
  }
  stream.seekg(0);
  std::vector<float> result(count);
  std::vector<char> input_bytes(static_cast<size_t>(bytes));
  if (!stream.read(input_bytes.data(), bytes)) {
    throw std::runtime_error("Cannot read " + path);
  }
  std::memcpy(result.data(), input_bytes.data(), input_bytes.size());
  return result;
}

// Declared before Module so unregister happens after Module and Adapter
// teardown, including on an exception. Do not clear the process-wide registry
// indiscriminately.
struct Registration final {
  vulkan_shared::SharedVulkanDeviceKey key;
  vulkan_shared::SharedVulkanDevicePtr device;
  ~Registration() {
    auto& registry = vulkan_shared::SharedVulkanDeviceRegistry::Get();
    if (registry.lookup(key) == device) {
      (void)registry.unregister_device(key);
    }
  }
};

int run(const std::string& directory) {
  vulkan_shared::SharedVulkanRuntimeConfig config;
  config.device_name = "vgf-vulkan-e2e";
  config.group_id = 7;
  config.device_mode = vulkan_shared::SharedDeviceMode::kCreateOnly;
  auto prepared = vgf::resolve_vgf_shared_device(config);
  if (!prepared.ok()) {
    check(prepared.error(), "VGF-capable shared device creation");
  }
  Registration registration{
      {config.device_name, config.group_id}, prepared.get()};

  executorch::runtime::BackendOptions<3> options;
  check(
      options.set_option("vulkan_shared_device_name", "vgf-vulkan-e2e"),
      "name");
  check(options.set_option("vulkan_shared_device_mode", "lookup_only"), "mode");
  check(options.set_option("vulkan_shared_group_id", 7), "group");
  executorch::runtime::LoadBackendOptionsMap backend_options;
  check(
      backend_options.set_options("VgfBackend", options.view()), "VGF options");
  check(
      backend_options.set_options("VulkanBackend", options.view()),
      "Vulkan options");

  // Exercise actual Adapter resolver and keep that Adapter alive.
  // The Vulkan delegate subsequently reuses this same cached Adapter.
  const auto view = options.view();
  executorch::runtime::BackendInitContext context(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      executorch::runtime::Span<const executorch::runtime::BackendOption>(
          view.data(), view.size()));
  auto adapter = vulkan::resolve_vulkan_shared_adapter(context);
  if (!adapter.ok()) {
    check(adapter.error(), "Vulkan shared Adapter resolution");
  }
  if (!adapter.get() || adapter.get()->shared_device() != registration.device ||
      adapter.get()->device_handle() != registration.device->device()) {
    throw std::runtime_error(
        "Vulkan Adapter did not reuse the VGF-created device");
  }
  std::printf(
      "Shared VkDevice verified: VGF=%p Vulkan=%p\n",
      static_cast<void*>(registration.device->device()),
      static_cast<void*>(adapter.get()->device_handle()));

  auto input = read_floats(directory + "/input0.bin", 192);
  const auto expected = read_floats(directory + "/expected0.bin", 48);
  auto tensor = executorch::extension::make_tensor_ptr(
      {1, 3, 8, 8}, input.data(), executorch::aten::ScalarType::Float);
  executorch::extension::Module module(directory + "/vgf_vulkan_vgf.pte");
  check(module.load(backend_options), "Module::load");
  check(module.load_method("forward"), "Module::load_method");

  // Re-execution checks pool/fence and representation reuse, not performance.
  for (int iteration = 0; iteration != 3; ++iteration) {
    auto outputs = module.forward(std::vector<EValue>{EValue(*tensor)});
    if (!outputs.ok()) {
      check(outputs.error(), "Module::forward");
    }
    if (outputs->size() != 1 || !outputs->at(0).isTensor()) {
      throw std::runtime_error("Expected one output tensor");
    }
    const auto& output = outputs->at(0).toTensor();
    if (output.scalar_type() != executorch::aten::ScalarType::Float ||
        output.dim() != 4 || output.size(0) != 1 || output.size(1) != 3 ||
        output.size(2) != 4 || output.size(3) != 4 ||
        static_cast<size_t>(output.numel()) != expected.size()) {
      throw std::runtime_error("Output shape or dtype mismatch");
    }
    const float* values = output.const_data_ptr<float>();
    float max_abs_error = 0;
    for (size_t i = 0; i != expected.size(); ++i) {
      const float error = std::abs(values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);
      if (!std::isfinite(values[i]) || !std::isfinite(expected[i]) ||
          error > 1e-5f + 1e-4f * std::abs(expected[i])) {
        throw std::runtime_error(
            "Numerical mismatch at output element " + std::to_string(i));
      }
    }
    std::ofstream saved(directory + "/actual0.bin", std::ios::binary);
    std::vector<char> output_bytes(output.nbytes());
    std::memcpy(output_bytes.data(), values, output_bytes.size());
    if (!saved.write(
            output_bytes.data(),
            static_cast<std::streamsize>(output_bytes.size()))) {
      throw std::runtime_error("Cannot write actual0.bin");
    }
    std::printf(
        "PASS iteration=%d max_abs_error=%.9g\n", iteration, max_abs_error);
  }
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s EXPORTED_DIRECTORY\n", argv[0]);
    return 2;
  }
  executorch::runtime::runtime_init();
  try {
    return run(argv[1]);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
