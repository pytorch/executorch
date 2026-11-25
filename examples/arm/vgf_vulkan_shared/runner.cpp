/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/VGFSharedDevice.h>
#include <executorch/backends/vulkan/runtime/ResolveSharedDevice.h>
#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
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

// Check supported -> enabled propagation, not just numerical correctness.
void check_optional_features(const vulkan_shared::SharedVulkanDevice& device) {
  VkPhysicalDeviceVulkan13Features available_13{};
  available_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  VkPhysicalDeviceFeatures2 available{};
  available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  available.pNext = &available_13;
  vkGetPhysicalDeviceFeatures2(device.physical_device(), &available);
  const auto& enabled = device.enabled_device_features();
  if (enabled.shader_integer_dot_product !=
          (available_13.shaderIntegerDotProduct == VK_TRUE) ||
      enabled.subgroup_size_control !=
          (available_13.subgroupSizeControl == VK_TRUE) ||
      enabled.compute_full_subgroups !=
          (available_13.subgroupSizeControl == VK_TRUE &&
           available_13.computeFullSubgroups == VK_TRUE) ||
      enabled.shader_float64 != (available.features.shaderFloat64 == VK_TRUE)) {
    throw std::runtime_error(
        "Shared device lost supported optional Vulkan features");
  }
#if defined(VK_KHR_cooperative_matrix)
  uint32_t count = 0;
  if (vkEnumerateDeviceExtensionProperties(
          device.physical_device(), nullptr, &count, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Cannot enumerate device extensions");
  }
  std::vector<VkExtensionProperties> extensions(count);
  if (vkEnumerateDeviceExtensionProperties(
          device.physical_device(), nullptr, &count, extensions.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("Cannot read device extensions");
  }
  extensions.resize(count);
  const bool advertised = std::any_of(
      extensions.begin(), extensions.end(), [](const auto& extension) {
        return std::strcmp(
                   extension.extensionName,
                   VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
      });
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{};
  cooperative.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  if (advertised) {
    available.pNext = &cooperative;
    vkGetPhysicalDeviceFeatures2(device.physical_device(), &available);
  }
  if (enabled.cooperative_matrix !=
          (cooperative.cooperativeMatrix == VK_TRUE) ||
      (enabled.cooperative_matrix &&
       !device.has_device_extension(
           VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))) {
    throw std::runtime_error(
        "Shared device lost cooperative-matrix enablement");
  }
#endif
}

int execute_model(
    const std::string& directory,
    const executorch::runtime::LoadBackendOptionsMap* backend_options =
        nullptr) {
  auto input = read_floats(directory + "/input0.bin", 192);
  const auto expected = read_floats(directory + "/expected0.bin", 48);
  auto tensor = executorch::extension::make_tensor_ptr(
      {1, 3, 8, 8}, input.data(), executorch::aten::ScalarType::Float);
  executorch::extension::Module module(directory + "/vgf_vulkan_vgf.pte");
  if (backend_options != nullptr) {
    check(module.load(*backend_options), "Module::load");
  } else {
    // No registry lookup or sharing options: exercise the legacy combined
    // build.
    check(module.load(), "Module::load");
  }
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
  check_optional_features(*registration.device);

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

  return execute_model(directory, &backend_options);
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(
        stderr,
        "Usage: %s EXPORTED_DIRECTORY [--legacy|--legacy-vulkan-first|--shared-vulkan-first]\n",
        argv[0]);
    return 2;
  }
  const std::string mode = argc == 3 ? argv[2] : "--shared";
  const bool legacy = mode == "--legacy" || mode == "--legacy-vulkan-first";
  const bool vulkan_first =
      mode == "--legacy-vulkan-first" || mode == "--shared-vulkan-first";
  if (mode != "--shared" && !legacy && mode != "--shared-vulkan-first") {
    std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
    return 2;
  }
  executorch::runtime::runtime_init();
  try {
    if (vulkan_first) {
      // Before any VGF probe/pre-registration: reproduces the old 1.1 latch.
      (void)vkcompute::vkapi::runtime()->get_adapter_p();
    }
    // Each invocation repeats load/execute/destroy, including registry cleanup.
    for (int cycle = 0; cycle != 2; ++cycle) {
      const int result = legacy ? execute_model(argv[1]) : run(argv[1]);
      if (result != 0) {
        return result;
      }
      std::printf("PASS teardown cycle=%d mode=%s\n", cycle, mode.c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
