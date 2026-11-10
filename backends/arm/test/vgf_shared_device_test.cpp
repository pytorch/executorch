/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * Licensed under the BSD-style license in the root LICENSE file.
 */
#include <executorch/backends/arm/runtime/VGFSharedDevice.h>
#include <executorch/runtime/backend/options.h>
#include <gtest/gtest.h>

#include <memory>

// Cppcheck may not expand GoogleTest headers. In that analysis-only case,
// supply plain C++ definitions so the test bodies remain parseable. A normal
// compiler sees GoogleTest's TEST macro and does not use these definitions.
#ifndef TEST
#define TEST(suite, name)                                                     \
  void cppcheck_##suite##_##name();                                           \
  [[maybe_unused]] static void (*const                                        \
                                    cppcheck_test_ref_##suite##_##name[])() = \
      {&cppcheck_##suite##_##name};                                           \
  void cppcheck_##suite##_##name()
#define ASSERT_TRUE(expr) ((void)(expr))
#define EXPECT_TRUE(expr) ((void)(expr))
#define EXPECT_FALSE(expr) ((void)(expr))
#define ASSERT_EQ(lhs, rhs) ((void)((lhs) == (rhs)))
#define EXPECT_EQ(lhs, rhs) ((void)((lhs) == (rhs)))
#endif

namespace {
using namespace executorch::backends::vgf;
using namespace executorch::backends::vulkan_shared;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::Span;

template <size_t N>
BackendInitContext context(BackendOptions<N>& options) {
  const auto view = options.view();
  return BackendInitContext(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      Span<const BackendOption>(view.data(), view.size()));
}

TEST(VgfSharedDevice, NoOptionsPreserveLegacy) {
  BackendInitContext init(nullptr);
  auto config = parse_vgf_shared_device_config(init);
  ASSERT_TRUE(config.ok());
  EXPECT_FALSE(config->enabled());
  auto state = resolve_vgf_shared_device_state(init);
  ASSERT_TRUE(state.ok());
  EXPECT_EQ(state.get(), nullptr);
}

TEST(VgfSharedDevice, InvalidOptionTypeIsRejected) {
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option("vulkan_shared_device_name", 7), Error::Ok);
  EXPECT_EQ(
      parse_vgf_shared_device_config(context(options)).error(),
      Error::InvalidArgument);
}

TEST(VgfSharedDevice, RuntimeNameAndGroupOptIn) {
  BackendOptions<2> options;
  ASSERT_EQ(
      options.set_option("vulkan_shared_device_name", "vgf-test"), Error::Ok);
  ASSERT_EQ(options.set_option("vulkan_shared_group_id", 9), Error::Ok);
  auto config = parse_vgf_shared_device_config(context(options));
  ASSERT_TRUE(config.ok());
  EXPECT_TRUE(config->lookup_or_create());
  EXPECT_EQ(config->device_name, "vgf-test");
  EXPECT_EQ(config->group_id, 9);
}

TEST(VgfSharedDevice, ExplicitDisableDoesNotCreateDevice) {
  BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option("vulkan_shared_device_mode", "disabled"), Error::Ok);
  auto state = resolve_vgf_shared_device_state(context(options));
  ASSERT_TRUE(state.ok());
  EXPECT_EQ(state.get(), nullptr);
}

TEST(VgfSharedDevice, MissingEnabledMetadataIsNotSupportedFeatures) {
  SharedVulkanDeviceFeatures features;
  EXPECT_FALSE(vgf_shared_features_compatible(features));
  features.arm_tensors = true;
  features.arm_shader_tensor_access = true;
  features.arm_data_graph = true;
  features.arm_data_graph_shader_module = true;
  features.shader_int16 = true;
  features.shader_int64 = true;
  features.shader_int8 = true;
  features.storage_buffer_8bit_access = true;
  features.host_query_reset = true;
  features.synchronization2 = true;
  features.maintenance4 = true;
  features.pipeline_creation_cache_control = true;
  features.vulkan_memory_model = true;
  features.shader_replicated_composites = true;
  EXPECT_TRUE(vgf_shared_features_compatible(features));
  // VGF_STAGE4_OPTIONAL_REPLICATED_COMPOSITES_V1
  auto without_replicated_composites = features;
  without_replicated_composites.shader_replicated_composites = false;
  EXPECT_TRUE(vgf_shared_features_compatible(without_replicated_composites));
  auto without_graph_module = features;
  without_graph_module.arm_data_graph_shader_module = false;
  EXPECT_FALSE(vgf_shared_features_compatible(without_graph_module));
  auto without_sync = features;
  without_sync.synchronization2 = false;
  EXPECT_FALSE(vgf_shared_features_compatible(without_sync));
}

TEST(VgfSharedDevice, MissingExtensionsAreRejectedByMetadataPolicy) {
  SharedVulkanDeviceCreateInfo info;
  info.enabled_device_extensions = {"VK_ARM_tensors", "VK_ARM_data_graph"};
  SharedVulkanDevice partial(std::move(info));
  EXPECT_TRUE(vgf_missing_required_device_extension(partial) != nullptr);
  SharedVulkanDeviceCreateInfo complete_info;
  complete_info.enabled_device_extensions = {
      "VK_ARM_tensors",
      "VK_ARM_data_graph",
      "VK_KHR_maintenance5",
      "VK_KHR_deferred_host_operations"};
  SharedVulkanDevice complete(std::move(complete_info));
  EXPECT_TRUE(vgf_missing_required_device_extension(complete) == nullptr);
}

TEST(VgfSharedDevice, SharedQueueRequiresComputeAndDataGraph) {
  EXPECT_FALSE(vgf_shared_queue_compatible(VK_QUEUE_COMPUTE_BIT));
  EXPECT_FALSE(vgf_shared_queue_compatible(VK_QUEUE_DATA_GRAPH_BIT_ARM));
  EXPECT_TRUE(vgf_shared_queue_compatible(
      VK_QUEUE_COMPUTE_BIT | VK_QUEUE_DATA_GRAPH_BIT_ARM));
}

TEST(VgfSharedDevice, StateReleasesItsDeviceReferenceOnTeardown) {
  SharedVulkanDeviceCreateInfo info;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> owner_weak = owner;
  info.lifetime_anchor = owner;
  auto device = std::make_shared<SharedVulkanDevice>(std::move(info));
  owner.reset();
  {
    auto state = std::make_shared<VgfSharedDeviceState>();
    state->device = device;
    EXPECT_FALSE(owner_weak.expired());
  }
  EXPECT_FALSE(owner_weak.expired());
  device.reset();
  EXPECT_TRUE(owner_weak.expired());
}

#if defined(EXECUTORCH_VGF_SHARED_DEVICE) && EXECUTORCH_VGF_SHARED_DEVICE
TEST(VgfSharedDevice, ImplicitCreationIsRejectedAtDelegateInit) {
  BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option("vulkan_shared_device_name", "vgf-order-test"),
      Error::Ok);
  EXPECT_EQ(
      resolve_vgf_shared_device_state(context(options)).error(),
      Error::InvalidArgument);
}

TEST(VgfSharedDevice, LookupOnlyMissingDoesNotInitializeVulkan) {
  SharedVulkanRuntimeConfig config;
  config.device_name = "stage4-unit-test-no-such-device";
  config.device_mode = SharedDeviceMode::kLookupOnly;
  EXPECT_EQ(resolve_vgf_shared_device(config).error(), Error::NotFound);
}
#endif
} // namespace
