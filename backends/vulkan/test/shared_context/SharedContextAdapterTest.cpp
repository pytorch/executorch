/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/SharedContext.h>
#include <executorch/backends/vulkan/runtime/api/Context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace executorch::backends::vulkan_shared;
using executorch::backends::vulkan::create_vulkan_shared_context;
using executorch::backends::vulkan::resolve_vulkan_shared_adapter;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::Span;
namespace vkapi = vkcompute::vkapi;

BackendInitContext context_for(BackendOptions<3>& options) {
  auto view = options.view();
  return BackendInitContext(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      Span<const BackendOption>(view.data(), view.size()));
}

TEST(SharedContextAdapterBuildTest, DisabledDoesNotInitializeVulkan) {
  BackendInitContext init(nullptr);
  auto result = resolve_vulkan_shared_adapter(init);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), nullptr);
}

TEST(SharedContextAdapterBuildTest, InvalidArgumentsDoNotEnterInitialization) {
  executorch::runtime::runtime_init();
  std::array<uint8_t, 16> bytes{};
  int releases = 0;
  executorch::runtime::FreeableBuffer buffer(
      bytes.data(),
      bytes.size(),
      [](void* state, void*, size_t) { ++*static_cast<int*>(state); },
      &releases);
  BackendInitContext init(nullptr);
  auto* backend = executorch::runtime::get_backend_class("VulkanBackend");
  ASSERT_NE(backend, nullptr);
  auto result = backend->init(init, &buffer, {});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
  EXPECT_EQ(releases, 0);
  buffer.Free();
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(backend->init(init, nullptr, {}).error(), Error::InvalidArgument);
}

class SharedContextAdapterTest : public ::testing::Test {
 protected:
  SharedVulkanContextPtr shared;
  SharedVulkanContextKey key{"phase3-gpu-test", 7};

  void SetUp() override {
    const char* enabled = std::getenv("ETVK_RUN_GPU_TESTS");
    if (!enabled || std::string(enabled) != "1") {
      GTEST_SKIP()
          << "Set ETVK_RUN_GPU_TESTS=1 to require real Vulkan execution";
    }
    executorch::runtime::runtime_init();
    SharedVulkanContextRegistry::Get().clear_for_testing();
    auto created = create_vulkan_shared_context(key);
    ASSERT_TRUE(created.ok())
        << "Requested GPU tests require a working Vulkan ICD";
    shared = created.get();
    ASSERT_EQ(
        SharedVulkanContextRegistry::Get().register_context(shared), Error::Ok);
  }

  void TearDown() override {
    SharedVulkanContextRegistry::Get().clear_for_testing();
    shared.reset();
  }

  // Test-only copy used to mutate context metadata. The copied context shares
  // the same VkQueue as the original but owns a different queue mutex, so the
  // two contexts must not perform queue operations concurrently.
  SharedVulkanContextCreateInfo copy_info() {
    SharedVulkanContextCreateInfo info;
    info.key = key;
    info.instance = shared->instance();
    info.physical_device = shared->physical_device();
    info.device = shared->device();

    shared->with_locked_queue([&](VkQueue queue) { info.queue = queue; });
    info.queue_family_index = shared->queue_family_index();
    info.lifetime_anchor = shared;
    return info;
  }
};

// One command pool and fence per submission thread; no command-pool sharing.
void submit_empty(vkapi::Adapter& adapter) {
  auto queue = adapter.request_queue();
  struct Resources final {
    vkapi::Adapter& adapter;
    vkapi::Adapter::Queue queue;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    ~Resources() {
      try {
        adapter.wait_idle(queue);
      } catch (...) {
      }
      if (fence) {
        vkDestroyFence(adapter.device_handle(), fence, nullptr);
      }
      if (pool) {
        vkDestroyCommandPool(adapter.device_handle(), pool, nullptr);
      }
      adapter.return_queue(queue);
    }
  } resources{adapter, queue};
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue.family_index;
  VK_CHECK(vkCreateCommandPool(
      adapter.device_handle(), &pool_info, nullptr, &resources.pool));
  VkCommandBufferAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate.commandPool = resources.pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VK_CHECK(
      vkAllocateCommandBuffers(adapter.device_handle(), &allocate, &command));
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(command, &begin));
  VK_CHECK(vkEndCommandBuffer(command));
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(
      adapter.device_handle(), &fence_info, nullptr, &resources.fence));
  adapter.submit_cmd(queue, command, resources.fence);
  VK_CHECK(vkWaitForFences(
      adapter.device_handle(), 1, &resources.fence, VK_TRUE, 10000000000ull));
  adapter.wait_idle(queue);
}

TEST_F(SharedContextAdapterTest, ReusesHandlesAndOnlyRegisteredQueueMetadata) {
  vkapi::Adapter first(shared, ""), second(shared, "");
  EXPECT_EQ(first.instance_handle(), shared->instance());
  EXPECT_EQ(first.device_handle(), second.device_handle());
  EXPECT_EQ(first.physical_handle(), shared->physical_device());
  auto queue = first.request_queue();
  EXPECT_EQ(queue.family_index, shared->queue_family_index());
  EXPECT_EQ(queue.handle, VK_NULL_HANDLE); // No unlocked queue escapes.
  EXPECT_EQ(first.num_compute_queues(), 1u);
  first.return_queue(queue);
  EXPECT_NO_THROW(submit_empty(first));
  EXPECT_NO_THROW(submit_empty(second));
}

TEST_F(SharedContextAdapterTest, RejectsForeignQueueMetadataBeforeSubmission) {
  vkapi::Adapter adapter(shared, "");
  auto queue = adapter.request_queue();
  auto wrong = queue;
  wrong.queue_index = 1;
  EXPECT_ANY_THROW(adapter.wait_idle(wrong));
  EXPECT_ANY_THROW(adapter.submit_cmd(wrong, VK_NULL_HANDLE));
  adapter.return_queue(queue);
  EXPECT_NO_THROW(submit_empty(adapter));
}

TEST_F(SharedContextAdapterTest, ImportedFeaturesAreConservativelyDisabled) {
  auto info = copy_info();
  auto conservative = std::make_shared<SharedVulkanContext>(std::move(info));
  vkapi::Adapter adapter(conservative, "");
  EXPECT_FALSE(adapter.supports_int16_shader_types());
  EXPECT_FALSE(adapter.supports_int64_shader_types());
  EXPECT_FALSE(adapter.supports_float64_shader_types());
  EXPECT_FALSE(adapter.supports_16bit_storage_buffers());
  EXPECT_FALSE(adapter.supports_8bit_storage_buffers());
  EXPECT_FALSE(adapter.supports_float16_shader_types());
  EXPECT_FALSE(adapter.supports_int8_shader_types());
  EXPECT_FALSE(adapter.supports_int8_dot_product());
  EXPECT_FALSE(adapter.supports_cooperative_matrix());
  EXPECT_FALSE(adapter.supports_int8_cooperative_matrix());
  EXPECT_FALSE(adapter.supports_nv_cooperative_matrix2());
  EXPECT_FALSE(adapter.supports_subgroup_size_control());
  EXPECT_FALSE(adapter.supports_compute_full_subgroups());
  EXPECT_NO_THROW(submit_empty(adapter));
}

TEST_F(
    SharedContextAdapterTest,
    RejectsInvalidQueueFamilyWithoutDestroyingDevice) {
  auto info = copy_info();
  info.queue_family_index = std::numeric_limits<uint32_t>::max() - 1;
  auto incompatible = std::make_shared<SharedVulkanContext>(std::move(info));
  EXPECT_ANY_THROW(vkapi::Adapter adapter(incompatible, ""));
  vkapi::Adapter valid(shared, "");
  EXPECT_NO_THROW(submit_empty(valid));
}

TEST_F(SharedContextAdapterTest, RejectsInvalidContextBeforeVulkanCalls) {
  EXPECT_ANY_THROW(vkapi::Adapter adapter(SharedVulkanContextPtr{}, ""));
  auto invalid =
      std::make_shared<SharedVulkanContext>(SharedVulkanContextCreateInfo{});
  EXPECT_ANY_THROW(vkapi::Adapter adapter(invalid, ""));
}

TEST_F(
    SharedContextAdapterTest,
    BorrowedDeviceHandleSurvivesExceptionUnwinding) {
  EXPECT_THROW(
      ([&] {
        vkapi::DeviceHandle borrowed(shared->device(), false);
        throw std::runtime_error("simulate later Adapter member throwing");
      }()),
      std::runtime_error);
  vkapi::Adapter still_valid(shared, "");
  EXPECT_NO_THROW(submit_empty(still_valid));
}

TEST_F(SharedContextAdapterTest, UnregisterKeepsDeviceAliveUntilLastBorrower) {
  std::weak_ptr<SharedVulkanContext> weak = shared;
  auto first = std::make_unique<vkapi::Adapter>(shared, "");
  auto second = std::make_unique<vkapi::Adapter>(shared, "");
  ASSERT_EQ(
      SharedVulkanContextRegistry::Get().unregister_context(key), Error::Ok);
  shared.reset();
  first.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_NO_THROW(submit_empty(*second));
  second.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(SharedContextAdapterTest, IndependentAdaptersSubmitAndWaitConcurrently) {
  constexpr size_t count = 4;
  std::array<std::exception_ptr, count> errors{};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < count; ++i) {
    threads.emplace_back([&, i, retained = shared] {
      try {
        vkapi::Adapter adapter(retained, "");
        for (int repeat = 0; repeat < 8; ++repeat) {
          submit_empty(adapter);
        }
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    EXPECT_FALSE(static_cast<bool>(error));
  }
}

TEST_F(
    SharedContextAdapterTest,
    DifferentKeysShareInstanceAndHaveIndependentDevices) {
  auto other = create_vulkan_shared_context({"phase3-other-device", 99});
  ASSERT_TRUE(other.ok());
  // We keep one Volk-loaded VkInstance but allow independent
  // logical devices beneath it.
  EXPECT_EQ(other.get()->instance(), shared->instance());
  EXPECT_NE(other.get()->device(), shared->device());
  vkapi::Adapter first(shared, ""), second(other.get(), "");
  EXPECT_NO_THROW(submit_empty(first));
  EXPECT_NO_THROW(submit_empty(second));
  EXPECT_NO_THROW(
      submit_empty(first)); // No global device dispatch was rebound.
}

TEST_F(
    SharedContextAdapterTest,
    RejectsContextFromDifferentInstanceBeforeVulkanCalls) {
  auto info = copy_info();
  const SharedVulkanContextKey foreign_key{"foreign-instance", 3};
  info.key = foreign_key;
  info.instance =
      reinterpret_cast<VkInstance>(static_cast<std::uintptr_t>(0x1234));
  auto foreign = std::make_shared<SharedVulkanContext>(std::move(info));
  ASSERT_TRUE(foreign->is_valid());
  ASSERT_EQ(
      SharedVulkanContextRegistry::Get().register_context(foreign), Error::Ok);

  BackendOptions<3> options;
  ASSERT_EQ(
      options.set_option(
          kSharedContextNameOption, foreign_key.context_name.c_str()),
      Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedGroupIdOption, foreign_key.group_id),
      Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);

  auto result = resolve_vulkan_shared_adapter(context_for(options));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::DelegateInvalidCompatibility);
  EXPECT_EQ(SharedVulkanContextRegistry::Get().lookup(foreign_key), foreign);
  EXPECT_EQ(
      SharedVulkanContextRegistry::Get().unregister_context(foreign_key),
      Error::Ok);
}

TEST_F(SharedContextAdapterTest, RuntimeOptionsReachTheRealAdapter) {
  BackendOptions<3> options;
  ASSERT_EQ(
      options.set_option(kSharedContextNameOption, "phase3-gpu-test"),
      Error::Ok);
  ASSERT_EQ(options.set_option(kSharedGroupIdOption, 7), Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);
  auto result = resolve_vulkan_shared_adapter(context_for(options));
  ASSERT_TRUE(result.ok());
  ASSERT_NE(result.get(), nullptr);
  EXPECT_EQ(result.get()->device_handle(), shared->device());
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "create_only"), Error::Ok);
  auto duplicate = resolve_vulkan_shared_adapter(context_for(options));
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error(), Error::AlreadyLoaded);
  ASSERT_EQ(
      options.set_option(kSharedContextNameOption, "missing-phase3"),
      Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);
  EXPECT_EQ(
      resolve_vulkan_shared_adapter(context_for(options)).error(),
      Error::NotFound);
}

TEST_F(
    SharedContextAdapterTest,
    FailedBackendCompilationReleasesBufferAndBorrower) {
  std::array<uint8_t, 65536> arena{};
  executorch::runtime::MemoryAllocator allocator(arena.size(), arena.data());
  BackendOptions<3> options;
  ASSERT_EQ(
      options.set_option(kSharedContextNameOption, "phase3-gpu-test"),
      Error::Ok);
  ASSERT_EQ(options.set_option(kSharedGroupIdOption, 7), Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);
  auto view = options.view();
  BackendInitContext context(
      &allocator,
      nullptr,
      nullptr,
      nullptr,
      Span<const BackendOption>(view.data(), view.size()));
  std::array<uint8_t, 64> invalid_blob{};
  int releases = 0;
  executorch::runtime::FreeableBuffer buffer(
      invalid_blob.data(),
      invalid_blob.size(),
      [](void* state, void*, size_t) { ++*static_cast<int*>(state); },
      &releases);
  auto* backend = executorch::runtime::get_backend_class("VulkanBackend");
  ASSERT_NE(backend, nullptr);
  const auto before = shared.use_count();
  auto result = backend->init(context, &buffer, {});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(shared.use_count(), before);
  EXPECT_EQ(SharedVulkanContextRegistry::Get().lookup(key), shared);
  vkapi::Adapter valid(shared, "");
  EXPECT_NO_THROW(submit_empty(valid));
}

TEST_F(SharedContextAdapterTest, BorrowsApplicationDeviceWithNonDefaultQueue) {
  vkapi::PhysicalDevice physical(shared->instance(), shared->physical_device());
  uint32_t family = std::numeric_limits<uint32_t>::max();
  uint32_t index = 0;
  for (uint32_t i = 0; i < physical.queue_families.size(); ++i) {
    const auto& q = physical.queue_families[i];
    if (!(q.queueFlags & VK_QUEUE_COMPUTE_BIT) || q.queueCount == 0) {
      continue;
    }
    if (i != shared->queue_family_index() || q.queueCount > 1) {
      family = i;
      index = q.queueCount > 1 ? 1 : 0;
      break;
    }
  }
  if (family == std::numeric_limits<uint32_t>::max()) {
    GTEST_SKIP() << "This device exposes only the default compute queue";
  }
  std::vector<float> priorities(index + 1, 1.0f);
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = family;
  queue_info.queueCount = index + 1;
  queue_info.pQueuePriorities = priorities.data();
  std::vector<const char*> extensions;
  if (shared->has_device_extension("VK_KHR_portability_subset")) {
    extensions.push_back("VK_KHR_portability_subset");
  }
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  device_info.ppEnabledExtensionNames = extensions.data();
  VkDevice device = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDevice(
      shared->physical_device(), &device_info, nullptr, &device));
  vkapi::DeviceHandle device_guard(device);
  struct Owner final {
    SharedVulkanContextPtr instance_owner;
    vkapi::DeviceHandle device;
    Owner(SharedVulkanContextPtr owner, VkDevice handle)
        : instance_owner(std::move(owner)), device(handle) {}
  };
  auto owner = std::make_shared<Owner>(shared, device);
  device_guard.handle = VK_NULL_HANDLE;
  VkQueue supplied = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, family, index, &supplied);
  ASSERT_NE(supplied, VK_NULL_HANDLE);
  SharedVulkanContextCreateInfo info;
  info.key = {"application-nondefault", 1};
  info.instance = shared->instance();
  info.physical_device = shared->physical_device();
  info.device = device;
  info.queue = supplied;
  info.queue_family_index = family;
  info.enabled_device_extensions.assign(extensions.begin(), extensions.end());
  info.lifetime_anchor = owner;
  auto external = std::make_shared<SharedVulkanContext>(std::move(info));
  vkapi::Adapter adapter(external, "");
  auto metadata = adapter.request_queue();
  EXPECT_EQ(metadata.family_index, family);
  EXPECT_EQ(metadata.handle, VK_NULL_HANDLE);
  adapter.return_queue(metadata);
  EXPECT_NO_THROW(submit_empty(adapter));
}
} // namespace
