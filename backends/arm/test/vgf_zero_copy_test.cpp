/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/arm/runtime/VGFZeroCopy.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace executorch {
namespace backends {
namespace vgf {
namespace {

#if defined(VK_EXT_external_memory_host)

template <typename Handle>
Handle fake_handle(uintptr_t value) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(value);
  } else {
    return static_cast<Handle>(value);
  }
}

struct FakeVulkanState {
  VkResult host_pointer_properties_result = VK_SUCCESS;
  VkResult create_tensor_result = VK_SUCCESS;
  VkResult allocate_memory_result = VK_SUCCESS;
  VkResult bind_tensor_result = VK_SUCCESS;
  VkResult create_tensor_view_result = VK_SUCCESS;

  uint32_t host_memory_type_bits = 0x1;
  uint32_t tensor_memory_type_bits = 0x1;
  VkDeviceSize tensor_memory_size = 64;
  VkBool32 requires_dedicated_allocation = VK_FALSE;
  VkMemoryPropertyFlags memory_type_flags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  uint32_t create_tensor_calls = 0;
  uint32_t allocate_memory_calls = 0;
  uint32_t bind_tensor_calls = 0;
  uint32_t create_tensor_view_calls = 0;
  uint32_t destroy_tensor_view_calls = 0;
  uint32_t destroy_tensor_calls = 0;
  uint32_t free_memory_calls = 0;

  void* queried_host_pointer = nullptr;
  VkDeviceSize allocation_size = 0;
  uint32_t allocation_memory_type_index = UINT32_MAX;
  VkExternalMemoryHandleTypeFlags tensor_external_handle_types = 0;
  VkTensorTilingARM tensor_tiling = VK_TENSOR_TILING_OPTIMAL_ARM;
  VkFormat tensor_format = VK_FORMAT_UNDEFINED;
  VkTensorUsageFlagsARM tensor_usage = 0;
  std::vector<int64_t> tensor_dimensions;
  std::vector<int64_t> tensor_strides;
  std::vector<std::string> destruction_order;
};

VkResult fake_get_memory_host_pointer_properties(
    void* context,
    VkDevice,
    VkExternalMemoryHandleTypeFlagBits handle_type,
    const void* host_pointer,
    VkMemoryHostPointerPropertiesEXT* properties) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  EXPECT_EQ(
      handle_type, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
  state.queried_host_pointer = const_cast<void*>(host_pointer);
  properties->memoryTypeBits = state.host_memory_type_bits;
  return state.host_pointer_properties_result;
}

void fake_get_physical_device_memory_properties(
    void* context,
    VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties* properties) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  *properties = {};
  properties->memoryTypeCount = 1;
  properties->memoryTypes[0].propertyFlags = state.memory_type_flags;
}

VkResult fake_create_tensor(
    void* context,
    VkDevice,
    const VkTensorCreateInfoARM* create_info,
    const VkAllocationCallbacks*,
    VkTensorARM* tensor) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.create_tensor_calls;
  const auto* external =
      static_cast<const VkExternalMemoryTensorCreateInfoARM*>(
          create_info->pNext);
  EXPECT_NE(external, nullptr);
  state.tensor_external_handle_types = external->handleTypes;
  const auto* description = create_info->pDescription;
  state.tensor_tiling = description->tiling;
  state.tensor_format = description->format;
  state.tensor_usage = description->usage;
  state.tensor_dimensions.assign(
      description->pDimensions,
      description->pDimensions + description->dimensionCount);
  if (description->pStrides != nullptr) {
    state.tensor_strides.assign(
        description->pStrides,
        description->pStrides + description->dimensionCount);
  } else {
    state.tensor_strides.clear();
  }
  if (state.create_tensor_result == VK_SUCCESS) {
    *tensor = fake_handle<VkTensorARM>(0x1000 + state.create_tensor_calls);
  }
  return state.create_tensor_result;
}

void fake_get_tensor_memory_requirements(
    void* context,
    VkDevice,
    const VkTensorMemoryRequirementsInfoARM*,
    VkMemoryRequirements2* requirements) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  requirements->memoryRequirements.size = state.tensor_memory_size;
  requirements->memoryRequirements.alignment = 64;
  requirements->memoryRequirements.memoryTypeBits =
      state.tensor_memory_type_bits;
  auto* dedicated =
      static_cast<VkMemoryDedicatedRequirements*>(requirements->pNext);
  if (dedicated != nullptr) {
    dedicated->requiresDedicatedAllocation =
        state.requires_dedicated_allocation;
  }
}

VkResult fake_allocate_memory(
    void* context,
    VkDevice,
    const VkMemoryAllocateInfo* allocate_info,
    const VkAllocationCallbacks*,
    VkDeviceMemory* memory) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.allocate_memory_calls;
  state.allocation_size = allocate_info->allocationSize;
  state.allocation_memory_type_index = allocate_info->memoryTypeIndex;
  const auto* import_info =
      static_cast<const VkImportMemoryHostPointerInfoEXT*>(
          allocate_info->pNext);
  EXPECT_NE(import_info, nullptr);
  EXPECT_EQ(
      import_info->handleType,
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
  EXPECT_EQ(import_info->pHostPointer, state.queried_host_pointer);
  if (state.allocate_memory_result == VK_SUCCESS) {
    *memory = fake_handle<VkDeviceMemory>(0x2000 + state.allocate_memory_calls);
  }
  return state.allocate_memory_result;
}

VkResult fake_bind_tensor_memory(
    void* context,
    VkDevice,
    uint32_t bind_info_count,
    const VkBindTensorMemoryInfoARM*) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  EXPECT_EQ(bind_info_count, 1u);
  ++state.bind_tensor_calls;
  return state.bind_tensor_result;
}

VkResult fake_create_tensor_view(
    void* context,
    VkDevice,
    const VkTensorViewCreateInfoARM*,
    const VkAllocationCallbacks*,
    VkTensorViewARM* tensor_view) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.create_tensor_view_calls;
  if (state.create_tensor_view_result == VK_SUCCESS) {
    *tensor_view =
        fake_handle<VkTensorViewARM>(0x3000 + state.create_tensor_view_calls);
  }
  return state.create_tensor_view_result;
}

void fake_destroy_tensor_view(
    void* context,
    VkDevice,
    VkTensorViewARM,
    const VkAllocationCallbacks*) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.destroy_tensor_view_calls;
  state.destruction_order.emplace_back("view");
}

void fake_destroy_tensor(
    void* context,
    VkDevice,
    VkTensorARM,
    const VkAllocationCallbacks*) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.destroy_tensor_calls;
  state.destruction_order.emplace_back("tensor");
}

void fake_free_memory(
    void* context,
    VkDevice,
    VkDeviceMemory,
    const VkAllocationCallbacks*) {
  auto& state = *static_cast<FakeVulkanState*>(context);
  ++state.free_memory_calls;
  state.destruction_order.emplace_back("memory");
}

VgfZeroCopyVulkanApi fake_api(FakeVulkanState* state) {
  VgfZeroCopyVulkanApi api{};
  api.context = state;
  api.get_memory_host_pointer_properties =
      fake_get_memory_host_pointer_properties;
  api.get_physical_device_memory_properties =
      fake_get_physical_device_memory_properties;
  api.create_tensor = fake_create_tensor;
  api.get_tensor_memory_requirements = fake_get_tensor_memory_requirements;
  api.allocate_memory = fake_allocate_memory;
  api.bind_tensor_memory = fake_bind_tensor_memory;
  api.create_tensor_view = fake_create_tensor_view;
  api.destroy_tensor_view = fake_destroy_tensor_view;
  api.destroy_tensor = fake_destroy_tensor;
  api.free_memory = fake_free_memory;
  return api;
}

VgfHostMemoryImportCapabilities enabled_capabilities() {
  VgfHostMemoryImportCapabilities caps{};
  caps.physical_device_advertised = true;
  caps.logical_device_enabled = true;
  caps.min_imported_host_pointer_alignment = 64;
  return caps;
}

VgfZeroCopyIoMetadata eligible_metadata() {
  VgfZeroCopyIoMetadata metadata{};
  metadata.eligible = true;
  metadata.is_model_boundary_resource = true;
  metadata.mapped_to_model_boundary = true;
  metadata.is_input = true;
  metadata.executorch_argument_index = 0;
  metadata.descriptor_type = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
  metadata.tensor_backed = true;
  metadata.tiling = VK_TENSOR_TILING_LINEAR_ARM;
  metadata.format = VK_FORMAT_R32_SFLOAT;
  metadata.usage = vgf_tensor_usage_flags(false);
  metadata.dimensions = {1, 16};
  metadata.has_explicit_strides = true;
  metadata.strides = {16, 1};
  metadata.host_memory_import_advertised = true;
  metadata.host_memory_import_enabled = true;
  metadata.external_tensor_query_performed = true;
  metadata.external_tensor_host_allocation_supported = true;
  metadata.external_tensor_requires_dedicated_allocation = false;
  metadata.bindings.push_back(
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 0});
  return metadata;
}

// The manager is intentionally non-movable/non-copyable.
void reset_manager(
    VgfZeroCopyImportManager* manager,
    size_t io_count,
    FakeVulkanState* state) {
  const auto api = fake_api(state);
  manager->reset(
      fake_handle<VkPhysicalDevice>(0x10),
      fake_handle<VkDevice>(0x20),
      enabled_capabilities(),
      io_count,
      &api);
}

// Cppcheck cannot expand GoogleTest registration macros correctly.
// The C++ compiler and test runner validate these tests.
// cppcheck-suppress-begin syntaxError

TEST(VgfZeroCopyImportTest, SuccessfulCandidateCreation) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  ASSERT_TRUE(result.accepted);
  EXPECT_FALSE(result.cache_hit);
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_TRUE(manager.slot(0)->valid());
  EXPECT_EQ(manager.slot(0)->host_pointer, host.data());
  EXPECT_EQ(manager.slot(0)->logical_size, host.size());
  EXPECT_EQ(manager.slot(0)->imported_size, host.size());
  EXPECT_EQ(state.queried_host_pointer, host.data());
  EXPECT_EQ(state.allocation_size, host.size());
  EXPECT_EQ(state.allocation_memory_type_index, 0u);
  EXPECT_EQ(
      state.tensor_external_handle_types,
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
  EXPECT_EQ(state.tensor_tiling, VK_TENSOR_TILING_LINEAR_ARM);
  EXPECT_EQ(state.tensor_format, VK_FORMAT_R32_SFLOAT);
  EXPECT_EQ(state.tensor_usage, vgf_tensor_usage_flags(false));
  EXPECT_EQ(state.tensor_dimensions, (std::vector<int64_t>{1, 16}));
  EXPECT_EQ(state.tensor_strides, (std::vector<int64_t>{16, 1}));
}

TEST(VgfZeroCopyImportTest, BadPointerAlignmentFallsBack) {
  alignas(64) std::array<std::byte, 192> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result = manager.try_import_candidate(
      0, host.data() + 1, 128, eligible_metadata());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, VgfZeroCopyImportRejectionReason::kPointerAlignment);
  EXPECT_EQ(state.create_tensor_calls, 0u);
}

TEST(VgfZeroCopyImportTest, LogicalSizeMustMeetPhase1AlignmentPolicy) {
  alignas(64) std::array<std::byte, 192> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result =
      manager.try_import_candidate(0, host.data(), 96, eligible_metadata());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(
      result.reason, VgfZeroCopyImportRejectionReason::kLogicalSizeAlignment);
  EXPECT_EQ(state.create_tensor_calls, 0u);
}

TEST(VgfZeroCopyImportTest, UnsupportedExternalTensorFallsBack) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);
  auto metadata = eligible_metadata();
  metadata.eligible = false;
  metadata.external_tensor_host_allocation_supported = false;

  const auto result =
      manager.try_import_candidate(0, host.data(), host.size(), metadata);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(
      result.reason,
      VgfZeroCopyImportRejectionReason::kExternalTensorUnsupported);
  EXPECT_EQ(state.create_tensor_calls, 0u);
}

TEST(VgfZeroCopyImportTest, NoCompatibleCoherentMemoryTypeFallsBack) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  state.memory_type_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(
      result.reason,
      VgfZeroCopyImportRejectionReason::kNoCompatibleCoherentMemoryType);
  EXPECT_EQ(state.destroy_tensor_calls, 1u);
  EXPECT_EQ(state.free_memory_calls, 0u);
}

TEST(
    VgfZeroCopyImportTest,
    TensorRequirementsLargerThanKnownHostRangeFallBack) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  state.tensor_memory_size = 192;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(
      result.reason,
      VgfZeroCopyImportRejectionReason::kMemoryRequirementsTooLarge);
  EXPECT_EQ(state.allocate_memory_calls, 0u);
  EXPECT_EQ(state.destroy_tensor_calls, 1u);
}

TEST(VgfZeroCopyImportTest, DedicatedAllocationRequirementFallsBack) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  state.requires_dedicated_allocation = VK_TRUE;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  const auto result = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(
      result.reason,
      VgfZeroCopyImportRejectionReason::kDedicatedAllocationUnsupported);
  EXPECT_EQ(state.allocate_memory_calls, 0u);
  EXPECT_EQ(state.destroy_tensor_calls, 1u);
}

TEST(VgfZeroCopyImportTest, PartialCreationFailureKeepsPreviousImport) {
  alignas(64) std::array<std::byte, 128> first_host{};
  alignas(64) std::array<std::byte, 128> replacement_host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(
      manager
          .try_import_candidate(
              0, first_host.data(), first_host.size(), eligible_metadata())
          .accepted);
  const auto old_tensor = manager.slot(0)->tensor;
  const auto old_memory = manager.slot(0)->memory;
  const auto old_view = manager.slot(0)->tensor_view;

  state.create_tensor_view_result = VK_ERROR_INITIALIZATION_FAILED;
  state.destruction_order.clear();
  const auto replacement = manager.try_import_candidate(
      0, replacement_host.data(), replacement_host.size(), eligible_metadata());

  EXPECT_FALSE(replacement.accepted);
  EXPECT_EQ(
      replacement.reason,
      VgfZeroCopyImportRejectionReason::kTensorViewCreateFailed);
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_EQ(manager.slot(0)->host_pointer, first_host.data());
  EXPECT_EQ(manager.slot(0)->tensor, old_tensor);
  EXPECT_EQ(manager.slot(0)->memory, old_memory);
  EXPECT_EQ(manager.slot(0)->tensor_view, old_view);
  EXPECT_EQ(
      state.destruction_order, (std::vector<std::string>{"tensor", "memory"}));
}

TEST(VgfZeroCopyImportTest, DuplicateActiveRangeOwnedByAnotherIoFallsBack) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 2, &state);

  ASSERT_TRUE(manager
                  .try_import_candidate(
                      0, host.data(), host.size(), eligible_metadata())
                  .accepted);
  const uint32_t create_calls = state.create_tensor_calls;
  const auto duplicate = manager.try_import_candidate(
      1, host.data(), host.size(), eligible_metadata());

  EXPECT_FALSE(duplicate.accepted);
  EXPECT_EQ(
      duplicate.reason,
      VgfZeroCopyImportRejectionReason::kDuplicateActiveRange);
  EXPECT_EQ(state.create_tensor_calls, create_calls);
}

TEST(VgfZeroCopyImportTest, PartiallyOverlappingActiveRangeFallsBack) {
  alignas(64) std::array<std::byte, 256> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 2, &state);

  ASSERT_TRUE(
      manager.try_import_candidate(0, host.data(), 128, eligible_metadata())
          .accepted);
  const uint32_t create_calls = state.create_tensor_calls;
  const auto overlap = manager.try_import_candidate(
      1, host.data() + 64, 128, eligible_metadata());

  EXPECT_FALSE(overlap.accepted);
  EXPECT_EQ(
      overlap.reason,
      VgfZeroCopyImportRejectionReason::kOverlappingActiveRange);
  EXPECT_EQ(state.create_tensor_calls, create_calls);
}

TEST(
    VgfZeroCopyImportTest,
    SameSlotReplacementWithLargerOverlappingRangeFallsBack) {
  alignas(64) std::array<std::byte, 256> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(
      manager.try_import_candidate(0, host.data(), 128, eligible_metadata())
          .accepted);
  const auto old_memory = manager.slot(0)->memory;
  const uint32_t create_calls = state.create_tensor_calls;
  const uint32_t allocate_calls = state.allocate_memory_calls;

  const auto replacement = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  EXPECT_FALSE(replacement.accepted);
  EXPECT_EQ(
      replacement.reason,
      VgfZeroCopyImportRejectionReason::kOverlappingActiveRange);
  EXPECT_EQ(state.create_tensor_calls, create_calls);
  EXPECT_EQ(state.allocate_memory_calls, allocate_calls);
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_EQ(manager.slot(0)->host_pointer, host.data());
  EXPECT_EQ(manager.slot(0)->logical_size, 128u);
  EXPECT_EQ(manager.slot(0)->memory, old_memory);
}

TEST(
    VgfZeroCopyImportTest,
    SameSlotReplacementWithPartiallyOverlappingRangeFallsBack) {
  alignas(64) std::array<std::byte, 256> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(
      manager.try_import_candidate(0, host.data(), 128, eligible_metadata())
          .accepted);
  const auto old_memory = manager.slot(0)->memory;
  const uint32_t create_calls = state.create_tensor_calls;
  const uint32_t allocate_calls = state.allocate_memory_calls;

  const auto replacement = manager.try_import_candidate(
      0, host.data() + 64, 128, eligible_metadata());

  EXPECT_FALSE(replacement.accepted);
  EXPECT_EQ(
      replacement.reason,
      VgfZeroCopyImportRejectionReason::kOverlappingActiveRange);
  EXPECT_EQ(state.create_tensor_calls, create_calls);
  EXPECT_EQ(state.allocate_memory_calls, allocate_calls);
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_EQ(manager.slot(0)->host_pointer, host.data());
  EXPECT_EQ(manager.slot(0)->logical_size, 128u);
  EXPECT_EQ(manager.slot(0)->memory, old_memory);
}

TEST(VgfZeroCopyImportTest, StableCacheHitDoesNotReimport) {
  alignas(64) std::array<std::byte, 128> host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(manager
                  .try_import_candidate(
                      0, host.data(), host.size(), eligible_metadata())
                  .accepted);
  const uint32_t create_calls = state.create_tensor_calls;
  const uint32_t allocate_calls = state.allocate_memory_calls;

  const auto cached = manager.try_import_candidate(
      0, host.data(), host.size(), eligible_metadata());

  EXPECT_TRUE(cached.accepted);
  EXPECT_TRUE(cached.cache_hit);
  EXPECT_EQ(state.create_tensor_calls, create_calls);
  EXPECT_EQ(state.allocate_memory_calls, allocate_calls);
}

TEST(VgfZeroCopyImportTest, FirstAcceptedImportSelectionChoosesImportedView) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7050);
  const auto imported = fake_handle<VkTensorViewARM>(0x7060);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);

  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/true, imported));

  EXPECT_TRUE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, imported);
  EXPECT_EQ(binding_state.bound_view, owned);
}

TEST(VgfZeroCopyImportTest, AcceptedCachedImportReselectsAfterOwnedFallback) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7070);
  const auto imported = fake_handle<VkTensorViewARM>(0x7080);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);

  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/true, imported));
  binding_state.mark_clean();
  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/false, VK_NULL_HANDLE));
  binding_state.mark_clean();

  // The import manager may return a cache hit for the same host allocation.
  // Accepted still means the cached imported view must become selected again.
  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/true, imported));
  EXPECT_TRUE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, imported);
}

TEST(VgfZeroCopyImportTest, RejectedImportSelectionRestoresOwnedView) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7090);
  const auto imported = fake_handle<VkTensorViewARM>(0x70a0);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/true, imported));
  binding_state.mark_clean();

  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, /*import_accepted=*/false, VK_NULL_HANDLE));

  EXPECT_FALSE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, owned);
  EXPECT_EQ(binding_state.bound_view, imported);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingOwnedToImportedMarksDirty) {
  auto metadata = eligible_metadata();
  metadata.bindings = {
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 3}};
  const auto owned = fake_handle<VkTensorViewARM>(0x7100);
  const auto imported = fake_handle<VkTensorViewARM>(0x7200);

  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported, true));

  EXPECT_TRUE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, imported);
  const auto updates = vgf_collect_boundary_descriptor_updates(
      std::vector<VgfZeroCopyIoMetadata>{metadata},
      std::vector<VgfBoundaryBindingState>{binding_state});
  ASSERT_EQ(updates.size(), 1u);
  EXPECT_EQ(updates[0].tensor_view, imported);
  EXPECT_EQ(updates[0].binding.binding, 3u);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingImportedToOwnedMarksDirty) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7300);
  const auto imported = fake_handle<VkTensorViewARM>(0x7400);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported, true));
  binding_state.mark_clean();

  EXPECT_TRUE(binding_state.select(owned, false));
  EXPECT_FALSE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, owned);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingImportedAToImportedBMarksDirty) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7500);
  const auto imported_a = fake_handle<VkTensorViewARM>(0x7600);
  const auto imported_b = fake_handle<VkTensorViewARM>(0x7700);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported_a, true));
  binding_state.mark_clean();

  EXPECT_TRUE(binding_state.select(imported_b, true));
  EXPECT_TRUE(binding_state.import_active);
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, imported_b);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingUnchangedSelectionIsNoOp) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7800);
  const auto imported = fake_handle<VkTensorViewARM>(0x7900);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported, true));
  binding_state.mark_clean();

  EXPECT_FALSE(binding_state.select(imported, true));
  EXPECT_FALSE(binding_state.dirty);
  EXPECT_TRUE(binding_state.import_active);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingPendingChangeCanCancelBeforeFlush) {
  const auto owned = fake_handle<VkTensorViewARM>(0x7950);
  const auto imported = fake_handle<VkTensorViewARM>(0x7960);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);

  ASSERT_TRUE(binding_state.select(imported, true));
  ASSERT_TRUE(binding_state.dirty);
  ASSERT_TRUE(binding_state.select(owned, false));
  EXPECT_FALSE(binding_state.dirty);
  EXPECT_EQ(binding_state.selected_view, owned);
  EXPECT_EQ(binding_state.bound_view, owned);
}

TEST(VgfZeroCopyImportTest, BoundaryBindingFansOutToEveryRecordedBinding) {
  auto metadata = eligible_metadata();
  metadata.bindings = {
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 1},
      VgfBoundaryBindingRef{.segment_index = 2, .set_index = 0, .binding = 7},
  };
  const auto owned = fake_handle<VkTensorViewARM>(0x7a00);
  const auto imported = fake_handle<VkTensorViewARM>(0x7b00);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported, true));

  const auto updates = vgf_collect_boundary_descriptor_updates(
      std::vector<VgfZeroCopyIoMetadata>{metadata},
      std::vector<VgfBoundaryBindingState>{binding_state});
  ASSERT_EQ(updates.size(), 2u);
  EXPECT_EQ(updates[0].binding.segment_index, 0u);
  EXPECT_EQ(updates[0].binding.binding, 1u);
  EXPECT_EQ(updates[1].binding.segment_index, 2u);
  EXPECT_EQ(updates[1].binding.binding, 7u);
  EXPECT_EQ(updates[0].tensor_view, imported);
  EXPECT_EQ(updates[1].tensor_view, imported);
}

TEST(VgfZeroCopyImportTest, CleanBoundaryBindingNeedsNoRefreshOrRerecord) {
  auto metadata = eligible_metadata();
  const auto owned = fake_handle<VkTensorViewARM>(0x7c00);
  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);

  EXPECT_FALSE(vgf_boundary_bindings_dirty(
      std::vector<VgfBoundaryBindingState>{binding_state}));
  EXPECT_TRUE(vgf_collect_boundary_descriptor_updates(
                  std::vector<VgfZeroCopyIoMetadata>{metadata},
                  std::vector<VgfBoundaryBindingState>{binding_state})
                  .empty());
}

TEST(
    VgfZeroCopyImportTest,
    FailedImportedReplacementCanRestoreOwnedBindingBeforeDispatch) {
  alignas(64) std::array<std::byte, 128> first_host{};
  alignas(64) std::array<std::byte, 128> replacement_host{};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(
      manager
          .try_import_candidate(
              0, first_host.data(), first_host.size(), eligible_metadata())
          .accepted);
  ASSERT_NE(manager.slot(0), nullptr);
  const auto imported_a = manager.slot(0)->tensor_view;
  const auto owned = fake_handle<VkTensorViewARM>(0x7d00);

  VgfBoundaryBindingState binding_state;
  binding_state.initialize(owned);
  ASSERT_TRUE(binding_state.select(imported_a, true));
  binding_state.mark_clean();

  state.create_tensor_view_result = VK_ERROR_INITIALIZATION_FAILED;
  const auto replacement = manager.try_import_candidate(
      0, replacement_host.data(), replacement_host.size(), eligible_metadata());
  ASSERT_FALSE(replacement.accepted);
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_EQ(manager.slot(0)->tensor_view, imported_a);

  ASSERT_TRUE(vgf_apply_import_binding_selection(
      &binding_state, replacement.accepted, VK_NULL_HANDLE));
  EXPECT_TRUE(binding_state.dirty);
  EXPECT_FALSE(binding_state.import_active);
  EXPECT_EQ(binding_state.selected_view, owned);
}

TEST(VgfZeroCopyImportTest, ReleaseUsesSafeDestructionOrderAndDoesNotOwnHost) {
  alignas(64) std::array<std::byte, 128> host{};
  host[0] = std::byte{0x2a};
  FakeVulkanState state;
  VgfZeroCopyImportManager manager;
  reset_manager(&manager, 1, &state);

  ASSERT_TRUE(manager
                  .try_import_candidate(
                      0, host.data(), host.size(), eligible_metadata())
                  .accepted);
  state.destruction_order.clear();

  manager.release(0);

  EXPECT_EQ(
      state.destruction_order,
      (std::vector<std::string>{"view", "tensor", "memory"}));
  ASSERT_NE(manager.slot(0), nullptr);
  EXPECT_FALSE(manager.slot(0)->valid());
  EXPECT_EQ(std::to_integer<int>(host[0]), 0x2a);
}

#else
TEST(VgfZeroCopyImportTest, ExternalMemoryHostHeadersUnavailable) {
  GTEST_SKIP() << "VK_EXT_external_memory_host is unavailable in these headers";
}

// cppcheck-suppress-end syntaxError

#endif

} // namespace
} // namespace vgf
} // namespace backends
} // namespace executorch
