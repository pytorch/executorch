/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace vgf {

constexpr const char* kVgfDiagnosticsCapabilitiesEventName =
    "VGF::capabilities";
constexpr const char* kVgfDiagnosticsExecutionEventName = "VGF::runtime_report";
constexpr const char* kVgfDiagnosticsInputCopyEventName = "VGF::input_copy";
constexpr const char* kVgfDiagnosticsOutputCopyEventName = "VGF::output_copy";
constexpr const char* kVgfDiagnosticsDirEnv = "EXECUTORCH_VGF_DIAGNOSTICS_DIR";
constexpr const char* kVgfDiagnosticsSourceRevisionEnv =
    "EXECUTORCH_VGF_SOURCE_REVISION";

struct VgfExtensionCapability {
  std::string name;
  bool available = false;
  uint32_t spec_version = 0;
};

struct VgfIoCapability {
  size_t io_index = 0;
  std::string direction;
  std::string descriptor_type;
  uint32_t vk_format = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  uint64_t logical_bytes = 0;

  // Requirements/properties for the persistently mapped IO allocation. For
  // image IO this is the host-visible staging buffer, not the optimal image.
  uint64_t memory_requirement_size = 0;
  uint64_t memory_requirement_alignment = 0;
  uint64_t memory_allocation_capacity = 0;
  uint32_t memory_type_bits = 0;
  uint32_t memory_type_index = 0xffffffffu;
  uint32_t memory_property_flags = 0;
  std::string dedicated_allocation_requirement = "unknown";

  // Reaching capability emission means the exact resource description used by
  // this IO was successfully created and bound during process_vgf().
  bool exact_resource_created = false;
  bool persistent_mapped = false;
  bool device_staging_copy = false;
  bool tensor_image_aliasing = false;
};

struct VgfCapabilityReport {
  uint64_t method_instance_id = 0;
  std::string source_revision;
  int32_t mlsdk_vgf_api_major = -1;
  int32_t mlsdk_vgf_api_minor = -1;

  uint32_t vulkan_api_version = 0;
  uint32_t driver_version = 0;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  std::string device_name;
  uint32_t queue_family_index = 0xffffffffu;
  uint32_t queue_flags = 0;
  uint64_t non_coherent_atom_size = 0;
  uint64_t buffer_image_granularity = 0;

  uint32_t memory_type_count = 0;
  uint32_t host_visible_memory_type_count = 0;
  uint32_t host_coherent_memory_type_count = 0;
  uint32_t device_local_memory_type_count = 0;

  bool device_extension_enumeration_ok = false;
  bool tensor_capabilities_queried = false;
  bool tensor_feature_tensors = false;
  bool tensor_feature_tensor_non_packed = false;
  bool tensor_feature_shader_tensor_access = false;
  uint32_t max_tensor_dimension_count = 0;
  uint64_t max_tensor_elements = 0;
  uint64_t max_tensor_size = 0;
  uint64_t max_tensor_stride = 0;
  bool external_memory_host_properties_queried = false;
  uint64_t min_imported_host_pointer_alignment = 0;

  size_t model_input_count = 0;
  size_t model_output_count = 0;
  size_t io_count = 0;
  size_t persistently_mapped_io_count = 0;
  size_t unique_persistent_mapping_count = 0;
  size_t init_image_layout_transition_count = 0;
  uint32_t frame_slot_count = 1;
  bool multiple_in_flight_invocations_supported = false;
  std::string io_mapping_creation_phase = "process_vgf_init";
  std::string io_mapping_lifetime = "until_VgfRepr_free_vgf";

  // Current VGF allocation policy requests these properties for CPU-visible
  // IO memory. Report the policy and actual selected property flags per IO.
  bool io_policy_requires_host_visible = true;
  bool io_policy_requires_host_coherent = true;

  std::vector<VgfExtensionCapability> extensions;
  std::vector<VgfIoCapability> ios;
};

struct VgfTransferObservation {
  std::string kind;
  std::string direction;
  size_t slot = 0;
  int io_index = -1;
  uint64_t bytes_copied = 0;
  std::string transfer_mode = "STAGED_COPY";
  std::string binding_provenance = "BACKEND_OWNED";
  std::string descriptor_type;
};

struct VgfExecutionReport {
  uint64_t method_instance_id = 0;
  uint64_t invocation = 0;
  bool success = false;
  bool synchronous_execution = true;

  uint64_t input_cpu_copy_count = 0;
  uint64_t input_cpu_copy_bytes = 0;
  uint64_t output_cpu_copy_count = 0;
  uint64_t output_cpu_copy_bytes = 0;

  uint64_t input_device_copy_count = 0;
  uint64_t input_device_copy_bytes = 0;
  uint64_t output_device_copy_count = 0;
  uint64_t output_device_copy_bytes = 0;

  uint64_t queue_submit_count = 0;
  uint64_t fence_wait_count = 0;
  uint64_t host_to_device_barrier_count = 0;
  uint64_t device_to_host_barrier_count = 0;
  uint64_t segment_barrier_count = 0;
  uint64_t input_image_transfer_barrier_count = 0;
  uint64_t output_image_transfer_barrier_count = 0;
  uint64_t image_layout_transition_barrier_count = 0;
  uint64_t explicit_flush_count = 0;
  uint64_t explicit_invalidate_count = 0;

  // These are zero specifically inside the VGF runtime. Portable q/dq/layout
  // work outside the delegate is not visible here and is reported separately.
  uint64_t vgf_runtime_cpu_quantize_bytes = 0;
  uint64_t vgf_runtime_cpu_dequantize_bytes = 0;
  uint64_t vgf_runtime_cpu_layout_conversion_bytes = 0;
  std::string portable_boundary_conversion_visibility = "unknown";
  std::string vgf_internal_transfer_visibility = "unknown";
};

std::string serialize_vgf_capability_report(const VgfCapabilityReport& report);
std::string serialize_vgf_transfer_observation(
    const VgfTransferObservation& observation);
std::string serialize_vgf_execution_report(const VgfExecutionReport& report);

// If EXECUTORCH_VGF_DIAGNOSTICS_DIR is unset, this is a no-op that returns
// true and clears written_path. When enabled, writes atomically and returns the
// resulting file path in written_path.
bool write_vgf_diagnostics_report(
    const std::string& file_name,
    const std::string& json,
    std::string* written_path = nullptr);

} // namespace vgf
} // namespace backends
} // namespace executorch
