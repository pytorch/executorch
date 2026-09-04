/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/arm/runtime/VGFDiagnostics.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace executorch {
namespace backends {
namespace vgf {
namespace {

std::atomic<uint64_t> g_vgf_diagnostics_write_sequence{0};

uint64_t current_process_id_for_diagnostics() {
#if defined(_WIN32)
  return static_cast<uint64_t>(_getpid());
#else
  return static_cast<uint64_t>(getpid());
#endif
}

std::filesystem::path process_unique_report_path(
    const std::filesystem::path& root,
    const std::string& file_name,
    uint64_t process_id) {
  const std::filesystem::path requested(file_name);
  const std::string unique_name = requested.stem().string() + ".p" +
      std::to_string(process_id) + requested.extension().string();
  return root / requested.parent_path() / unique_name;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

template <typename T>
void write_integer_array(
    std::ostringstream& out,
    const std::vector<T>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << static_cast<int64_t>(values[i]);
  }
  out << ']';
}

} // namespace

std::string serialize_vgf_capability_report(const VgfCapabilityReport& report) {
  std::ostringstream out;
  out << '{';
  out << "\"schema\":\"executorch.vgf.capabilities\"";
  out << ",\"schema_version\":1";
  out << ",\"method_instance_id\":" << report.method_instance_id;
  out << ",\"source_revision\":";
  if (report.source_revision.empty()) {
    out << "null";
  } else {
    out << json_escape(report.source_revision);
  }
  out << ",\"mlsdk_vgf_api_version\":{";
  if (report.mlsdk_vgf_api_major >= 0) {
    out << "\"major\":" << report.mlsdk_vgf_api_major;
  } else {
    out << "\"major\":null";
  }
  if (report.mlsdk_vgf_api_minor >= 0) {
    out << ",\"minor\":" << report.mlsdk_vgf_api_minor;
  } else {
    out << ",\"minor\":null";
  }
  out << '}';
  out << ",\"device\":{";
  out << "\"name\":" << json_escape(report.device_name);
  out << ",\"vendor_id\":" << report.vendor_id;
  out << ",\"device_id\":" << report.device_id;
  out << ",\"api_version\":" << report.vulkan_api_version;
  out << ",\"driver_version\":" << report.driver_version;
  out << ",\"queue_family_index\":" << report.queue_family_index;
  out << ",\"queue_flags\":" << report.queue_flags;
  out << '}';
  out << ",\"limits\":{";
  out << "\"non_coherent_atom_size\":" << report.non_coherent_atom_size;
  out << ",\"buffer_image_granularity\":" << report.buffer_image_granularity;
  out << '}';
  out << ",\"memory_types\":{";
  out << "\"count\":" << report.memory_type_count;
  out << ",\"host_visible_count\":" << report.host_visible_memory_type_count;
  out << ",\"host_coherent_count\":" << report.host_coherent_memory_type_count;
  out << ",\"device_local_count\":" << report.device_local_memory_type_count;
  out << '}';
  out << ",\"device_extension_enumeration_ok\":"
      << (report.device_extension_enumeration_ok ? "true" : "false");
  out << ",\"tensor_features_available\":{";
  out << "\"queried\":"
      << (report.tensor_capabilities_queried ? "true" : "false");
  out << ",\"tensors\":";
  if (report.tensor_capabilities_queried) {
    out << (report.tensor_feature_tensors ? "true" : "false");
  } else {
    out << "null";
  }
  out << ",\"tensor_non_packed\":";
  if (report.tensor_capabilities_queried) {
    out << (report.tensor_feature_tensor_non_packed ? "true" : "false");
  } else {
    out << "null";
  }
  out << ",\"shader_tensor_access\":";
  if (report.tensor_capabilities_queried) {
    out << (report.tensor_feature_shader_tensor_access ? "true" : "false");
  } else {
    out << "null";
  }
  out << '}';
  out << ",\"tensor_limits\":{";
  out << "\"queried\":"
      << (report.tensor_capabilities_queried ? "true" : "false");
  if (report.tensor_capabilities_queried) {
    out << ",\"max_dimension_count\":" << report.max_tensor_dimension_count;
    out << ",\"max_elements\":" << report.max_tensor_elements;
    out << ",\"max_size\":" << report.max_tensor_size;
    out << ",\"max_stride\":" << report.max_tensor_stride;
  } else {
    out << ",\"max_dimension_count\":null";
    out << ",\"max_elements\":null";
    out << ",\"max_size\":null";
    out << ",\"max_stride\":null";
  }
  out << '}';
  out << ",\"external_memory_host\":{";
  out << "\"properties_queried\":"
      << (report.external_memory_host_properties_queried ? "true" : "false");
  out << ",\"min_imported_host_pointer_alignment\":";
  if (report.external_memory_host_properties_queried) {
    out << report.min_imported_host_pointer_alignment;
  } else {
    out << "null";
  }
  out << '}';
  out << ",\"io_policy\":{";
  out << "\"requires_host_visible\":"
      << (report.io_policy_requires_host_visible ? "true" : "false");
  out << ",\"requires_host_coherent\":"
      << (report.io_policy_requires_host_coherent ? "true" : "false");
  out << '}';
  out << ",\"model\":{";
  out << "\"input_count\":" << report.model_input_count;
  out << ",\"output_count\":" << report.model_output_count;
  out << ",\"io_count\":" << report.io_count;
  out << ",\"persistently_mapped_io_count\":"
      << report.persistently_mapped_io_count;
  out << ",\"unique_persistent_mapping_count\":"
      << report.unique_persistent_mapping_count;
  out << '}';
  out << ",\"mapping_contract\":{";
  out << "\"creation_phase\":" << json_escape(report.io_mapping_creation_phase);
  out << ",\"lifetime\":" << json_escape(report.io_mapping_lifetime);
  out << ",\"frame_slot_count\":" << report.frame_slot_count;
  out << ",\"multiple_in_flight_invocations_supported\":"
      << (report.multiple_in_flight_invocations_supported ? "true" : "false");
  out << '}';
  out << ",\"initialization_synchronization\":{";
  out << "\"image_layout_transition_count\":"
      << report.init_image_layout_transition_count;
  out << ",\"queue_submit_count\":"
      << report.init_image_layout_transition_count;
  out << ",\"fence_wait_count\":" << report.init_image_layout_transition_count;
  out << '}';
  out << ",\"extensions\":[";
  for (size_t i = 0; i < report.extensions.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto& extension = report.extensions[i];
    out << '{';
    out << "\"name\":" << json_escape(extension.name);
    out << ",\"available\":" << (extension.available ? "true" : "false");
    out << ",\"spec_version\":" << extension.spec_version;
    out << '}';
  }
  out << ']';
  out << ",\"ios\":[";
  for (size_t i = 0; i < report.ios.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto& io = report.ios[i];
    out << '{';
    out << "\"io_index\":" << io.io_index;
    out << ",\"direction\":" << json_escape(io.direction);
    out << ",\"descriptor_type\":" << json_escape(io.descriptor_type);
    out << ",\"vk_format\":" << io.vk_format;
    out << ",\"shape\":";
    write_integer_array(out, io.shape);
    out << ",\"strides\":";
    write_integer_array(out, io.strides);
    out << ",\"logical_bytes\":" << io.logical_bytes;
    out << ",\"mapped_memory_requirements\":{";
    out << "\"size\":" << io.memory_requirement_size;
    out << ",\"alignment\":" << io.memory_requirement_alignment;
    out << ",\"allocation_capacity\":" << io.memory_allocation_capacity;
    out << ",\"memory_type_bits\":" << io.memory_type_bits;
    out << ",\"memory_type_index\":" << io.memory_type_index;
    out << ",\"memory_property_flags\":" << io.memory_property_flags;
    out << ",\"dedicated_allocation_requirement\":"
        << json_escape(io.dedicated_allocation_requirement);
    out << '}';
    out << ",\"exact_resource_created\":"
        << (io.exact_resource_created ? "true" : "false");
    out << ",\"persistent_mapped\":"
        << (io.persistent_mapped ? "true" : "false");
    out << ",\"device_staging_copy\":"
        << (io.device_staging_copy ? "true" : "false");
    out << ",\"tensor_image_aliasing\":"
        << (io.tensor_image_aliasing ? "true" : "false");
    out << '}';
  }
  out << ']';
  out << '}';
  return out.str();
}

std::string serialize_vgf_transfer_observation(
    const VgfTransferObservation& observation) {
  std::ostringstream out;
  out << '{';
  out << "\"schema\":\"executorch.vgf.transfer_observation\"";
  out << ",\"schema_version\":1";
  out << ",\"kind\":" << json_escape(observation.kind);
  out << ",\"direction\":" << json_escape(observation.direction);
  out << ",\"slot\":" << observation.slot;
  out << ",\"io_index\":" << observation.io_index;
  out << ",\"bytes_copied\":" << observation.bytes_copied;
  out << ",\"transfer_mode\":" << json_escape(observation.transfer_mode);
  out << ",\"binding_provenance\":"
      << json_escape(observation.binding_provenance);
  out << ",\"descriptor_type\":" << json_escape(observation.descriptor_type);
  out << '}';
  return out.str();
}

std::string serialize_vgf_execution_report(const VgfExecutionReport& report) {
  std::ostringstream out;
  out << '{';
  out << "\"schema\":\"executorch.vgf.runtime_execution\"";
  out << ",\"schema_version\":1";
  out << ",\"method_instance_id\":" << report.method_instance_id;
  out << ",\"invocation\":" << report.invocation;
  out << ",\"success\":" << (report.success ? "true" : "false");
  out << ",\"synchronous_execution\":"
      << (report.synchronous_execution ? "true" : "false");
  out << ",\"cpu_boundary_copies\":{";
  out << "\"input_count\":" << report.input_cpu_copy_count;
  out << ",\"input_bytes\":" << report.input_cpu_copy_bytes;
  out << ",\"output_count\":" << report.output_cpu_copy_count;
  out << ",\"output_bytes\":" << report.output_cpu_copy_bytes;
  out << '}';
  out << ",\"device_staging_copies\":{";
  out << "\"input_count\":" << report.input_device_copy_count;
  out << ",\"input_bytes\":" << report.input_device_copy_bytes;
  out << ",\"output_count\":" << report.output_device_copy_count;
  out << ",\"output_bytes\":" << report.output_device_copy_bytes;
  out << '}';
  out << ",\"synchronization\":{";
  out << "\"queue_submit_count\":" << report.queue_submit_count;
  out << ",\"fence_wait_count\":" << report.fence_wait_count;
  out << ",\"host_to_device_barrier_count\":"
      << report.host_to_device_barrier_count;
  out << ",\"device_to_host_barrier_count\":"
      << report.device_to_host_barrier_count;
  out << ",\"segment_barrier_count\":" << report.segment_barrier_count;
  out << ",\"input_image_transfer_barrier_count\":"
      << report.input_image_transfer_barrier_count;
  out << ",\"output_image_transfer_barrier_count\":"
      << report.output_image_transfer_barrier_count;
  out << ",\"image_layout_transition_barrier_count\":"
      << report.image_layout_transition_barrier_count;
  out << ",\"explicit_flush_count\":" << report.explicit_flush_count;
  out << ",\"explicit_invalidate_count\":" << report.explicit_invalidate_count;
  out << '}';
  out << ",\"conversions\":{";
  out << "\"vgf_runtime_cpu_quantize_bytes\":"
      << report.vgf_runtime_cpu_quantize_bytes;
  out << ",\"vgf_runtime_cpu_dequantize_bytes\":"
      << report.vgf_runtime_cpu_dequantize_bytes;
  out << ",\"vgf_runtime_cpu_layout_conversion_bytes\":"
      << report.vgf_runtime_cpu_layout_conversion_bytes;
  out << ",\"portable_boundary_conversion_visibility\":"
      << json_escape(report.portable_boundary_conversion_visibility);
  out << ",\"vgf_internal_transfer_visibility\":"
      << json_escape(report.vgf_internal_transfer_visibility);
  out << '}';
  out << '}';
  return out.str();
}

bool write_vgf_diagnostics_report(
    const std::string& file_name,
    const std::string& json,
    std::string* written_path) {
  if (written_path != nullptr) {
    written_path->clear();
  }

  const char* root_env = std::getenv(kVgfDiagnosticsDirEnv);
  if (root_env == nullptr || root_env[0] == '\0') {
    return true;
  }

  const std::filesystem::path root(root_env);
  const uint64_t process_id = current_process_id_for_diagnostics();
  const uint64_t write_sequence =
      g_vgf_diagnostics_write_sequence.fetch_add(1, std::memory_order_relaxed);
  const std::filesystem::path final_path =
      process_unique_report_path(root, file_name, process_id);
  std::filesystem::path tmp_path = final_path;
  tmp_path += ".tmp." + std::to_string(write_sequence);

  std::error_code ec;
  std::filesystem::create_directories(final_path.parent_path(), ec);
  if (ec) {
    return false;
  }

  {
    std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
    if (!out) {
      return false;
    }
    out << json << '\n';
    out.close();
    if (!out) {
      std::filesystem::remove(tmp_path, ec);
      return false;
    }
  }

  // Never delete the previous valid report before replacement. If rename
  // fails, final_path is left untouched.
  ec.clear();
  std::filesystem::rename(tmp_path, final_path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    return false;
  }

  if (written_path != nullptr) {
    *written_path = final_path.string();
  }
  return true;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
