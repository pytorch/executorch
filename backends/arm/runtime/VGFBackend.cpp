/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

#include <c10/util/safe_numerics.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#endif

using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::toString;

#ifdef ET_EVENT_TRACER_ENABLED
using executorch::runtime::event_tracer_end_profiling_delegate;
using executorch::runtime::event_tracer_start_profiling_delegate;
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerEntry;
#endif

// We use the platform and runtime environment provided by the Vulkan delegate
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

// Dependencies for processing VGF files into Vulkan calls
#include <vgf/decoder.hpp>
#if __has_include(<vgf/version.h>)
#include <vgf/version.h>
#endif
#include <vgf/vulkan_helpers.generated.hpp>

#include <executorch/backends/arm/runtime/VGFDiagnostics.h>
#include <executorch/backends/arm/runtime/VGFSetup.h>

namespace executorch {
namespace backends {
namespace vgf {

/*
 * Simple function to populate function pointers for the relevant Tensor
 * and DataGraph extension APIs.
 */
VkResult vkml_load_extensions(VkDevice const* device) {
  // Note:
  //    We no longer PFN_vkCreateTensorARM)vkGetDeviceProcAddr(*device,
  //    "vkCreateTensorARM"); We just verify that the function pointers have
  //    been populated by the loader
  if (vkCreateTensorARM && vkDestroyTensorARM && vkCreateTensorViewARM &&
      vkDestroyTensorViewARM && vkGetTensorMemoryRequirementsARM &&
      vkBindTensorMemoryARM && vkCreateDataGraphPipelinesARM &&
      vkCmdDispatchDataGraphARM && vkCreateDataGraphPipelineSessionARM) {
    ET_LOG(Info, "VKML Extensions loaded");
    return VK_SUCCESS;
  }
  ET_LOG(Error, "Failed to load VKML extensions");
  return VK_ERROR_UNKNOWN;
}

/*
 * Fetch vulkan basic objects - intended to be replaced with a shared
 * device setup with the Vulkan backend.
 */
VkResult vkml_allocate_basics(
    VkInstance* instance,
    VkPhysicalDevice* physical_device,
    VkDevice* device,
    VkQueue* queue,
    VkCommandPool* command_pool,
    uint32_t* queue_family_index,
    bool request_neural_statistics,
    bool* neural_statistics_device_enabled);

// Helper functions to dump VGF Delegate Boundary Inputs
constexpr const char* kVgfDumpInputsDirEnv = "EXECUTORCH_VGF_DUMP_INPUTS_DIR";
constexpr const char* kVgfDumpInputsAndExitEnv =
    "EXECUTORCH_VGF_DUMP_INPUTS_AND_EXIT";
std::atomic<uint64_t> g_vgf_dump_invocation{0};
std::atomic<uint64_t> g_vgf_diagnostics_instance{0};
std::atomic<uint64_t> g_vgf_diagnostics_invocation{0};

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }

  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "False") != 0 && std::strcmp(value, "FALSE") != 0 &&
      std::strcmp(value, "off") != 0 && std::strcmp(value, "Off") != 0 &&
      std::strcmp(value, "OFF") != 0;
}

const char* descriptor_type_to_string(VkDescriptorType type) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_TENSOR_ARM:
      return "VK_DESCRIPTOR_TYPE_TENSOR_ARM";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    default:
      return "VK_DESCRIPTOR_TYPE_UNKNOWN";
  }
}

bool is_image_descriptor_type_for_diagnostics(VkDescriptorType type) {
  return type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
      type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
      type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
}

VgfCapabilityReport make_vgf_capability_report(
    VkPhysicalDevice physical_device,
    uint32_t queue_family_index,
    const VgfRepr& repr) {
  VgfCapabilityReport report;
  report.method_instance_id = repr.diagnostics_instance_id;

  const char* source_revision = std::getenv(kVgfDiagnosticsSourceRevisionEnv);
  if (source_revision != nullptr && source_revision[0] != '\0') {
    report.source_revision = source_revision;
  }

#if defined(MLSDK_VGF_LIBRARY_API_VERSION_MAJOR)
  report.mlsdk_vgf_api_major = MLSDK_VGF_LIBRARY_API_VERSION_MAJOR;
#endif
#if defined(MLSDK_VGF_LIBRARY_API_VERSION_MINOR)
  report.mlsdk_vgf_api_minor = MLSDK_VGF_LIBRARY_API_VERSION_MINOR;
#endif

  VkPhysicalDeviceProperties properties = {};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  report.vulkan_api_version = properties.apiVersion;
  report.driver_version = properties.driverVersion;
  report.vendor_id = properties.vendorID;
  report.device_id = properties.deviceID;
  report.device_name = properties.deviceName;
  report.queue_family_index = queue_family_index;
  report.non_coherent_atom_size = properties.limits.nonCoherentAtomSize;
  report.buffer_image_granularity = properties.limits.bufferImageGranularity;

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families.data());
  if (queue_family_index < queue_families.size()) {
    report.queue_flags = queue_families[queue_family_index].queueFlags;
  }

  VkPhysicalDeviceMemoryProperties memory_properties = {};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
  report.memory_type_count = memory_properties.memoryTypeCount;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    const auto flags = memory_properties.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      ++report.host_visible_memory_type_count;
    }
    if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
      ++report.host_coherent_memory_type_count;
    }
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
      ++report.device_local_memory_type_count;
    }
  }

  uint32_t extension_count = 0;
  VkResult extension_result = vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_extensions;
  if (extension_result == VK_SUCCESS) {
    available_extensions.resize(extension_count);
    extension_result = vkEnumerateDeviceExtensionProperties(
        physical_device,
        nullptr,
        &extension_count,
        available_extensions.data());
  }
  report.device_extension_enumeration_ok = extension_result == VK_SUCCESS;

  const char* relevant_extensions[] = {
      "VK_ARM_tensors",
      "VK_ARM_data_graph",
      "VK_KHR_maintenance4",
      "VK_KHR_maintenance5",
      "VK_KHR_deferred_host_operations",
      "VK_EXT_shader_replicated_composites",
      "VK_EXT_external_memory_host",
      "VK_KHR_external_memory",
      "VK_EXT_external_memory_dma_buf",
      "VK_ANDROID_external_memory_android_hardware_buffer",
  };
  for (const char* extension_name : relevant_extensions) {
    VgfExtensionCapability extension;
    extension.name = extension_name;
    if (extension_result == VK_SUCCESS) {
      auto it = std::find_if(
          available_extensions.begin(),
          available_extensions.end(),
          [&](const auto& available) {
            return std::strcmp(available.extensionName, extension_name) == 0;
          });
      if (it != available_extensions.end()) {
        extension.available = true;
        extension.spec_version = it->specVersion;
      }
    }
    report.extensions.push_back(std::move(extension));
  }

  const auto extension_available = [&](const char* name) {
    return std::any_of(
        report.extensions.begin(),
        report.extensions.end(),
        [&](const auto& extension) {
          return extension.name == name && extension.available;
        });
  };

  if (extension_available("VK_ARM_tensors")) {
    report.tensor_capabilities_queried = true;
    VkPhysicalDeviceTensorFeaturesARM tensor_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
        .pNext = nullptr,
    };
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &tensor_features,
    };
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    report.tensor_feature_tensors = tensor_features.tensors == VK_TRUE;
    report.tensor_feature_tensor_non_packed =
        tensor_features.tensorNonPacked == VK_TRUE;
    report.tensor_feature_shader_tensor_access =
        tensor_features.shaderTensorAccess == VK_TRUE;

    VkPhysicalDeviceTensorPropertiesARM tensor_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM,
        .pNext = nullptr,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &tensor_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    report.max_tensor_dimension_count =
        tensor_properties.maxTensorDimensionCount;
    report.max_tensor_elements = tensor_properties.maxTensorElements;
    report.max_tensor_size = tensor_properties.maxTensorSize;
    report.max_tensor_stride = tensor_properties.maxTensorStride;
  }

  if (extension_available("VK_EXT_external_memory_host")) {
    report.external_memory_host_properties_queried = true;
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT external_host_properties{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
        .pNext = nullptr,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &external_host_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    report.min_imported_host_pointer_alignment =
        external_host_properties.minImportedHostPointerAlignment;
  }

  report.model_input_count = repr.model_input_count;
  report.model_output_count = repr.model_output_count;
  report.io_count = repr.IOs.size();

  std::vector<VkDeviceMemory> unique_mapped_memories;
  report.ios.reserve(repr.IOs.size());
  for (size_t io_index = 0; io_index < repr.IOs.size(); ++io_index) {
    const auto& io = repr.IOs[io_index];
    VgfIoCapability io_report;
    io_report.io_index = io_index;
    io_report.direction = io.is_input ? "INPUT" : "OUTPUT";
    io_report.descriptor_type = descriptor_type_to_string(io.descriptor_type);
    io_report.vk_format = static_cast<uint32_t>(io.format);
    io_report.shape = io.size;
    io_report.strides = io.stride;
    io_report.logical_bytes = io.allocation_size;
    io_report.memory_requirement_size = io.memory_requirement_size;
    io_report.memory_requirement_alignment = io.memory_requirement_alignment;
    io_report.memory_allocation_capacity = io.memory_allocation_capacity;
    io_report.memory_type_bits = io.memory_type_bits;
    io_report.memory_type_index = io.memory_type_index;
    io_report.memory_property_flags = io.memory_property_flags;
    if (io.memory_dedicated_requirement_known) {
      if (io.memory_requires_dedicated_allocation) {
        io_report.dedicated_allocation_requirement = "required";
      } else if (io.memory_prefers_dedicated_allocation) {
        io_report.dedicated_allocation_requirement = "preferred";
      } else {
        io_report.dedicated_allocation_requirement = "not_required";
      }
    }
    io_report.exact_resource_created = true;
    io_report.persistent_mapped = io.persistent_memory != nullptr;
    io_report.device_staging_copy =
        is_image_descriptor_type_for_diagnostics(io.descriptor_type);
    io_report.tensor_image_aliasing = io.tensor_image_aliasing;
    report.ios.push_back(std::move(io_report));

    if (io.persistent_memory != nullptr) {
      ++report.persistently_mapped_io_count;
      if (std::find(
              unique_mapped_memories.begin(),
              unique_mapped_memories.end(),
              io.memory) == unique_mapped_memories.end()) {
        unique_mapped_memories.push_back(io.memory);
      }
    }
  }
  report.unique_persistent_mapping_count = unique_mapped_memories.size();
  report.init_image_layout_transition_count =
      std::count_if(repr.IOs.begin(), repr.IOs.end(), [](const auto& io) {
        return is_image_descriptor_type_for_diagnostics(io.descriptor_type);
      });
  report.init_image_layout_transition_count += std::count_if(
      repr.extra_allocs.begin(),
      repr.extra_allocs.end(),
      [](const auto& alloc) {
        return is_image_descriptor_type_for_diagnostics(alloc.descriptor_type);
      });

  return report;
}

void write_diagnostics_or_log(
    const std::string& file_name,
    const std::string& metadata) {
  std::string written_path;
  if (!write_vgf_diagnostics_report(file_name, metadata, &written_path)) {
    ET_LOG(
        Error, "Failed to write VGF diagnostics report %s", file_name.c_str());
    return;
  }
  if (!written_path.empty()) {
    ET_LOG(Info, "Wrote VGF diagnostics report to %s", written_path.c_str());
  }
}

#ifdef ET_EVENT_TRACER_ENABLED
void emit_vgf_metadata_event(
    EventTracer* event_tracer,
    const char* event_name,
    const std::string& metadata) {
  if (event_tracer == nullptr) {
    return;
  }
  EventTracerEntry event = event_tracer_start_profiling_delegate(
      event_tracer, event_name, /*delegate_debug_id=*/-1);
  event_tracer_end_profiling_delegate(
      event_tracer, event, metadata.data(), metadata.size());
}
#endif

template <typename T>
std::string array_ref_to_json(ArrayRef<T> values) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << static_cast<int64_t>(values[i]);
  }
  out << "]";
  return out.str();
}

template <typename T>
std::string vector_to_json(const std::vector<T>& values) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << static_cast<int64_t>(values[i]);
  }
  out << "]";
  return out.str();
}

bool write_binary_file(
    const std::filesystem::path& path,
    const void* data,
    size_t nbytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    return false;
  }

  file.write(
      static_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
  return file.good();
}

std::string make_vgf_call_dir_name(uint64_t invocation) {
  std::ostringstream call_name;
  call_name << "call_" << std::setw(6) << std::setfill('0') << invocation;
  return call_name.str();
}

std::filesystem::path make_vgf_tmp_call_dir(
    const std::filesystem::path& dump_root,
    const std::string& call_dir_name) {
  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();

  std::ostringstream tmp_name;
  tmp_name << "." << call_dir_name << ".tmp." << nonce;

  return dump_root / tmp_name.str();
}

class ScopedTmpDir {
 public:
  explicit ScopedTmpDir(std::filesystem::path path)
      : path_(std::move(path)), active_(true) {}

  ~ScopedTmpDir() {
    if (active_) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }

  ScopedTmpDir(const ScopedTmpDir&) = delete;
  ScopedTmpDir& operator=(const ScopedTmpDir&) = delete;

  void release() {
    active_ = false;
  }

 private:
  std::filesystem::path path_;
  bool active_;
};

Error dump_vgf_delegate_inputs(
    const VgfRepr& repr,
    Span<EValue*> args,
    const char* dump_root) {
  const uint64_t invocation = g_vgf_dump_invocation.fetch_add(1);

  const std::filesystem::path dump_root_path(dump_root);
  const std::string call_dir_name = make_vgf_call_dir_name(invocation);
  const std::filesystem::path final_call_dir = dump_root_path / call_dir_name;
  const std::filesystem::path tmp_call_dir =
      make_vgf_tmp_call_dir(dump_root_path, call_dir_name);

  std::error_code ec;

  std::filesystem::create_directories(dump_root_path, ec);
  if (ec) {
    ET_LOG(
        Error,
        "Failed to create VGF dump root %s: %s",
        dump_root_path.string().c_str(),
        ec.message().c_str());
    return Error::Internal;
  }

  if (std::filesystem::exists(final_call_dir, ec)) {
    ET_LOG(
        Error,
        "VGF dump output directory already exists: %s",
        final_call_dir.string().c_str());
    return Error::Internal;
  }

  std::filesystem::create_directory(tmp_call_dir, ec);
  if (ec) {
    ET_LOG(
        Error,
        "Failed to create temporary VGF dump directory %s: %s",
        tmp_call_dir.string().c_str(),
        ec.message().c_str());
    return Error::Internal;
  }

  ScopedTmpDir tmp_dir_guard(tmp_call_dir);

  std::vector<std::string> input_file_names;
  input_file_names.reserve(repr.model_input_count);

  for (size_t input_arg_idx = 0; input_arg_idx < repr.model_input_count;
       ++input_arg_idx) {
    const int io_idx = repr.model_input_io_index[input_arg_idx];

    if (io_idx < 0 || static_cast<size_t>(io_idx) >= repr.IOs.size()) {
      ET_LOG(
          Error,
          "Invalid VGF input IO index %d for input arg %zu",
          io_idx,
          input_arg_idx);
      return Error::InvalidArgument;
    }

    if (args[input_arg_idx] == nullptr || !args[input_arg_idx]->isTensor()) {
      ET_LOG(Error, "VGF input arg %zu is not a tensor", input_arg_idx);
      return Error::InvalidArgument;
    }

    const Tensor& tensor = args[input_arg_idx]->toTensor();
    const IO& io = repr.IOs[io_idx];

    if (tensor.nbytes() != io.allocation_size) {
      ET_LOG(
          Error,
          "VGF input arg %zu size mismatch: tensor nbytes=%zu, IO allocation_size=%zu",
          input_arg_idx,
          tensor.nbytes(),
          io.allocation_size);
      return Error::InvalidArgument;
    }

    std::ostringstream file_name;
    file_name << "input_" << std::setw(3) << std::setfill('0') << input_arg_idx
              << "_io_" << io_idx << ".bin";

    input_file_names.push_back(file_name.str());

    const std::filesystem::path input_path = tmp_call_dir / file_name.str();

    if (!write_binary_file(
            input_path, tensor.const_data_ptr(), tensor.nbytes())) {
      ET_LOG(
          Error,
          "Failed to write VGF input dump file %s",
          input_path.string().c_str());
      return Error::Internal;
    }
  }

  const std::filesystem::path metadata_path = tmp_call_dir / "metadata.json";
  {
    std::ofstream metadata(metadata_path, std::ios::out | std::ios::trunc);
    if (!metadata) {
      ET_LOG(
          Error,
          "Failed to open VGF metadata file %s",
          metadata_path.string().c_str());
      return Error::Internal;
    }

    metadata << "{\n";
    metadata << "  \"format_version\": 1,\n";
    metadata << "  \"invocation\": " << invocation << ",\n";
    metadata << "  \"input_count\": " << repr.model_input_count << ",\n";
    metadata << "  \"inputs\": [\n";

    for (size_t input_arg_idx = 0; input_arg_idx < repr.model_input_count;
         ++input_arg_idx) {
      const int io_idx = repr.model_input_io_index[input_arg_idx];
      const Tensor& tensor = args[input_arg_idx]->toTensor();
      const IO& io = repr.IOs[io_idx];

      metadata << "    {\n";
      metadata << "      \"arg_index\": " << input_arg_idx << ",\n";
      metadata << "      \"io_index\": " << io_idx << ",\n";
      metadata << "      \"file\": \"" << input_file_names[input_arg_idx]
               << "\",\n";
      metadata << "      \"nbytes\": " << tensor.nbytes() << ",\n";
      metadata << "      \"scalar_type\": \"" << toString(tensor.scalar_type())
               << "\",\n";
      metadata << "      \"tensor_shape\": "
               << array_ref_to_json(tensor.sizes()) << ",\n";
      metadata << "      \"tensor_strides\": "
               << array_ref_to_json(tensor.strides()) << ",\n";
      metadata << "      \"io_shape\": " << vector_to_json(io.size) << ",\n";
      metadata << "      \"io_strides\": " << vector_to_json(io.stride)
               << ",\n";
      metadata << "      \"io_element_size\": " << io.elt_size << ",\n";
      metadata << "      \"io_allocation_size\": " << io.allocation_size
               << ",\n";
      metadata << "      \"io_descriptor_type\": \""
               << descriptor_type_to_string(io.descriptor_type) << "\"\n";
      metadata << "    }";

      if (input_arg_idx + 1 < repr.model_input_count) {
        metadata << ",";
      }
      metadata << "\n";
    }

    metadata << "  ]\n";
    metadata << "}\n";

    metadata.close();
    if (!metadata) {
      ET_LOG(
          Error,
          "Failed to write VGF metadata file %s",
          metadata_path.string().c_str());
      return Error::Internal;
    }
  }

  std::filesystem::rename(tmp_call_dir, final_call_dir, ec);
  if (ec) {
    ET_LOG(
        Error,
        "Failed to publish VGF dump directory %s -> %s: %s",
        tmp_call_dir.string().c_str(),
        final_call_dir.string().c_str(),
        ec.message().c_str());
    return Error::Internal;
  }

  tmp_dir_guard.release();

  ET_LOG(
      Info,
      "Wrote VGF delegate input dump to %s",
      final_call_dir.string().c_str());

  return Error::Ok;
}

class VGFBackend final : public ::executorch::runtime::BackendInterface {
 public:
  VGFBackend() = default;

  // Lazy Vulkan init — runs on first use, not in the constructor.
  void ensure_initialized() {
    if (is_initialized_) {
      return;
    }

    VkResult result;
    neural_statistics_config_ = get_vgf_neural_statistics_runtime_config();

    // Fetch basic vulkan objects once
    result = vkml_allocate_basics(
        &vk_instance,
        &vk_physical_device,
        &vk_device,
        &vk_queue,
        &vk_command_pool,
        &vk_queue_family_index,
        neural_statistics_config_.requested,
        &neural_statistics_device_enabled_);
    if (result != VK_SUCCESS) {
      ET_LOG(
          Error, "Failed to initialize the Vulkan device error 0x%08X", result);
      return;
    }

    // Query the device to ensure it has needed extensions
    result = vkml_load_extensions(&vk_device);
    if (result != VK_SUCCESS) {
      ET_LOG(
          Error,
          "Failed to verify VKML extensions needed, error 0x%08X",
          result);
      return;
    }

    is_initialized_ = true;
  }

  ~VGFBackend() = default;

  bool is_available() const override {
    ET_LOG(Info, "Checking VGFBackend is available");
    const_cast<VGFBackend*>(this)->ensure_initialized();
    if (!is_initialized_) {
      return false;
    }
    return vkml_load_extensions(&vk_device) == VK_SUCCESS;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ET_LOG(Info, "Entered VGF init");

#ifdef ET_EVENT_TRACER_ENABLED
    EventTracer* event_tracer = context.event_tracer();

    EventTracerEntry init_total_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_INIT_TOTAL",
        /*delegate_debug_id=*/-1);

    EventTracerEntry ensure_initialized_event =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "VGF_INIT_ENSURE_INITIALIZED",
            /*delegate_debug_id=*/-1);
#endif

    const_cast<VGFBackend*>(this)->ensure_initialized();

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, ensure_initialized_event);
#endif

    if (!is_initialized_) {
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(event_tracer, init_total_event);
#endif
      ET_LOG(
          Error,
          "VGF backend is unavailable because Vulkan initialization failed");
      return Error::NotSupported;
    }

    const char* vgf_data = reinterpret_cast<const char*>(processed->data());

#ifdef ET_EVENT_TRACER_ENABLED
    EventTracerEntry allocate_repr_event =
        event_tracer_start_profiling_delegate(
            event_tracer,
            "VGF_INIT_ALLOCATE_REPR",
            /*delegate_debug_id=*/-1);
#endif

    MemoryAllocator* allocator = context.get_runtime_allocator();
    VgfRepr* repr = allocator->allocateInstance<VgfRepr>();
    new (repr) VgfRepr(
        vk_instance,
        vk_physical_device,
        vk_device,
        vk_queue,
        vk_command_pool,
        vk_queue_family_index,
        neural_statistics_config_.requested,
        neural_statistics_device_enabled_,
        neural_statistics_config_.mode_index);

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, allocate_repr_event);

    EventTracerEntry process_vgf_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_INIT_PROCESS_VGF_BACKEND",
        /*delegate_debug_id=*/-1);
#endif

#ifdef ET_EVENT_TRACER_ENABLED
    auto valid_vgf = repr->process_vgf(
        vgf_data, processed->size(), compile_specs, event_tracer);
#else
    auto valid_vgf =
        repr->process_vgf(vgf_data, processed->size(), compile_specs);
#endif

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, process_vgf_event);
#endif

    if (!valid_vgf) {
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(event_tracer, init_total_event);
#endif
      ET_LOG(Error, "Failed to process VGF blob.");
      return Error::Internal;
    }

    repr->diagnostics_instance_id = g_vgf_diagnostics_instance.fetch_add(1);
    const VgfCapabilityReport capability_report = make_vgf_capability_report(
        vk_physical_device, vk_queue_family_index, *repr);
    const std::string capability_metadata =
        serialize_vgf_capability_report(capability_report);
    std::ostringstream capability_file_name;
    capability_file_name << "capabilities_" << std::setw(6) << std::setfill('0')
                         << repr->diagnostics_instance_id << ".json";
    write_diagnostics_or_log(capability_file_name.str(), capability_metadata);
#ifdef ET_EVENT_TRACER_ENABLED
    emit_vgf_metadata_event(
        event_tracer,
        kVgfDiagnosticsCapabilitiesEventName,
        capability_metadata);
    event_tracer_end_profiling_delegate(event_tracer, init_total_event);
#endif

    return repr;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    VgfRepr* repr = static_cast<VgfRepr*>(handle);
    const size_t input_count = repr->model_input_count;
    const size_t output_count = repr->model_output_count;
    ET_LOG(
        Info,
        "VGF execute: args=%zu IOs=%zu inputs=%zu outputs=%zu",
        args.size(),
        repr->IOs.size(),
        input_count,
        output_count);
    if (args.size() < input_count + output_count) {
      ET_LOG(Error, "Insufficient args for IOs");
      return Error::InvalidArgument;
    }

    // Helper block to dump VGF delegate boundary inputs for testing
    // with scenari0 runner
    const char* dump_inputs_dir = std::getenv(kVgfDumpInputsDirEnv);
    if (dump_inputs_dir != nullptr && dump_inputs_dir[0] != '\0') {
      Error dump_status =
          dump_vgf_delegate_inputs(*repr, args, dump_inputs_dir);
      if (dump_status != Error::Ok) {
        return dump_status;
      }

      if (env_flag_enabled(kVgfDumpInputsAndExitEnv)) {
        ET_LOG(
            Info,
            "Exiting after VGF delegate input dump because %s is set",
            kVgfDumpInputsAndExitEnv);
        return Error::EndOfMethod;
      }
    }

#ifdef ET_EVENT_TRACER_ENABLED
    EventTracer* event_tracer = context.event_tracer();

    EventTracerEntry vgf_execute_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_EXECUTE",
        /*delegate_debug_id=*/-1);

    EventTracerEntry copy_inputs_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_COPY_INPUTS",
        /*delegate_debug_id=*/-1);
#else
    (void)context;
#endif

    VgfExecutionReport execution_report;
    execution_report.method_instance_id = repr->diagnostics_instance_id;
    execution_report.invocation = g_vgf_diagnostics_invocation.fetch_add(1);
    execution_report.host_to_device_barrier_count = 1;
    execution_report.device_to_host_barrier_count = 1;
    execution_report.segment_barrier_count =
        repr->segments.empty() ? 0 : repr->segments.size() - 1;

    for (const auto& io : repr->IOs) {
      if (!is_image_descriptor_type_for_diagnostics(io.descriptor_type)) {
        continue;
      }
      if (io.is_input) {
        ++execution_report.input_device_copy_count;
        execution_report.input_device_copy_bytes += io.allocation_size;
      } else {
        ++execution_report.output_device_copy_count;
        execution_report.output_device_copy_bytes += io.allocation_size;
      }
    }
    execution_report.input_image_transfer_barrier_count =
        execution_report.input_device_copy_count == 0 ? 0 : 1;
    execution_report.output_image_transfer_barrier_count =
        execution_report.output_device_copy_count == 0 ? 0 : 1;
    execution_report.image_layout_transition_barrier_count =
        repr->execute_image_layout_transition_barrier_count;

    // Current mapped IO allocation policy requires HOST_COHERENT memory, so
    // no explicit vkFlushMappedMemoryRanges/vkInvalidateMappedMemoryRanges are
    // issued by the VGF runtime. Portable/AoT conversions outside the delegate
    // remain intentionally reported as unknown.

    // Copy all inputs from EValue to VkDeviceMemory
    for (size_t input_arg_idx = 0; input_arg_idx < input_count;
         ++input_arg_idx) {
      const int io_idx = repr->model_input_io_index[input_arg_idx];
      if (io_idx < 0) {
        ET_LOG(Info, "Skipping eliminated VGF input %zu", input_arg_idx);
        // See test_addmm_vgf_no_quant[beta_only]
        // two inputs are eliminated from the graph by the converter
        continue;
      }
      if (!args[input_arg_idx]->isTensor()) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_inputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(
            Error,
            "Expected input EValue %zu to be tensor, got %d",
            input_arg_idx,
            static_cast<uint32_t>(args[input_arg_idx]->tag));
        return Error::InvalidArgument;
      }

      Tensor* tensor = &args[input_arg_idx]->toTensor();
      IO* io = &repr->IOs[io_idx];

      ET_LOG(Info, "Copy input IO[%d] -> args[%zu]", io_idx, input_arg_idx);
      size_t io_size = tensor->nbytes();
      if (io_size != io->allocation_size) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_inputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(
            Error,
            "Input tensor byte size %zu does not match IO allocation %zu",
            io_size,
            io->allocation_size);
        return Error::InvalidArgument;
      }

      void* data;
      if (!repr->map_io(io, &data)) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_inputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(Error, "Failed to map Vulkan IO memory");
        return Error::Internal;
      }
#ifdef ET_EVENT_TRACER_ENABLED
      VgfTransferObservation input_observation;
      input_observation.kind = "INPUT_COPY";
      input_observation.direction = "INPUT";
      input_observation.slot = input_arg_idx;
      input_observation.io_index = io_idx;
      input_observation.bytes_copied = io_size;
      input_observation.descriptor_type =
          descriptor_type_to_string(io->descriptor_type);
      const std::string input_copy_metadata =
          serialize_vgf_transfer_observation(input_observation);
      EventTracerEntry input_copy_event = event_tracer_start_profiling_delegate(
          event_tracer,
          kVgfDiagnosticsInputCopyEventName,
          /*delegate_debug_id=*/-1);
#endif
      memcpy(data, tensor->mutable_data_ptr(), io_size);
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(
          event_tracer,
          input_copy_event,
          input_copy_metadata.data(),
          input_copy_metadata.size());
#endif
      ++execution_report.input_cpu_copy_count;
      execution_report.input_cpu_copy_bytes += io_size;
      repr->unmap_io(io);
    }

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, copy_inputs_event);

    EventTracerEntry dispatch_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_DISPATCH_AND_WAIT",
        /*delegate_debug_id=*/-1);
#endif

    // Execute the workload
    bool execute_ok = false;
    repr->execution_queue_submit_count = 0;
    repr->execution_fence_wait_count = 0;

#ifdef ET_EVENT_TRACER_ENABLED
    execute_ok = repr->execute_vgf(event_tracer);
#else
    execute_ok = repr->execute_vgf();
#endif

    execution_report.queue_submit_count = repr->execution_queue_submit_count;
    execution_report.fence_wait_count = repr->execution_fence_wait_count;

    if (!execute_ok) {
      std::ostringstream diagnostics_file_name;
      diagnostics_file_name << "execution_" << std::setw(6) << std::setfill('0')
                            << execution_report.method_instance_id << "_"
                            << std::setw(6) << execution_report.invocation
                            << ".json";
      write_diagnostics_or_log(
          diagnostics_file_name.str(),
          serialize_vgf_execution_report(execution_report));
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(event_tracer, dispatch_event);
      event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
      ET_LOG(Error, "Failed to execute the VGF representation");
      return Error::Internal;
    }

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, dispatch_event);

    if (event_tracer != nullptr && repr->neural_statistics_requested()) {
      // We attach the neural statistics JSON blob to ETDump as delegate
      // metadata, when event tracer is active.

      // This is “synthetic” event, which we use as a carrier for metadata
      EventTracerEntry neural_statistics_event =
          event_tracer_start_profiling_delegate(
              event_tracer,
              kVgfNeuralStatisticsDelegateEventName,
              /*delegate_debug_id=*/-1);

      // Ask VGF representation for neural accelerator diagnostics
      std::string neural_statistics_metadata =
          repr->collect_neural_statistics_metadata();

      event_tracer_end_profiling_delegate(
          event_tracer,
          neural_statistics_event,
          neural_statistics_metadata.data(),
          neural_statistics_metadata.size());
    }

    EventTracerEntry copy_outputs_event = event_tracer_start_profiling_delegate(
        event_tracer,
        "VGF_COPY_OUTPUTS",
        /*delegate_debug_id=*/-1);
#endif

    // Copy all outputs from VKDeviceMemory to EValue
    for (size_t output_rel_idx = 0; output_rel_idx < output_count;
         ++output_rel_idx) {
      const size_t output_arg_idx = input_count + output_rel_idx;
      const int io_idx = repr->model_output_io_index[output_rel_idx];
      if (io_idx < 0) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(Error, "Missing IO mapping for output %zu", output_rel_idx);
        return Error::InvalidArgument;
      }
      if (!args[output_arg_idx]->isTensor()) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(
            Error,
            "Expected output EValue %zu to be tensor, got %d",
            output_arg_idx,
            static_cast<uint32_t>(args[output_arg_idx]->tag));
        return Error::InvalidArgument;
      }
      Tensor* tensor = &args[output_arg_idx]->toTensor();
      IO* io = &repr->IOs[io_idx];

      ET_LOG(Info, "Copy output IO[%d] -> args[%zu]", io_idx, output_arg_idx);
      size_t io_size = tensor->nbytes();
      if (io_size != io->allocation_size) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(
            Error,
            "Output tensor byte size %zu does not match IO allocation %zu",
            io_size,
            io->allocation_size);
        return Error::InvalidArgument;
      }

      void* data;
      if (!repr->map_io(io, &data)) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(Error, "Failed to map Vulkan IO memory");
        return Error::Internal;
      }
#ifdef ET_EVENT_TRACER_ENABLED
      VgfTransferObservation output_observation;
      output_observation.kind = "OUTPUT_COPY";
      output_observation.direction = "OUTPUT";
      output_observation.slot = output_rel_idx;
      output_observation.io_index = io_idx;
      output_observation.bytes_copied = io_size;
      output_observation.descriptor_type =
          descriptor_type_to_string(io->descriptor_type);
      const std::string output_copy_metadata =
          serialize_vgf_transfer_observation(output_observation);
      EventTracerEntry output_copy_event =
          event_tracer_start_profiling_delegate(
              event_tracer,
              kVgfDiagnosticsOutputCopyEventName,
              /*delegate_debug_id=*/-1);
#endif
      memcpy(tensor->mutable_data_ptr(), data, io_size);
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(
          event_tracer,
          output_copy_event,
          output_copy_metadata.data(),
          output_copy_metadata.size());
#endif
      ++execution_report.output_cpu_copy_count;
      execution_report.output_cpu_copy_bytes += io_size;
      repr->unmap_io(io);
    }

    execution_report.success = true;
    const std::string execution_metadata =
        serialize_vgf_execution_report(execution_report);
    std::ostringstream diagnostics_file_name;
    diagnostics_file_name << "execution_" << std::setw(6) << std::setfill('0')
                          << execution_report.method_instance_id << "_"
                          << std::setw(6) << execution_report.invocation
                          << ".json";
    write_diagnostics_or_log(diagnostics_file_name.str(), execution_metadata);

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
    emit_vgf_metadata_event(
        event_tracer, kVgfDiagnosticsExecutionEventName, execution_metadata);
    event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    VgfRepr* repr = static_cast<VgfRepr*>(handle);
    repr->~VgfRepr();
  }

 private:
  VkInstance vk_instance = VK_NULL_HANDLE;
  VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
  VkDevice vk_device = VK_NULL_HANDLE;
  VkQueue vk_queue = VK_NULL_HANDLE;
  VkCommandPool vk_command_pool = VK_NULL_HANDLE;
  uint32_t vk_queue_family_index = UINT32_MAX;
  VgfNeuralStatisticsRuntimeConfig neural_statistics_config_{};
  bool neural_statistics_device_enabled_ = false;
  bool is_initialized_ = false;
};

namespace {
auto cls = VGFBackend();
Backend backend{"VgfBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

VkResult vkml_allocate_basics(
    VkInstance* instance,
    VkPhysicalDevice* physical_device,
    VkDevice* device,
    VkQueue* queue,
    VkCommandPool* command_pool,
    uint32_t* queue_family_index,
    bool request_neural_statistics,
    bool* neural_statistics_device_enabled) {
  VkResult result;

  if (neural_statistics_device_enabled != nullptr) {
    *neural_statistics_device_enabled = false;
  }

  if (VK_SUCCESS != volkInitialize()) {
    ET_LOG(Error, "Volk failed to initialize");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkApplicationInfo app_info{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = "VGF",
      .applicationVersion = 0,
      .pEngineName = "executorch",
      .engineVersion = 0,
      .apiVersion = VK_API_VERSION_1_3,
  };

  std::vector<const char*> requested_extensions;
  VkInstanceCreateFlags instance_flags = 0;

#ifdef __APPLE__
  instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

  uint32_t extension_count = 0;
  result = vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, nullptr);

  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to enumerate instance extensions");
    return result;
  }

  std::vector<VkExtensionProperties> extension_properties(extension_count);
  result = vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, extension_properties.data());

  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to enumerate instance extensions");
    return result;
  }

  if (std::any_of(
          extension_properties.begin(),
          extension_properties.end(),
          [](const auto& extension) {
            return strcmp(
                       VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
                       extension.extensionName) == 0;
          })) {
    requested_extensions.push_back(
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  }

  if (requested_extensions.empty()) {
    ET_LOG(Error, "VK_KHR_portability_enumeration not found");
  }

#endif

  VkInstanceCreateInfo instance_info{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = instance_flags,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount =
          static_cast<uint32_t>(requested_extensions.size()),
      .ppEnabledExtensionNames = requested_extensions.data(),
  };
  result = vkCreateInstance(&instance_info, nullptr, instance);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create VkInstance");
    return result;
  }
  volkLoadInstance(*instance);

  // Bail out if the driver lacks ARM tensor/datagraph extensions.
  if (!vkCreateTensorARM) {
    ET_LOG(
        Error,
        "Vulkan driver does not support ARM tensor extensions (VK_ARM_tensors)");
    vkDestroyInstance(*instance, nullptr);
    *instance = VK_NULL_HANDLE;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  // Pick first GPU
  uint32_t gpu_count = 0;
  vkEnumeratePhysicalDevices(*instance, &gpu_count, nullptr);
  if (gpu_count == 0) {
    ET_LOG(Error, "Found no suitable devices");
    return VK_ERROR_UNKNOWN;
  }
  vector<VkPhysicalDevice> gpus(gpu_count);
  result = vkEnumeratePhysicalDevices(*instance, &gpu_count, gpus.data());
  *physical_device = gpus[0];
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to select physical device");
    return result;
  }

  // Find suitable queue family
  uint32_t qf_count;
  vkGetPhysicalDeviceQueueFamilyProperties(
      *physical_device, &qf_count, nullptr);
  vector<VkQueueFamilyProperties> qps(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      *physical_device, &qf_count, qps.data());
  uint32_t qf = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; ++i) {
    if (qps[i].queueFlags &
        (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_DATA_GRAPH_BIT_ARM)) {
      qf = i;
      break;
    }
  }
  if (qf == UINT32_MAX) {
    ET_LOG(Error, "Failed to find suitable queue");
    return VK_ERROR_UNKNOWN;
  }
  if (queue_family_index != nullptr) {
    *queue_family_index = qf;
  }

  // Device with ML tensor extension
  float qp = 1.0f;
  VkDeviceQueueCreateInfo queue_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &qp,
  };

  // Query features
  VkPhysicalDeviceVulkan12Features available_12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = NULL,
  };
  VkPhysicalDeviceVulkan11Features available_11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &available_12,
  };
#if defined(VK_ARM_data_graph_neural_accelerator_statistics)
  VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM
      available_neural_statistics{
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
          .pNext = &available_11,
          .dataGraphNeuralAcceleratorStatistics = VK_FALSE,
      };
  void* available_features_pnext = &available_neural_statistics;
#else
  void* available_features_pnext = &available_11;
#endif
  VkPhysicalDeviceFeatures2 available_2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = available_features_pnext,
  };
  vkGetPhysicalDeviceFeatures2(*physical_device, &available_2);

  // Select features
  VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT features_c{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT,
      nullptr};
  features_c.shaderReplicatedComposites = true;
  VkPhysicalDeviceVulkan13Features features_13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, nullptr};
  features_13.synchronization2 = true;
  features_13.maintenance4 = true;
  features_13.pipelineCreationCacheControl = true;
  features_13.pNext = &features_c;
  VkPhysicalDeviceVulkan12Features features_12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, nullptr};
  features_12.hostQueryReset = true;
  features_12.storageBuffer8BitAccess = true;
  features_12.uniformAndStorageBuffer8BitAccess =
      available_12.uniformAndStorageBuffer8BitAccess;
  features_12.shaderInt8 = true;
  features_12.shaderFloat16 = available_12.shaderFloat16;
  features_12.vulkanMemoryModel = true;
  features_12.vulkanMemoryModelDeviceScope =
      available_12.vulkanMemoryModelDeviceScope;
  features_12.pNext = &features_13;
  VkPhysicalDeviceVulkan11Features features_11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr};
  features_11.storageBuffer16BitAccess = available_11.storageBuffer16BitAccess;
  features_11.uniformAndStorageBuffer16BitAccess =
      available_11.uniformAndStorageBuffer16BitAccess;
  features_11.pNext = &features_12;
  VkPhysicalDeviceTensorFeaturesARM features_tensor{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM, nullptr};
  features_tensor.shaderTensorAccess = true;
  features_tensor.tensors = true;
  features_tensor.pNext = &features_11;
  VkPhysicalDeviceDataGraphFeaturesARM features_graph{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM, nullptr};
  features_graph.dataGraph = true;
  features_graph.pNext = &features_tensor;
#if defined(VK_ARM_data_graph_neural_accelerator_statistics)
  VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM
      features_neural_statistics{
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
          .pNext = &features_tensor,
          .dataGraphNeuralAcceleratorStatistics = VK_FALSE,
      };
#endif

  VkPhysicalDeviceFeatures device_features = {};
  device_features.shaderInt16 = VK_TRUE;
  device_features.shaderInt64 = VK_TRUE;

  // Extension strings to enable
  auto dev_exts = {
      "VK_ARM_tensors",
      "VK_ARM_data_graph",
      "VK_KHR_maintenance4",
      "VK_KHR_maintenance5",
      "VK_KHR_deferred_host_operations",
      "VK_EXT_shader_replicated_composites"};

  uint32_t exts = 0;
  vkEnumerateDeviceExtensionProperties(
      *physical_device, nullptr, &exts, nullptr);
  vector<VkExtensionProperties> available(exts);
  vkEnumerateDeviceExtensionProperties(
      *physical_device, nullptr, &exts, available.data());

  vector<const char*> requested_exts;

#if defined(VK_ARM_data_graph_neural_accelerator_statistics)
  const bool neural_statistics_extension_available = std::any_of(
      available.begin(), available.end(), [](const auto& ext_avail) {
        return std::strcmp(
                   VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME,
                   ext_avail.extensionName) == 0;
      });
  const bool neural_statistics_feature_available =
      available_neural_statistics.dataGraphNeuralAcceleratorStatistics ==
      VK_TRUE;
  const bool enable_neural_statistics_device = request_neural_statistics &&
      neural_statistics_extension_available &&
      neural_statistics_feature_available;

  if (request_neural_statistics && !neural_statistics_extension_available) {
    ET_LOG(
        Info,
        "%s was requested but the Vulkan device does not expose %s",
        kVgfNeuralStatisticsEnableEnv,
        VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME);
  } else if (
      request_neural_statistics && !neural_statistics_feature_available) {
    ET_LOG(
        Info,
        "%s was requested but dataGraphNeuralAcceleratorStatistics is not supported",
        kVgfNeuralStatisticsEnableEnv);
  }

  if (enable_neural_statistics_device) {
    requested_exts.push_back(
        VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME);
    features_neural_statistics.dataGraphNeuralAcceleratorStatistics = VK_TRUE;
    features_graph.pNext = &features_neural_statistics;
    if (neural_statistics_device_enabled != nullptr) {
      *neural_statistics_device_enabled = true;
    }
  }
#else
  if (request_neural_statistics) {
    ET_LOG(
        Info,
        "%s was requested but Vulkan headers do not expose "
        "VK_ARM_data_graph_neural_accelerator_statistics",
        kVgfNeuralStatisticsEnableEnv);
  }
#endif

  for (auto& ext : dev_exts) {
    bool found = false;
    for (auto const& ext_avail : available) {
      if (strcmp(ext, ext_avail.extensionName) == 0) {
        found = true;
        requested_exts.push_back(ext);
      }
    }
    if (found == false) {
      ET_LOG(Info, "Failed to find extension %s", ext);
    }
  }

  // Create the device with our subset of features
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &queue_info;
  dci.enabledExtensionCount = requested_exts.size();
  dci.ppEnabledExtensionNames = requested_exts.data();
  ;
  dci.pEnabledFeatures = &device_features;
  dci.pNext = &features_graph;
  result = vkCreateDevice(*physical_device, &dci, nullptr, device);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create VkDevice");
    return result;
  }
  // Load the device with volk and populate function pointers
  volkLoadDevice(*device);

  vkGetDeviceQueue(*device, qf, 0, queue);

  VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = qf,
  };
  result = vkCreateCommandPool(*device, &poolInfo, nullptr, command_pool);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create VkCommandPool");
    return result;
  }

  return result;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
