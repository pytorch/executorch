/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
#include <vgf/vulkan_helpers.generated.hpp>

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
    uint32_t* queue_family_index);

void vkml_free_basics(
    VkInstance* instance,
    VkDevice* device,
    VkCommandPool* command_pool) {
  if (*device != VK_NULL_HANDLE && *command_pool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(*device, *command_pool, nullptr);
  }
  // Note: These primitives are used by the emulation layer for vulkan
  //       object allocation, the vulkan objects are freed in in library
  //       shutdown, so we can't yet destroy these here without causing
  //       a crash there.
  //  vkDestroyDevice(*device, nullptr);
  //  vkDestroyInstance(*instance, nullptr);
}

// Helper functions to dump VGF Delegate Boundary Inputs
constexpr const char* kVgfDumpInputsDirEnv = "EXECUTORCH_VGF_DUMP_INPUTS_DIR";
constexpr const char* kVgfDumpInputsAndExitEnv =
    "EXECUTORCH_VGF_DUMP_INPUTS_AND_EXIT";

std::atomic<uint64_t> g_vgf_dump_invocation{0};

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
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

    // Fetch basic vulkan objects once
    result = vkml_allocate_basics(
        &vk_instance,
        &vk_physical_device,
        &vk_device,
        &vk_queue,
        &vk_command_pool,
        &vk_queue_family_index);
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
  ~VGFBackend() {
    vkml_free_basics(&vk_instance, &vk_device, &vk_command_pool);
  }

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
        vk_queue_family_index);

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

#ifdef ET_EVENT_TRACER_ENABLED
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

    // Copy all inputs from EValue to VkDeviceMemory
    for (size_t input_arg_idx = 0; input_arg_idx < input_count;
         ++input_arg_idx) {
      const int io_idx = repr->model_input_io_index[input_arg_idx];
      if (io_idx < 0) {
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_end_profiling_delegate(event_tracer, copy_inputs_event);
        event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
        ET_LOG(Error, "Missing IO mapping for input %zu", input_arg_idx);
        return Error::InvalidArgument;
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
      memcpy(data, tensor->mutable_data_ptr(), io_size);
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
#ifdef ET_EVENT_TRACER_ENABLED
    execute_ok = repr->execute_vgf(event_tracer);
#else
    execute_ok = repr->execute_vgf();
#endif

    if (!execute_ok) {
#ifdef ET_EVENT_TRACER_ENABLED
      event_tracer_end_profiling_delegate(event_tracer, dispatch_event);
      event_tracer_end_profiling_delegate(event_tracer, vgf_execute_event);
#endif
      ET_LOG(Error, "Failed to execute the VGF representation");
      return Error::Internal;
    }

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, dispatch_event);

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
      memcpy(tensor->mutable_data_ptr(), data, io_size);
      repr->unmap_io(io);
    }

#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_end_profiling_delegate(event_tracer, copy_outputs_event);
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
    uint32_t* queue_family_index) {
  VkResult result;

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
  VkPhysicalDeviceFeatures2 available_2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_11,
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