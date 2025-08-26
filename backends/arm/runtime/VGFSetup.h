/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string>
#include <vector>
using namespace std;

#include <executorch/runtime/backend/interface.h>

using executorch::runtime::ArrayRef;
using executorch::runtime::CompileSpec;

// We use the platform and runtime environment provided by the Vulkan delegate
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

namespace executorch {
namespace backends {
namespace vgf {

class VgfRepr;

/*
 * Info about IOs used during execution
 */
typedef struct IO {
  vector<int64_t> size;
  vector<int64_t> stride;
  size_t elt_size;
  VkTensorARM tensor;
  VkTensorViewARM tensor_view;
  VkDeviceMemory tensor_memory;
  bool is_input;
} IO;

/*
 * In memory, and in-vulkan-object representation of the loaded
 * VGF graph - ready to be dispatched based on provided inputs.
 */
class VgfRepr {
 public:
  VgfRepr(
      VkInstance inst,
      VkPhysicalDevice phys,
      VkDevice dev,
      VkQueue queue,
      VkCommandPool pool)
      : vk_instance(inst),
        vk_physical(phys),
        vk_device(dev),
        vk_queue(queue),
        vk_command_pool(pool) {}

  /*
   * Process a VGF ready for execution, allocate necessary Vulkan objects.
   */
  bool process_vgf(const char* vgf_data, ArrayRef<CompileSpec> specs);

  /*
   * Execute the VGF we've previously processed.
   */
  bool execute_vgf();

  /*
   * Free any allocations made in process_vgf.
   */
  void free_vgf();

  /*
   * input and outputs from the VGF - these are memory mapped and populated
   * with the EValues coming the backend execute call
   */
  vector<IO> IOs;
  vector<VkDeviceMemory> intermediates;

  bool map_io(IO* io, void** handle) {
    VkResult result =
        vkMapMemory(vk_device, io->tensor_memory, 0, VK_WHOLE_SIZE, 0, handle);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to map Vulkan IO memory");
      return false;
    }
    return true;
  }

  void unmap_io(IO* io) {
    vkUnmapMemory(vk_device, io->tensor_memory);
  }

  ~VgfRepr() {
    free_vgf();
  }

 private:
  // Basic Vulkan objects passed to us and re-used
  VkInstance vk_instance;
  VkPhysicalDevice vk_physical;
  VkDevice vk_device;
  VkQueue vk_queue;
  VkCommandPool vk_command_pool;

  // per-VgfRepr-instance objects allocated in process_vgf, used (can be more
  // than once) in execute_vgf
  VkCommandBuffer vk_execute_cmd = VK_NULL_HANDLE;
  VkDataGraphPipelineSessionARM vk_session = VK_NULL_HANDLE;
  VkPipeline vk_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool vk_descriptor_pool;
  VkDescriptorSetLayout vk_layout;
  VkShaderModule vk_shader;
  // Note: the vector of tensor memory is stored in IOs above
  vector<VkDescriptorSet> descriptor_sets;
};

} // namespace vgf
} // namespace backends
} // namespace executorch
