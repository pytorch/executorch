/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/event_tracer.h>

using executorch::runtime::ArrayRef;
using executorch::runtime::CompileSpec;

// We use the platform and runtime environment provided by the Vulkan delegate
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/arm/runtime/VGFNeuralStatistics.h>

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
  size_t allocation_size;
  VkDescriptorType descriptor_type;
  VkTensorARM tensor;
  VkTensorViewARM tensor_view;
  VkBuffer buffer;
  VkImage image;
  VkImageView image_view;
  VkSampler sampler;
  VkDeviceMemory image_memory;
  VkDeviceMemory memory;
  VkExtent3D image_extent;
  void* persistent_memory = nullptr;
  bool owns_memory = true;
  bool owns_image_memory = true;
  bool is_input;
} IO;

typedef struct PersistentMappedMemory {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* data = nullptr;
} PersistentMappedMemory;

typedef struct SegmentState {
  int segment_id = -1;
  bool use_data_graph_pipeline = true;
  VkPipeline vk_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout vk_layout = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptor_sets;
  VkDataGraphPipelineSessionARM vk_session = VK_NULL_HANDLE;
  VkShaderModule vk_shader = VK_NULL_HANDLE;
  std::array<uint32_t, 3> dispatch_shape = {1, 1, 1};

  // to work with data provide by arm neural statistics api
  bool neural_statistics_bind_point_available = false;
  VkDeviceMemory neural_statistics_memory = VK_NULL_HANDLE;
  VkDeviceSize neural_statistics_memory_size = 0;
  bool neural_statistics_memory_host_visible = false;
  bool neural_statistics_memory_host_coherent = false;
  std::string neural_statistics_status;
} SegmentState;

typedef struct ResourceAlloc {
  VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  VkTensorARM tensor = VK_NULL_HANDLE;
  VkTensorViewARM tensor_view = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDeviceMemory image_memory = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  bool owns_memory = true;
  bool owns_image_memory = true;
} ResourceAlloc;

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
      VkCommandPool pool,
      uint32_t queue_family_index = UINT32_MAX)
      : vk_instance(inst),
        vk_physical(phys),
        vk_device(dev),
        vk_queue(queue),
        vk_command_pool(pool),
        vk_queue_family_index(queue_family_index) {}

  /*
   * Process a VGF ready for execution, allocate necessary Vulkan objects.
   */
  bool process_vgf(
      const char* vgf_data,
      size_t vgf_size,
      ArrayRef<CompileSpec> specs,
      executorch::runtime::EventTracer* event_tracer = nullptr);

  /*
   * Execute the VGF we've previously processed.
   */
  bool execute_vgf(executorch::runtime::EventTracer* event_tracer = nullptr);
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
  vector<int> model_input_io_index;
  vector<int> model_output_io_index;
  size_t model_input_count = 0;
  size_t model_output_count = 0;
  std::vector<SegmentState> segments;
  std::vector<ResourceAlloc> extra_allocs;

  // Mapping to persistent IO memory
  static bool map_io(IO* io, void** handle) {
    if (io->persistent_memory == nullptr) {
      ET_LOG(Error, "Vulkan IO memory is not persistently mapped");
      return false;
    }
    *handle = io->persistent_memory;
    return true;
  }

  // Unmapping to persistent IO memory
  static void unmap_io(IO* io) {
    (void)io;
  }

  // to work with arm neural statistics data
  std::vector<VgfNeuralStatisticsSegmentContext>
  get_neural_statistics_segment_contexts() const;

  std::string collect_neural_statistics_metadata() const;

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
  uint32_t vk_queue_family_index = UINT32_MAX;

  bool timestamp_queries_enabled = false;
  uint32_t timestamp_valid_bits = 0;
  double timestamp_period_ns = 0.0;
  VkQueryPool vk_timestamp_query_pool = VK_NULL_HANDLE;

  // per-VgfRepr-instance objects allocated in process_vgf, used (can be more
  // than once) in execute_vgf
  VkCommandBuffer vk_execute_cmd = VK_NULL_HANDLE;
  VkFence vk_execute_fence = VK_NULL_HANDLE;
  // Note: the vector of tensor memory is stored in IOs above

  bool init_timestamp_queries();
  void read_timestamp_queries(executorch::runtime::EventTracer* event_tracer);

  std::vector<PersistentMappedMemory> persistent_mapped_memories;
  bool map_persistent_io_memory();
  void unmap_persistent_io_memory();
};

} // namespace vgf
} // namespace backends
} // namespace executorch
