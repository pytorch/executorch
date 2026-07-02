/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

namespace executorch {
namespace backends {
namespace vgf {

constexpr const char* kVgfNeuralStatisticsDelegateEventName =
    "VGF_NEURAL_STATISTICS";
constexpr const char* kVgfNeuralStatisticsSchema =
    "executorch.vgf.neural_statistics";
constexpr int kVgfNeuralStatisticsSchemaVersion = 1;

// One binary payload from neural statistics API
struct VgfNeuralStatisticsBlob {
  // Whether we got it successfully
  bool available = false;
  // Whether it is a text like JSON
  bool is_text = false;
  // Vulkan result code from query
  int32_t vulkan_result = 0;
  // Why blob is not available
  std::string reason;
  // Actual payload, raw bytes
  std::vector<uint8_t> data;
};

// Info needed to collect stats for one VGF segment
struct VgfNeuralStatisticsSegmentContext {
  // id of the segment
  int segment_id = -1;
  // Stats are only for data graph pipeline
  // We skip if it is false
  bool is_data_graph_pipeline = false;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
  // We record whether stats bind point was available.
  bool statistics_bind_point_available = false;
  // We record memory properties
  VkDeviceMemory statistics_memory = VK_NULL_HANDLE;
  VkDeviceSize statistics_memory_size = 0;
  bool statistics_memory_host_visible = false;
  bool statistics_memory_host_coherent = false;
  // Why it is unavailable
  std::string statistics_bind_point_reason;
};

// This is the output for one segment after collection.
struct VgfCollectedSegmentNeuralStatistics {
  // Status fields

  // id of the segment
  int segment_id = -1;
  // Stats are only for data graph pipeline
  // We skip if it is false
  bool is_data_graph_pipeline = false;
  // We record whether stats bind point was available.
  bool statistics_bind_point_available = false;
  // We record memory properties
  bool statistics_memory_host_visible = false;
  bool statistics_memory_host_coherent = false;
  // Why it is unavailable
  std::string statistics_bind_point_reason;

  // Debug database blob queried
  VgfNeuralStatisticsBlob debug_database;
  // API-provided information about the statistics
  VgfNeuralStatisticsBlob statistics_info;
  // Raw bytes read from the neural statistics memory bind point
  VgfNeuralStatisticsBlob statistics_memory;
};

// This is the top-level result for the whole VGF execution
struct VgfNeuralStatisticsCollection {
  // Whether API is supported at all
  bool api_available = false;
  // Whether driver provides data
  bool data_available = false;
  // Top level explanation to the user if something went wrong
  std::string reason;
  // Per segment results
  std::vector<VgfCollectedSegmentNeuralStatistics> segments;
};

// Checks whether the neural statistics Vulkan API can be used
bool vgf_neural_statistics_api_available();

VgfNeuralStatisticsCollection collect_vgf_neural_statistics(
    VkDevice device,
    const std::vector<VgfNeuralStatisticsSegmentContext>& segments);

// We convert collection into the JSON file that will be
// stored in ETDump delegate metadata.
std::string serialize_vgf_neural_statistics_collection(
    const VgfNeuralStatisticsCollection& collection);

// Creates metadata JSON string when collection cannot happen
std::string make_vgf_neural_statistics_unavailable_metadata(
    const std::string& reason);

// High level function used by the backend
std::string collect_vgf_neural_statistics_metadata(
    VkDevice device,
    const std::vector<VgfNeuralStatisticsSegmentContext>& segments);

// Functions for testing:
// Define mackable function for testing
using VgfNeuralStatisticsCollectorForTest = std::function<std::string(
    VkDevice,
    const std::vector<VgfNeuralStatisticsSegmentContext>&)>;

//  This lets a unit test override the real collector
void set_vgf_neural_statistics_collector_for_test(
    VgfNeuralStatisticsCollectorForTest collector);

void reset_vgf_neural_statistics_collector_for_test();

} // namespace vgf
} // namespace backends
} // namespace executorch