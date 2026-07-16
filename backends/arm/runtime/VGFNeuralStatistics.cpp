/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/arm/runtime/VGFNeuralStatistics.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

#include <executorch/runtime/platform/log.h>

// In this file we checks Vulkan API availability,
// queries debug database/statistics info,
// maps statistics memory if available,
// serializes everything into JSON.
namespace executorch {
namespace backends {
namespace vgf {
namespace {

// Converts a C++ string into a valid JSON string literal.
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
              << static_cast<int>(c) << std::dec;
        } else {
          out << c;
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

// Converts raw binary bytes into base64 text.
// We need this, because JSON can contain arbitrary
// raw binary data.
std::string base64_encode(const std::vector<uint8_t>& input) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);

  int val = 0;
  int valb = -6;
  for (uint8_t c : input) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      output.push_back(kAlphabet[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }

  if (valb > -6) {
    output.push_back(kAlphabet[((val << 8) >> (valb + 8)) & 0x3F]);
  }

  while (output.size() % 4 != 0) {
    output.push_back('=');
  }

  return output;
}

void append_bool(std::ostringstream& out, bool value) {
  out << (value ? "true" : "false");
}

void append_blob(
    std::ostringstream& out,
    const char* name,
    const VgfNeuralStatisticsBlob& blob) {
  out << json_escape(name) << ":{";
  out << "\"available\":";
  append_bool(out, blob.available);
  out << ",\"is_text\":";
  append_bool(out, blob.is_text);
  out << ",\"vulkan_result\":" << blob.vulkan_result;
  out << ",\"size\":" << blob.data.size();
  out << ",\"encoding\":\"base64\"";
  out << ",\"reason\":" << json_escape(blob.reason);
  out << ",\"data\":" << json_escape(base64_encode(blob.data));
  out << "}";
}

bool contains_property(
    const std::vector<VkDataGraphPipelinePropertyARM>& properties,
    VkDataGraphPipelinePropertyARM property) {
  return std::find(properties.begin(), properties.end(), property) !=
      properties.end();
}

VgfNeuralStatisticsBlob make_unavailable_blob(const std::string& reason) {
  VgfNeuralStatisticsBlob blob;
  blob.available = false;
  blob.reason = reason;
  return blob;
}

VgfNeuralStatisticsBlob query_pipeline_property(
    VkDevice device,
    VkPipeline pipeline,
    VkDataGraphPipelinePropertyARM property) {
  if (device == VK_NULL_HANDLE) {
    return make_unavailable_blob("VkDevice is null");
  }
  if (pipeline == VK_NULL_HANDLE) {
    return make_unavailable_blob("VkPipeline is null");
  }

#if defined(VK_ARM_data_graph) &&                                                 \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM) && \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM)

  if (!vkGetDataGraphPipelineAvailablePropertiesARM ||
      !vkGetDataGraphPipelinePropertiesARM) {
    return make_unavailable_blob(
        "VK_ARM_data_graph pipeline property query functions are not loaded");
  }

  VkDataGraphPipelineInfoARM pipeline_info{
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_INFO_ARM,
      .pNext = nullptr,
      .dataGraphPipeline = pipeline,
  };

  uint32_t property_count = 0;
  VkResult result = vkGetDataGraphPipelineAvailablePropertiesARM(
      device, &pipeline_info, &property_count, nullptr);
  if (result != VK_SUCCESS) {
    VgfNeuralStatisticsBlob blob;
    blob.available = false;
    blob.vulkan_result = static_cast<int32_t>(result);
    blob.reason =
        "vkGetDataGraphPipelineAvailablePropertiesARM failed when querying count";
    return blob;
  }

  std::vector<VkDataGraphPipelinePropertyARM> properties(property_count);
  if (property_count > 0) {
    result = vkGetDataGraphPipelineAvailablePropertiesARM(
        device, &pipeline_info, &property_count, properties.data());
    if (result != VK_SUCCESS) {
      VgfNeuralStatisticsBlob blob;
      blob.available = false;
      blob.vulkan_result = static_cast<int32_t>(result);
      blob.reason =
          "vkGetDataGraphPipelineAvailablePropertiesARM failed when querying properties";
      return blob;
    }
  }

  if (!contains_property(properties, property)) {
    return make_unavailable_blob(
        "Requested VK_ARM_data_graph pipeline property is not available");
  }

  VkDataGraphPipelinePropertyQueryResultARM query{
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT_ARM,
      .pNext = nullptr,
      .property = property,
      .isText = VK_FALSE,
      .dataSize = 0,
      .pData = nullptr,
  };

  result =
      vkGetDataGraphPipelinePropertiesARM(device, &pipeline_info, 1, &query);
  if (result != VK_SUCCESS) {
    VgfNeuralStatisticsBlob blob;
    blob.available = false;
    blob.vulkan_result = static_cast<int32_t>(result);
    blob.reason =
        "vkGetDataGraphPipelinePropertiesARM failed when querying property size";
    return blob;
  }

  std::vector<uint8_t> data(query.dataSize);
  if (!data.empty()) {
    query.pData = data.data();
    result =
        vkGetDataGraphPipelinePropertiesARM(device, &pipeline_info, 1, &query);
    if (result != VK_SUCCESS) {
      VgfNeuralStatisticsBlob blob;
      blob.available = false;
      blob.vulkan_result = static_cast<int32_t>(result);
      blob.reason =
          "vkGetDataGraphPipelinePropertiesARM failed when querying property data";
      return blob;
    }

    if (query.dataSize < data.size()) {
      data.resize(query.dataSize);
    }
  }

  VgfNeuralStatisticsBlob blob;
  blob.available = true;
  blob.is_text = query.isText == VK_TRUE;
  blob.vulkan_result = static_cast<int32_t>(result);
  blob.data = std::move(data);
  return blob;
#else
  (void)device;
  (void)pipeline;
  (void)property;
  return make_unavailable_blob(
      "Vulkan headers do not expose VK_ARM_data_graph neural accelerator properties");
#endif
}

VgfNeuralStatisticsBlob read_statistics_memory(
    VkDevice device,
    const VgfNeuralStatisticsSegmentContext& segment) {
  if (device == VK_NULL_HANDLE) {
    return make_unavailable_blob("VkDevice is null");
  }
  if (!segment.statistics_bind_point_available) {
    return make_unavailable_blob(
        segment.statistics_bind_point_reason.empty()
            ? "Neural accelerator statistics bind point is not available"
            : segment.statistics_bind_point_reason);
  }
  if (segment.statistics_memory == VK_NULL_HANDLE ||
      segment.statistics_memory_size == 0) {
    return make_unavailable_blob(
        segment.statistics_bind_point_reason.empty()
            ? "Neural accelerator statistics memory is not bound"
            : segment.statistics_bind_point_reason);
  }
  if (!segment.statistics_memory_host_visible) {
    return make_unavailable_blob(
        "Neural accelerator statistics memory is not host visible");
  }

  void* mapped = nullptr;
  VkResult result = vkMapMemory(
      device,
      segment.statistics_memory,
      /*offset=*/0,
      /*size=*/VK_WHOLE_SIZE,
      /*flags=*/0,
      &mapped);
  if (result != VK_SUCCESS) {
    VgfNeuralStatisticsBlob blob;
    blob.available = false;
    blob.vulkan_result = static_cast<int32_t>(result);
    blob.reason = "vkMapMemory failed for neural accelerator statistics memory";
    return blob;
  }

  if (!segment.statistics_memory_host_coherent) {
    const VkMappedMemoryRange mapped_range{
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = segment.statistics_memory,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    result = vkInvalidateMappedMemoryRanges(device, 1, &mapped_range);
    if (result != VK_SUCCESS) {
      vkUnmapMemory(device, segment.statistics_memory);

      VgfNeuralStatisticsBlob blob;
      blob.available = false;
      blob.vulkan_result = static_cast<int32_t>(result);
      blob.reason =
          "vkInvalidateMappedMemoryRanges failed for non-coherent neural accelerator statistics memory";
      return blob;
    }
  }

  std::vector<uint8_t> data(
      static_cast<size_t>(segment.statistics_memory_size));
  if (!data.empty()) {
    std::memcpy(data.data(), mapped, data.size());
  }

  vkUnmapMemory(device, segment.statistics_memory);

  VgfNeuralStatisticsBlob blob;
  blob.available = true;
  blob.is_text = false;
  blob.vulkan_result = static_cast<int32_t>(VK_SUCCESS);
  blob.data = std::move(data);
  return blob;
}

VgfNeuralStatisticsCollectorForTest& test_collector_storage() {
  static VgfNeuralStatisticsCollectorForTest collector;
  return collector;
}

} // namespace

bool vgf_neural_statistics_api_available() {
#if defined(VK_ARM_data_graph) &&                                                 \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM) && \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM)
  return vkGetDataGraphPipelineAvailablePropertiesARM != nullptr &&
      vkGetDataGraphPipelinePropertiesARM != nullptr;
#else
  return false;
#endif
}

VgfNeuralStatisticsCollection collect_vgf_neural_statistics(
    VkDevice device,
    const std::vector<VgfNeuralStatisticsSegmentContext>& segments) {
  VgfNeuralStatisticsCollection collection;

#if defined(VK_ARM_data_graph) &&                                                 \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM) && \
    defined(                                                                      \
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM)

  collection.api_available = vgf_neural_statistics_api_available();

  if (device == VK_NULL_HANDLE) {
    collection.reason = "VkDevice is null";
    return collection;
  }

  if (!collection.api_available) {
    collection.reason =
        "VK_ARM_data_graph neural accelerator property query API is unavailable";
    return collection;
  }

  if (segments.empty()) {
    collection.reason = "No VGF segments are available for collection";
    return collection;
  }

  for (const auto& segment : segments) {
    VgfCollectedSegmentNeuralStatistics collected;
    collected.segment_id = segment.segment_id;
    collected.is_data_graph_pipeline = segment.is_data_graph_pipeline;
    collected.statistics_bind_point_available =
        segment.statistics_bind_point_available;
    collected.statistics_memory_host_visible =
        segment.statistics_memory_host_visible;
    collected.statistics_memory_host_coherent =
        segment.statistics_memory_host_coherent;
    collected.statistics_bind_point_reason =
        segment.statistics_bind_point_reason;

    if (!segment.is_data_graph_pipeline) {
      collected.debug_database =
          make_unavailable_blob("Segment is not a data graph pipeline");
      collected.statistics_info =
          make_unavailable_blob("Segment is not a data graph pipeline");
      collected.statistics_memory =
          make_unavailable_blob("Segment is not a data graph pipeline");
      collection.segments.push_back(std::move(collected));
      continue;
    }

    collected.debug_database = query_pipeline_property(
        device,
        segment.pipeline,
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM);
    collected.statistics_info = query_pipeline_property(
        device,
        segment.pipeline,
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM);

    collected.statistics_memory = read_statistics_memory(device, segment);

    if (collected.debug_database.available ||
        collected.statistics_info.available ||
        collected.statistics_memory.available) {
      collection.data_available = true;
    }

    collection.segments.push_back(std::move(collected));
  }

  if (!collection.data_available) {
    collection.reason =
        "VK_ARM_data_graph neural accelerator statistics data is not available";
  }

  return collection;

#else
  (void)device;
  (void)segments;

  collection.api_available = false;
  collection.data_available = false;
  collection.reason =
      "VK_ARM_data_graph neural accelerator property query API is unavailable";
  return collection;
#endif
}

std::string serialize_vgf_neural_statistics_collection(
    const VgfNeuralStatisticsCollection& collection) {
  std::ostringstream out;
  out << "{";
  out << "\"schema\":" << json_escape(kVgfNeuralStatisticsSchema);
  out << ",\"schema_version\":" << kVgfNeuralStatisticsSchemaVersion;
  out << ",\"backend\":\"VgfBackend\"";
  out << ",\"api\":\"VK_ARM_data_graph\"";
  out << ",\"event_name\":"
      << json_escape(kVgfNeuralStatisticsDelegateEventName);
  out << ",\"api_available\":";
  append_bool(out, collection.api_available);
  out << ",\"data_available\":";
  append_bool(out, collection.data_available);

  // Write that not available
  out << ",\"available\":";
  append_bool(out, collection.data_available);

  out << ",\"reason\":" << json_escape(collection.reason);
  out << ",\"segments\":[";

  for (size_t i = 0; i < collection.segments.size(); ++i) {
    const auto& segment = collection.segments[i];
    if (i > 0) {
      out << ",";
    }

    out << "{";
    out << "\"segment_id\":" << segment.segment_id;
    out << ",\"is_data_graph_pipeline\":";
    append_bool(out, segment.is_data_graph_pipeline);
    out << ",\"statistics_bind_point_available\":";
    append_bool(out, segment.statistics_bind_point_available);
    out << ",\"statistics_memory_host_visible\":";
    append_bool(out, segment.statistics_memory_host_visible);
    out << ",\"statistics_memory_host_coherent\":";
    append_bool(out, segment.statistics_memory_host_coherent);
    out << ",\"statistics_bind_point_reason\":"
        << json_escape(segment.statistics_bind_point_reason);
    out << ",";
    append_blob(out, "debug_database", segment.debug_database);
    out << ",";
    append_blob(out, "statistics_info", segment.statistics_info);
    out << ",";
    append_blob(out, "statistics_memory", segment.statistics_memory);
    out << "}";
  }

  out << "]";
  out << "}";
  return out.str();
}

std::string make_vgf_neural_statistics_unavailable_metadata(
    const std::string& reason) {
  VgfNeuralStatisticsCollection collection;
  collection.api_available = false;
  collection.data_available = false;
  collection.reason = reason;
  return serialize_vgf_neural_statistics_collection(collection);
}

std::string collect_vgf_neural_statistics_metadata(
    VkDevice device,
    const std::vector<VgfNeuralStatisticsSegmentContext>& segments) {
  const auto& test_collector = test_collector_storage();
  if (test_collector) {
    return test_collector(device, segments);
  }

  return serialize_vgf_neural_statistics_collection(
      collect_vgf_neural_statistics(device, segments));
}

void set_vgf_neural_statistics_collector_for_test(
    VgfNeuralStatisticsCollectorForTest collector) {
  test_collector_storage() = std::move(collector);
}

void reset_vgf_neural_statistics_collector_for_test() {
  test_collector_storage() = nullptr;
}

} // namespace vgf
} // namespace backends
} // namespace executorch