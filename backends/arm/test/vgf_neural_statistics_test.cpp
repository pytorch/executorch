/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <executorch/backends/arm/runtime/VGFNeuralStatistics.h>

namespace vgf = executorch::backends::vgf;

TEST(VgfNeuralStatisticsTest, SerializesUnavailableWrapper) {
  const std::string metadata =
      vgf::make_vgf_neural_statistics_unavailable_metadata("api missing");

  EXPECT_NE(
      metadata.find("\"schema\":\"executorch.vgf.neural_statistics\""),
      std::string::npos);
  EXPECT_NE(metadata.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(metadata.find("\"api_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("api missing"), std::string::npos);
}

TEST(VgfNeuralStatisticsTest, SerializesMockedBlobs) {
  vgf::VgfNeuralStatisticsCollection collection;
  collection.api_available = true;
  collection.data_available = true;

  vgf::VgfCollectedSegmentNeuralStatistics segment;
  segment.segment_id = 7;
  segment.is_data_graph_pipeline = true;
  segment.statistics_bind_point_available = true;
  segment.statistics_memory_host_visible = true;
  segment.statistics_memory_host_coherent = true;

  segment.debug_database.available = true;
  segment.debug_database.data = {0x01, 0x02, 0x03};

  segment.statistics_info.available = true;
  segment.statistics_info.is_text = true;
  segment.statistics_info.data = {'i', 'n', 'f', 'o'};

  segment.statistics_memory.available = true;
  segment.statistics_memory.data = {0xDE, 0xAD};

  collection.segments.push_back(segment);

  const std::string metadata =
      vgf::serialize_vgf_neural_statistics_collection(collection);

  EXPECT_NE(metadata.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(metadata.find("\"api_available\":true"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":true"), std::string::npos);
  EXPECT_NE(metadata.find("\"segment_id\":7"), std::string::npos);

  // Base64("AQID") = {0x01,0x02,0x03}; Base64("3q0=") = {0xDE,0xAD}.
  EXPECT_NE(metadata.find("\"data\":\"AQID\""), std::string::npos);
  EXPECT_NE(metadata.find("\"data\":\"3q0=\""), std::string::npos);
}

TEST(VgfNeuralStatisticsTest, TestCollectorMocksVulkanApi) {
  vgf::set_vgf_neural_statistics_collector_for_test(
      [](VkDevice, const std::vector<vgf::VgfNeuralStatisticsSegmentContext>&)
          -> std::string {
        return "{\"schema\":\"executorch.vgf.neural_statistics\","
               "\"schema_version\":1,"
               "\"api_available\":true,"
               "\"data_available\":true,"
               "\"available\":true,"
               "\"segments\":[]}";
      });

  const std::string metadata =
      vgf::collect_vgf_neural_statistics_metadata(VK_NULL_HANDLE, {});

  EXPECT_NE(metadata.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":true"), std::string::npos);

  vgf::reset_vgf_neural_statistics_collector_for_test();
}

TEST(VgfNeuralStatisticsTest, DefaultCollectorHandlesUnavailableApi) {
  vgf::reset_vgf_neural_statistics_collector_for_test();

  const std::string metadata =
      vgf::collect_vgf_neural_statistics_metadata(VK_NULL_HANDLE, {});

  EXPECT_NE(metadata.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"available\":false"), std::string::npos);
}