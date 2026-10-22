/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include <executorch/backends/arm/runtime/VGFNeuralStatistics.h>
#include <executorch/runtime/platform/runtime.h>

namespace vgf = executorch::backends::vgf;

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value == nullptr ? "" : value);
#else
  if (value == nullptr) {
    unsetenv(name);
  } else {
    setenv(name, value, 1);
  }
#endif
}

class ScopedNeuralStatisticsEnv {
 public:
  ScopedNeuralStatisticsEnv() {
    // Reading the runtime config can ET_LOG, which aborts unless the PAL is up.
    executorch::runtime::runtime_init();

    const char* enable = std::getenv(vgf::kVgfNeuralStatisticsEnableEnv);
    const char* mode = std::getenv(vgf::kVgfNeuralStatisticsModeEnv);
    if (enable != nullptr) {
      old_enable_ = enable;
      had_enable_ = true;
    }
    if (mode != nullptr) {
      old_mode_ = mode;
      had_mode_ = true;
    }
  }

  ~ScopedNeuralStatisticsEnv() {
    set_env(
        vgf::kVgfNeuralStatisticsEnableEnv,
        had_enable_ ? old_enable_.c_str() : nullptr);
    set_env(
        vgf::kVgfNeuralStatisticsModeEnv,
        had_mode_ ? old_mode_.c_str() : nullptr);
  }

 private:
  bool had_enable_ = false;
  bool had_mode_ = false;
  std::string old_enable_;
  std::string old_mode_;
};

vgf::VgfNeuralStatisticsMetadataContext make_g2_metadata_context(
    int mode_index = 1) {
  vgf::VgfNeuralStatisticsMetadataContext metadata;
  metadata.mode_index = mode_index;
  metadata.target_available = true;
  metadata.device_name = "Mali-G2-Pro-NX MC1";
  metadata.vendor_id = 5045;
  metadata.device_id = 7;
  metadata.driver_version = 11;
  return metadata;
}

} // namespace

TEST(VgfNeuralStatisticsTest, RuntimeConfigDefaultsToMode1) {
  ScopedNeuralStatisticsEnv scoped_env;
  set_env(vgf::kVgfNeuralStatisticsEnableEnv, "1");
  set_env(vgf::kVgfNeuralStatisticsModeEnv, nullptr);

  const auto config = vgf::get_vgf_neural_statistics_runtime_config();
  EXPECT_TRUE(config.requested);
  EXPECT_EQ(config.mode_index, 1);
}

TEST(VgfNeuralStatisticsTest, RuntimeConfigMapsMode0ToStatistics0) {
  ScopedNeuralStatisticsEnv scoped_env;
  set_env(vgf::kVgfNeuralStatisticsEnableEnv, "true");
  set_env(vgf::kVgfNeuralStatisticsModeEnv, "0");

  const auto config = vgf::get_vgf_neural_statistics_runtime_config();
  EXPECT_TRUE(config.requested);
  EXPECT_EQ(config.mode_index, 0);
}

TEST(VgfNeuralStatisticsTest, RuntimeConfigFallsBackToMode1) {
  ScopedNeuralStatisticsEnv scoped_env;
  set_env(vgf::kVgfNeuralStatisticsEnableEnv, "1");
  set_env(vgf::kVgfNeuralStatisticsModeEnv, "not-a-mode");

  const auto config = vgf::get_vgf_neural_statistics_runtime_config();
  EXPECT_TRUE(config.requested);
  EXPECT_EQ(config.mode_index, 1);
}

TEST(VgfNeuralStatisticsTest, RuntimeConfigRecognizesFalseLikeValues) {
  ScopedNeuralStatisticsEnv scoped_env;
  set_env(vgf::kVgfNeuralStatisticsEnableEnv, "Off");
  set_env(vgf::kVgfNeuralStatisticsModeEnv, "1");

  const auto config = vgf::get_vgf_neural_statistics_runtime_config();
  EXPECT_FALSE(config.requested);
}

TEST(VgfNeuralStatisticsTest, SerializesUnavailableWrapper) {
  const std::string metadata =
      vgf::make_vgf_neural_statistics_unavailable_metadata(
          "api missing", make_g2_metadata_context());

  EXPECT_NE(
      metadata.find("\"schema\":\"executorch.vgf.neural_statistics\""),
      std::string::npos);
  EXPECT_NE(metadata.find("\"schema_version\":2"), std::string::npos);
  EXPECT_NE(metadata.find("\"api_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("api missing"), std::string::npos);
}

TEST(VgfNeuralStatisticsTest, SerializesMockedBlobs) {
  vgf::VgfNeuralStatisticsCollection collection;
  collection.api_available = true;
  collection.data_available = true;
  collection.metadata_context = make_g2_metadata_context();

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

  EXPECT_NE(metadata.find("\"schema_version\":2"), std::string::npos);
  EXPECT_NE(metadata.find("\"api_available\":true"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":true"), std::string::npos);
  EXPECT_NE(metadata.find("\"segment_id\":7"), std::string::npos);
  EXPECT_NE(
      metadata.find(
          "\"statistics_mode\":{\"executorch_index\":1,\"vulkan_value\":2,"
          "\"vulkan_name\":\"VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS1_ARM\"}"),
      std::string::npos);
  EXPECT_NE(
      metadata.find("\"device_name\":\"Mali-G2-Pro-NX MC1\""),
      std::string::npos);
  EXPECT_NE(metadata.find("\"vendor_id\":5045"), std::string::npos);
  EXPECT_NE(
      metadata.find("\"counter_layout\":{\"available\":true,"
                    "\"schema_id\":\"arm.mali-g2.neural-statistics.mode1\","
                    "\"schema_version\":1,\"endianness\":\"little\","
                    "\"word_type\":\"uint32\",\"words_per_task\":64,"
                    "\"leading_block_count\":1"),
      std::string::npos);

  // Base64("AQID") = {0x01,0x02,0x03}; Base64("3q0=") = {0xDE,0xAD}.
  EXPECT_NE(metadata.find("\"data\":\"AQID\""), std::string::npos);
  EXPECT_NE(metadata.find("\"data\":\"3q0=\""), std::string::npos);
}

TEST(VgfNeuralStatisticsTest, SerializesMode0InterpretationMetadata) {
  vgf::VgfNeuralStatisticsCollection collection;
  collection.metadata_context = make_g2_metadata_context(/*mode_index=*/0);

  const std::string metadata =
      vgf::serialize_vgf_neural_statistics_collection(collection);

  EXPECT_NE(
      metadata.find(
          "\"statistics_mode\":{\"executorch_index\":0,\"vulkan_value\":1,"
          "\"vulkan_name\":\"VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS0_ARM\"}"),
      std::string::npos);
  EXPECT_NE(
      metadata.find("\"schema_id\":\"arm.mali-g2.neural-statistics.mode0\""),
      std::string::npos);
}

TEST(VgfNeuralStatisticsTest, DoesNotClaimG2LayoutForUnknownTarget) {
  vgf::VgfNeuralStatisticsCollection collection;
  collection.metadata_context = make_g2_metadata_context();
  collection.metadata_context.device_name = "Unknown accelerator";

  const std::string metadata =
      vgf::serialize_vgf_neural_statistics_collection(collection);

  EXPECT_NE(
      metadata.find("\"counter_layout\":{\"available\":false"),
      std::string::npos);
  EXPECT_NE(metadata.find("\"schema_id\":\"\""), std::string::npos);
}

TEST(VgfNeuralStatisticsTest, TestCollectorMocksVulkanApi) {
  vgf::set_vgf_neural_statistics_collector_for_test(
      [](VkDevice, const std::vector<vgf::VgfNeuralStatisticsSegmentContext>&)
          -> vgf::VgfNeuralStatisticsCollection {
        vgf::VgfNeuralStatisticsCollection collection;
        collection.api_available = true;
        collection.data_available = true;
        return collection;
      });

  const std::string metadata = vgf::collect_vgf_neural_statistics_metadata(
      VK_NULL_HANDLE, {}, make_g2_metadata_context());

  EXPECT_NE(metadata.find("\"schema_version\":2"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":true"), std::string::npos);
  EXPECT_NE(
      metadata.find("\"statistics_mode\":{\"executorch_index\":1"),
      std::string::npos);
  EXPECT_NE(
      metadata.find("\"device_name\":\"Mali-G2-Pro-NX MC1\""),
      std::string::npos);
  EXPECT_NE(
      metadata.find("\"schema_id\":\"arm.mali-g2.neural-statistics.mode1\""),
      std::string::npos);

  vgf::reset_vgf_neural_statistics_collector_for_test();
}

TEST(VgfNeuralStatisticsTest, DefaultCollectorHandlesUnavailableApi) {
  vgf::reset_vgf_neural_statistics_collector_for_test();

  const std::string metadata = vgf::collect_vgf_neural_statistics_metadata(
      VK_NULL_HANDLE, {}, make_g2_metadata_context());

  EXPECT_NE(metadata.find("\"schema_version\":2"), std::string::npos);
  EXPECT_NE(metadata.find("\"data_available\":false"), std::string::npos);
  EXPECT_NE(metadata.find("\"available\":false"), std::string::npos);
}