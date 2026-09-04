/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <executorch/backends/arm/runtime/VGFDiagnostics.h>

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

class ScopedDiagnosticsDirEnv {
 public:
  ScopedDiagnosticsDirEnv() {
    const char* old = std::getenv(vgf::kVgfDiagnosticsDirEnv);
    if (old != nullptr) {
      old_value_ = old;
      had_old_value_ = true;
    }
  }

  ~ScopedDiagnosticsDirEnv() {
    set_env(
        vgf::kVgfDiagnosticsDirEnv,
        had_old_value_ ? old_value_.c_str() : nullptr);
  }

 private:
  bool had_old_value_ = false;
  std::string old_value_;
};

} // namespace

TEST(VgfDiagnosticsTest, TransferObservationUsesDpaCompatibleFields) {
  vgf::VgfTransferObservation observation;
  observation.kind = "INPUT_COPY";
  observation.direction = "INPUT";
  observation.slot = 3;
  observation.io_index = 7;
  observation.bytes_copied = 4096;
  observation.descriptor_type = "VK_DESCRIPTOR_TYPE_TENSOR_ARM";

  const std::string json = vgf::serialize_vgf_transfer_observation(observation);

  EXPECT_NE(json.find("\"kind\":\"INPUT_COPY\""), std::string::npos);
  EXPECT_NE(json.find("\"bytes_copied\":4096"), std::string::npos);
  EXPECT_NE(json.find("\"transfer_mode\":\"STAGED_COPY\""), std::string::npos);
  EXPECT_NE(
      json.find("\"binding_provenance\":\"BACKEND_OWNED\""), std::string::npos);
}

TEST(VgfDiagnosticsTest, CapabilityReportIncludesMappedRequirements) {
  vgf::VgfCapabilityReport report;
  report.source_revision = "abc123";
  report.device_name = "test-device";

  vgf::VgfIoCapability io;
  io.io_index = 1;
  io.direction = "OUTPUT";
  io.descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
  io.logical_bytes = 128;
  io.memory_requirement_size = 256;
  io.memory_requirement_alignment = 64;
  io.memory_allocation_capacity = 512;
  io.memory_type_bits = 5;
  io.memory_type_index = 2;
  io.memory_property_flags = 7;
  io.exact_resource_created = true;
  io.persistent_mapped = true;
  report.ios.push_back(io);

  const std::string json = vgf::serialize_vgf_capability_report(report);

  EXPECT_NE(json.find("\"source_revision\":\"abc123\""), std::string::npos);
  EXPECT_NE(json.find("\"size\":256"), std::string::npos);
  EXPECT_NE(json.find("\"alignment\":64"), std::string::npos);
  EXPECT_NE(json.find("\"allocation_capacity\":512"), std::string::npos);
  EXPECT_NE(json.find("\"memory_type_bits\":5"), std::string::npos);
  EXPECT_NE(json.find("\"exact_resource_created\":true"), std::string::npos);
  EXPECT_NE(json.find("\"persistent_mapped\":true"), std::string::npos);
}

TEST(
    VgfDiagnosticsTest,
    ExecutionReportDoesNotClaimExternalConversionsAreZero) {
  vgf::VgfExecutionReport report;
  report.invocation = 9;
  report.success = true;
  report.vgf_runtime_cpu_quantize_bytes = 0;
  report.vgf_runtime_cpu_dequantize_bytes = 0;
  report.vgf_runtime_cpu_layout_conversion_bytes = 0;
  report.portable_boundary_conversion_visibility = "unknown";

  const std::string json = vgf::serialize_vgf_execution_report(report);

  EXPECT_NE(
      json.find("\"vgf_runtime_cpu_quantize_bytes\":0"), std::string::npos);
  EXPECT_NE(
      json.find("\"portable_boundary_conversion_visibility\":\"unknown\""),
      std::string::npos);
}

TEST(VgfDiagnosticsTest, OptionalFileReportIsReproducibleJson) {
  ScopedDiagnosticsDirEnv scoped_env;
  const auto root = std::filesystem::temp_directory_path() /
      "executorch_vgf_diagnostics_test";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  set_env(vgf::kVgfDiagnosticsDirEnv, root.string().c_str());

  std::string written_path;
  ASSERT_TRUE(vgf::write_vgf_diagnostics_report(
      "execution_000001.json", "{\"ok\":true}", &written_path));
  ASSERT_FALSE(written_path.empty());

  std::ifstream in(written_path);
  ASSERT_TRUE(in.good());
  std::string contents;
  std::getline(in, contents);
  EXPECT_EQ(contents, "{\"ok\":true}");

  std::filesystem::remove_all(root, ec);
}
