// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

TEST(BenchmarkReport, ShaderBreakdownKeepsDispatchOrderAndDuplicateShaders) {
  const Report dispatches = Report::array(
      {{{"dispatch_index", 0},
        {"shader_name", "copy"},
        {"gwg_size", {{"method", "operator"}, {"value", {8, 1, 1}}}},
        {"lwg_size", {{"method", "operator"}, {"value", {4, 1, 1}}}}},
       {{"dispatch_index", 1},
        {"shader_name", "copy"},
        {"gwg_size", {{"method", "operator"}, {"value", {16, 1, 1}}}},
        {"lwg_size", {{"method", "operator"}, {"value", {8, 1, 1}}}}}});
  const Report breakdown =
      shader_breakdown_report(dispatches, {{1, 10}, {2, 20}, {3, 30}});
  ASSERT_EQ(breakdown.size(), 2);
  EXPECT_EQ(breakdown.at(0).at("shader_name"), "copy");
  EXPECT_EQ(breakdown.at(1).at("dispatch_index"), 1);
  EXPECT_EQ(
      breakdown.at(1).at("gwg_size").at("value"), Report::array({16, 1, 1}));
  EXPECT_EQ(breakdown.at(0).at("gpu_timestamp_shader_us").at("mean"), 2);
  EXPECT_NEAR(
      breakdown.at(1).at("gpu_timestamp_shader_us").at("p90").get<double>(),
      28,
      1e-12);
  EXPECT_TRUE(shader_breakdown_report(Report::array(), {{}, {}}).empty());
  const Report averaged = shader_breakdown_report(
      dispatches, {{1, 10, 3, 30}, {2, 20, 4, 40}, {3, 30, 5, 50}}, 2);
  EXPECT_EQ(averaged.at(0).at("gpu_timestamp_shader_us").at("mean"), 3);
  EXPECT_EQ(averaged.at(1).at("gpu_timestamp_shader_us").at("p90"), 38);
  EXPECT_THROW(
      shader_breakdown_report(dispatches, {{1}}), std::invalid_argument);
  EXPECT_THROW(
      shader_breakdown_report(dispatches, {}, 0), std::invalid_argument);
}

TEST(BenchmarkReport, TableShowsShaderBreakdownUnderOperator) {
  Report report = {
      {"operator_name", "aten.linear.default"},
      {"args", Report::array()},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "linear_shader"},
             {"gwg_size", {{"method", "operator"}, {"value", {8, 1, 1}}}},
             {"lwg_size", {{"method", "operator"}, {"value", {4, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1.25}, {"p90", 2.5}}}}})}};
  report.update(measurement_report(1, true, {5, 10}, {}));
  const std::string table = format_table(Report::array({report}));
  EXPECT_NE(table.find("  linear_shader"), std::string::npos);
  EXPECT_LT(table.find("aten.linear.default"), table.find("linear_shader"));
  EXPECT_NE(table.find("[8,1,1]"), std::string::npos);
  EXPECT_NE(table.find("[4,1,1]"), std::string::npos);
  EXPECT_NE(table.find("1.250"), std::string::npos);
  EXPECT_NE(table.find("2.500"), std::string::npos);
}

TEST(BenchmarkReport, TableShowsResolvedSizesInputsAndPerShaderStatistics) {
  Report report = {
      {"shader_name", "add"},
      {"gwg_size", {{"method", "default_pick_gwg"}, {"value", {8, 2, 1}}}},
      {"lwg_size", {{"method", "explicit"}, {"value", {4, 2, 1}}}},
      {"bindings",
       {{"out",
         {{"type", "buffer"},
          {"access", "write"},
          {"sizes", {99}},
          {"dtype", "int32"}}},
        {"a",
         {{"type", "texture3d"},
          {"access", "read"},
          {"sizes", {1, 1, 1, 8}},
          {"dtype", "float32"}}},
        {"b",
         {{"type", "buffer"},
          {"access", "read_write"},
          {"sizes", {8}},
          {"dtype", "float16"}}},
        {"meta", {{"type", "meta_ubo"}}}}}};
  report.update(measurement_report(10, false, {10, 20, 30}, {}));
  const std::string table = format_table(Report::array({report}));
  EXPECT_EQ(
      table,
      "Shader / operator  GWG      LWG      Inputs                               Mean GPU us  P90 GPU us\n"
      "-------------------------------------------------------------------------------------------------\n"
      "add                [8,2,1]  [4,2,1]  t3[1,1,1,8]f32.width, b[8]f16.width        2.000       2.800");
}

TEST(BenchmarkReport, TableSupportsOperatorsAndUnavailableTimestamps) {
  Report report = {
      {"operator_name", "aten.linear.default"},
      {"args",
       Report::array(
           {{{"tensor",
              {{"sizes", {16, 32}}, {"dtype", "int8"}, {"access", "read"}}}},
            nullptr,
            {{"int", 32}},
            {{"tensor",
              {{"sizes", {4, 32}},
               {"dtype", "float32"},
               {"requested_dtype", "float16"},
               {"access", "read_write"}}}},
            {{"tensor",
              {{"sizes", {4, 16}},
               {"dtype", "float32"},
               {"access", "write"}}}}})}};
  report.update(measurement_report(2, true, {}, {4}));
  const std::string table = format_table(Report::array({report}));
  EXPECT_NE(
      table.find("b[16,32]i8.width, null, 32, b[4,32]f32.width"),
      std::string::npos);
  EXPECT_EQ(table.find("[4,16]"), std::string::npos);
  EXPECT_EQ(table.find("f16"), std::string::npos);
  EXPECT_NE(table.find("N/A"), std::string::npos);
}

TEST(BenchmarkReport, TableIncludesShaderUniformsPushConstantsAndSweepIndexes) {
  const Json spec = {
      {"shader", "add"},
      {"sweep_index", 3},
      {"gwg_size", {8, 1, 1}},
      {"lwg_size", {4, 1, 1}},
      {"tensors", Json::object()},
      {"bindings",
       Json::array(
           {{{"uniform",
              Json::array({{{"dtype", "int32"}, {"data", {4, 5}}}})}}})},
      {"push_constants", Json::array({{{"dtype", "float32"}, {"data", {2}}}})}};
  Report report = shader_report(
      spec, {8, 1, 1}, {4, 1, 1}, Json::object(), Report::object());
  report.update(measurement_report(1, false, {2}, {}));
  const Report error = error_report(spec, "bad shader");
  EXPECT_EQ(report.at("sweep_index"), 3);
  EXPECT_EQ(error.at("sweep_index"), 3);
  const std::string table = format_table(Report::array({report, error}));
  EXPECT_NE(table.find("add [3]"), std::string::npos);
  EXPECT_NE(table.find("[4,5], [2]"), std::string::npos);
  EXPECT_NE(table.find("ERROR add [3]: bad shader"), std::string::npos);
}

TEST(BenchmarkReport, TableIncludesScalarInputsAndAllStorageTypes) {
  Report report = {
      {"operator_name", "aten.convolution.default"},
      {"args",
       Report::array(
           {{{"tensor",
              {{"type", "buffer"},
               {"sizes", {1, 3, 32, 32}},
               {"dtype", "float32"}}}},
            {{"tensor",
              {{"type", "constant"},
               {"sizes", {16, 3, 3, 3}},
               {"layout", "height"},
               {"dtype", "float16"}}}},
            {{"tensor",
              {{"type", "texture2d"},
               {"sizes", {16}},
               {"layout", "height"},
               {"dtype", "float32"}}}},
            {{"tensor",
              {{"type", "buffer"},
               {"sizes", {60, 64, 40, 51}},
               {"layout", "4w4c"},
               {"dtype", "int8x4"}}}},
            {{"int_list", {2, 2}}},
            {{"int_list", {1, 1}}},
            {{"int_list", {1, 1}}},
            {{"bool", false}},
            {{"float", 0.5}},
            {{"string", "same"}},
            {{"int", 1}},
            {{"tensor",
              {{"type", "texture3d"},
               {"sizes", {1, 16, 16, 16}},
               {"dtype", "float32"},
               {"access", "write"}}}}})}};
  report.update(measurement_report(1, true, {1}, {}));
  TableOptions options;
  options.inputs_max_chars = 0;
  const std::string table = format_table(Report::array({report}), options);
  EXPECT_NE(
      table.find(
          "b[1,3,32,32]f32.width, c[16,3,3,3]f16, t2[16]f32.height, b[60,64,40,51]i8x4.4w4c, [2,2], [1,1], [1,1], false, 0.5, \"same\", 1"),
      std::string::npos);
  EXPECT_EQ(table.find("c[16,3,3,3]f16."), std::string::npos);
  EXPECT_EQ(table.find("t3[1,16,16,16]"), std::string::npos);
}

TEST(BenchmarkReport, InputFloatsUseThreeSignificantFigures) {
  Report report = {
      {"operator_name", "op"},
      {"args",
       Report::array(
           {{{"float", 1.234567}},
            {{"float", -0.01234567}},
            {{"float", 9.999}},
            {{"float_list", {12345.67, 0.0000001234567, 0.0, 0.5}}},
            {{"int", 1234567}},
            {{"int_list", {12345, 67890}}},
            {{"string", "1.234567"}},
            false,
            nullptr})}};
  report.update(measurement_report(1, true, {1}, {}));
  const Report original = report;
  const std::string expected =
      "1.23, -0.0123, 10, [1.23e+04,1.23e-07,0,0.5], 1234567, [12345,67890], \"1.234567\", false, null";
  TableOptions options;
  options.inputs_max_chars = 0;
  EXPECT_NE(
      format_table(Report::array({report}), options).find(expected),
      std::string::npos);
  const Report compact = Report::parse(format_condensed_report(report));
  EXPECT_EQ(compact.at("args_str"), expected);
  EXPECT_EQ(
      Report::parse(format_report(report)).at("args"), original.at("args"));
  EXPECT_EQ(report, original);
}

TEST(BenchmarkReport, ShaderInputFloatsUseThreeSignificantFigures) {
  Report report = {
      {"shader_name", "shader"},
      {"gwg_size", {{"value", {1, 1, 1}}}},
      {"lwg_size", {{"value", {1, 1, 1}}}},
      {"bindings",
       {{"params",
         {{"type", "uniform"},
          {"values", Report::array({{{"data", {1.234567, 1234567}}}})}}}}},
      {"push_constants", Report::array({{{"data", {-0.01234567, 9.999}}}})}};
  report.update(measurement_report(1, false, {1}, {}));
  EXPECT_NE(
      format_table(Report::array({report}))
          .find("[1.23,1234567], [-0.0123,10]"),
      std::string::npos);
  const Report compact = Report::parse(format_condensed_report(report));
  EXPECT_EQ(compact.at("bindings"), report.at("bindings"));
  EXPECT_EQ(compact.at("push_constants"), report.at("push_constants"));
}

TEST(BenchmarkReport, TableSeparatesCasesButKeepsTheirShadersTogether) {
  Report report = {
      {"operator_name", "first"},
      {"args", Report::array()},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "child"},
             {"gwg_size", {{"value", {1, 1, 1}}}},
             {"lwg_size", {{"value", {1, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1}, {"p90", 1}}}}})}};
  report.update(measurement_report(1, true, {2}, {}));
  Report second = report;
  second["operator_name"] = "second";
  second.erase("shader_breakdown");
  std::istringstream stream(format_table(Report::array({report, second})));
  std::vector<std::string> lines;
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 6);
  EXPECT_EQ(lines[1], std::string(lines[0].size(), '-'));
  EXPECT_EQ(lines[4], lines[1]);
  EXPECT_EQ(lines[2].find("first"), 0);
  EXPECT_EQ(lines[3].find("  child"), 0);
  EXPECT_EQ(lines[5].find("second"), 0);
}

TEST(BenchmarkReport, TableLimitsTextWithoutChangingReports) {
  Report report = {
      {"operator_name", "operator_name_is_long"},
      {"args", Report::array({{{"string", "abcdefghijklmnopqrstuvwxyz"}}})},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "shader_name_is_long"},
             {"gwg_size", {{"value", {1, 1, 1}}}},
             {"lwg_size", {{"value", {1, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1}, {"p90", 1}}}}})}};
  report.update(measurement_report(1, true, {2}, {}));
  const Report original = report;
  TableOptions options;
  options.name_max_chars = 12;
  options.inputs_max_chars = 10;
  const std::string limited = format_table(Report::array({report}), options);
  EXPECT_NE(limited.find("operator_..."), std::string::npos);
  EXPECT_NE(limited.find("  shader_..."), std::string::npos);
  EXPECT_NE(limited.find("\"abcdef..."), std::string::npos);
  EXPECT_EQ(limited.find("shader_name_is_long"), std::string::npos);
  options.name_max_chars = 0;
  options.inputs_max_chars = 0;
  const std::string full = format_table(Report::array({report}), options);
  EXPECT_NE(full.find("operator_name_is_long"), std::string::npos);
  EXPECT_NE(full.find("  shader_name_is_long"), std::string::npos);
  EXPECT_NE(full.find("\"abcdefghijklmnopqrstuvwxyz\""), std::string::npos);
  options.name_max_chars = 3;
  options.inputs_max_chars = 2;
  const std::string tiny = format_table(Report::array({report}), options);
  EXPECT_NE(tiny.find("  ."), std::string::npos);
  EXPECT_NE(tiny.find(".."), std::string::npos);
  EXPECT_EQ(report, original);
}

TEST(BenchmarkReport, TableDisplaysOptionalValidationStatus) {
  Report report = {{"operator_name", "op"}, {"args", Report::array()}};
  report.update(measurement_report(1, true, {2}, {}));
  EXPECT_EQ(
      format_table(Report::array({report})).find("Validation"),
      std::string::npos);
  Report passed = report;
  passed["validation"] = {{"status", "PASS"}};
  Report failed = report;
  failed["validation"] = {{"status", "FAIL"}};
  const std::string table =
      format_table(Report::array({passed, failed, report}));
  EXPECT_NE(table.find("Validation"), std::string::npos);
  EXPECT_NE(table.find("PASS"), std::string::npos);
  EXPECT_NE(table.find("FAIL"), std::string::npos);
  EXPECT_NE(table.find("SKIP"), std::string::npos);
}

TEST(BenchmarkReport, ReportsOmitLegacyCaseMetadata) {
  const Json spec = {
      {"operator", "aten.linear.default"},
      {"args", Json::array()},
      {"tensors", Json::object()},
      {"legacy_case", {{"op", "linear"}, {"set", "debug"}, {"index", 0}}}};
  EXPECT_FALSE(operator_report(spec, {}).contains("legacy_case"));
  EXPECT_FALSE(error_report(spec, "test error").contains("legacy_case"));
}

TEST(BenchmarkReport, ExplicitRouteMetadataSurvivesResultsAndErrors) {
  Json spec = {
      {"operator", "aten.convolution.default"},
      {"args", Json::array()},
      {"tensors", Json::object()},
      {"route",
       {{"name", "im2col"},
        {"options", {{"intermediate_storage", "buffer"}}}}}};
  const Report expected = {
      {"requested", spec.at("route")}, {"resolved", nullptr}};
  EXPECT_EQ(operator_report(spec, Json::object()).at("route"), expected);
  EXPECT_EQ(error_report(spec, "Unsupported shape").at("route"), expected);
  spec.erase("route");
  EXPECT_FALSE(operator_report(spec, Json::object()).contains("route"));
  EXPECT_FALSE(error_report(spec, "Unsupported shape").contains("route"));
}

TEST(BenchmarkReport, TableShowsRequestedAndResolvedRoutes) {
  Report forced = {
      {"operator_name", "forced"},
      {"args", Report::array()},
      {"route",
       {{"requested",
         {{"name", "im2col"},
          {"options", {{"intermediate_storage", "buffer"}}}}},
        {"resolved",
         {{"name", "im2col"},
          {"options", {{"intermediate_storage", "buffer"}}}}}}},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "child"},
             {"gwg_size", {{"value", {1, 1, 1}}}},
             {"lwg_size", {{"value", {1, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1}, {"p90", 1}}}}})}};
  forced.update(measurement_report(1, true, {2}, {}));
  Report automatic = forced;
  automatic["operator_name"] = "automatic";
  automatic["route"] = {
      {"requested", {{"name", "auto"}, {"options", Report::object()}}},
      {"resolved", {{"name", "depthwise"}, {"options", Report::object()}}}};
  automatic.erase("shader_breakdown");
  Report unknown = automatic;
  unknown["operator_name"] = "unknown";
  unknown["route"]["resolved"] = nullptr;
  Report omitted = unknown;
  omitted["operator_name"] = "omitted";
  omitted.erase("route");
  const std::string table =
      format_table(Report::array({forced, automatic, unknown, omitted}));
  EXPECT_LT(table.find("Shader / operator"), table.find("Route"));
  EXPECT_LT(table.find("Route"), table.find("GWG"));
  EXPECT_NE(table.find("im2col[buffer]"), std::string::npos);
  EXPECT_NE(table.find("auto->depthwise"), std::string::npos);
  std::istringstream stream(table);
  std::vector<std::string> lines;
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 10);
  EXPECT_EQ(lines[1], std::string(lines[0].size(), '-'));
  std::istringstream child(lines[3]);
  std::string shader_name, route, gwg;
  child >> shader_name >> route >> gwg;
  EXPECT_EQ(shader_name, "child");
  EXPECT_EQ(route, "-");
  EXPECT_EQ(gwg, "[1,1,1]");
  EXPECT_NE(lines[7].find("auto"), std::string::npos);
  EXPECT_EQ(lines[7].find("auto->"), std::string::npos);
}

TEST(BenchmarkReport, TableKeepsRouteOnUnsupportedCaseErrors) {
  const Report error = error_report(
      {{"operator", "aten.convolution.default"},
       {"route",
        {{"name", "im2col"},
         {"options", {{"intermediate_storage", "texture3d"}}}}}},
      "Unsupported shape");
  const std::string table = format_table(Report::array({error}));
  EXPECT_NE(table.find("im2col[texture3d]"), std::string::npos);
  EXPECT_NE(table.find("ERROR"), std::string::npos);
  EXPECT_NE(table.find("Unsupported shape"), std::string::npos);
}

TEST(BenchmarkReport, TableRendersSkippedCasesWithoutTimingMeasurements) {
  const Report skipped = {
      {"case_name", "forced convolution"},
      {"operator_name", "aten.convolution.default"},
      {"sweep_index", 2},
      {"status", "SKIP"},
      {"skip_reason", "Unsupported\nshape"},
      {"route",
       {{"requested",
         {{"name", "im2col"},
          {"options", {{"intermediate_storage", "buffer"}}}}},
        {"resolved", nullptr}}},
      {"args",
       Report::array(
           {{{"tensor",
              {{"sizes", {1, 3, 8, 8}},
               {"dtype", "float32"},
               {"type", "buffer"}}}},
            {{"int_list", {2, 2}}},
            {{"tensor",
              {{"sizes", {1, 16, 4, 4}},
               {"dtype", "float32"},
               {"type", "buffer"},
               {"access", "write"}}}}})},
      {"throughput", {{"memory_bytes", nullptr}}}};
  TableOptions options;
  options.name_max_chars = 0;
  const std::string table = format_table(Report::array({skipped}), options);
  EXPECT_NE(
      table.find("aten.convolution.default [2] (SKIP)"), std::string::npos);
  EXPECT_NE(table.find("im2col[buffer]"), std::string::npos);
  EXPECT_NE(table.find("b[1,3,8,8]f32.width, [2,2]"), std::string::npos);
  EXPECT_EQ(table.find("b[1,16,4,4]"), std::string::npos);
  EXPECT_NE(table.find("N/A"), std::string::npos);
  EXPECT_NE(
      table.find("SKIP forced convolution [2]: Unsupported shape"),
      std::string::npos);
  Report without_args = skipped;
  without_args.erase("args");
  without_args["validation"] = {{"status", "SKIP"}};
  EXPECT_NE(
      format_table(Report::array({without_args})).find("Validation"),
      std::string::npos);
}

TEST(BenchmarkReport, TableDisplaysRequestedGpuThroughputColumns) {
  Report report = {{"operator_name", "op"}, {"args", Report::array()}};
  report.update(measurement_report(1, true, {2}, {}));
  const std::string disabled = format_table(Report::array({report}));
  EXPECT_EQ(disabled.find("GPU GiB/s"), std::string::npos);
  EXPECT_EQ(disabled.find("GPU GFLOPS"), std::string::npos);

  report["throughput"] = {
      {"memory_bytes", 4096},
      {"gpu_timestamp", {{"memory_gib_per_s", {{"mean", 4.125}}}}}};
  const std::string memory_only = format_table(Report::array({report}));
  EXPECT_NE(memory_only.find("GPU GiB/s"), std::string::npos);
  EXPECT_NE(memory_only.find("4.125"), std::string::npos);
  EXPECT_EQ(memory_only.find("GPU GFLOPS"), std::string::npos);

  report["throughput"] = {
      {"compute_flops", 8192},
      {"gpu_timestamp", {{"compute_gflops", {{"mean", 8.625}}}}}};
  const std::string compute_only = format_table(Report::array({report}));
  EXPECT_EQ(compute_only.find("GPU GiB/s"), std::string::npos);
  EXPECT_NE(compute_only.find("GPU GFLOPS"), std::string::npos);
  EXPECT_NE(compute_only.find("8.625"), std::string::npos);

  report["validation"] = {{"status", "PASS"}};
  report["throughput"]["memory_bytes"] = 0;
  report["throughput"]["gpu_timestamp"]["memory_gib_per_s"] = {{"mean", 0}};
  const std::string both = format_table(Report::array({report}));
  EXPECT_LT(both.find("Validation"), both.find("GPU GiB/s"));
  EXPECT_LT(both.find("GPU GiB/s"), both.find("GPU GFLOPS"));
  EXPECT_NE(both.find("0.000"), std::string::npos);
  EXPECT_NE(both.find("8.625"), std::string::npos);
}

TEST(BenchmarkReport, TableIncludesShaderThroughputBelowOperator) {
  Report report = {
      {"operator_name", "op"},
      {"args", Report::array()},
      {"throughput",
       {{"memory_bytes", 4096},
        {"compute_flops", 8192},
        {"gpu_timestamp",
         {{"memory_gib_per_s", {{"mean", 4.125}}},
          {"compute_gflops", {{"mean", 8.625}}}}}}},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "child"},
             {"gwg_size", {{"value", {1, 1, 1}}}},
             {"lwg_size", {{"value", {1, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1}, {"p90", 1}}},
             {"throughput",
              {{"memory_bytes", 1024},
               {"compute_flops", 2048},
               {"gpu_timestamp",
                {{"memory_gib_per_s", {{"mean", 1.125}}},
                 {"compute_gflops", {{"mean", 2.625}}}}}}}}})}};
  report.update(measurement_report(1, true, {2}, {}));
  std::istringstream stream(format_table(Report::array({report})));
  std::vector<std::string> lines;
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 4);
  EXPECT_EQ(lines[1], std::string(lines[0].size(), '-'));
  EXPECT_NE(lines[2].find("4.125"), std::string::npos);
  EXPECT_NE(lines[2].find("8.625"), std::string::npos);
  EXPECT_EQ(lines[3].find("  child"), 0);
  EXPECT_NE(lines[3].find("1.125"), std::string::npos);
  EXPECT_NE(lines[3].find("2.625"), std::string::npos);
}

TEST(
    BenchmarkReport,
    TableThroughputUsesNaForMissingEstimatesAndGpuTimestamps) {
  Report report = {
      {"operator_name", "no_timestamps"}, {"args", Report::array()}};
  report.update(measurement_report(1, true, {}, {2}));
  report["throughput"] = {
      {"memory_bytes", 4096},
      {"compute_flops", nullptr},
      {"gpu_timestamp",
       {{"memory_gib_per_s", nullptr}, {"compute_gflops", nullptr}}},
      {"wall_time", {{"memory_gib_per_s", {111.625}}}},
      {"errors", {{"compute", "No estimate"}}}};
  Report without_estimates = report;
  without_estimates["operator_name"] = "without_estimates";
  without_estimates.erase("throughput");
  const Report error = error_report({{"operator", "error_case"}}, "Bad input");
  std::istringstream stream(
      format_table(Report::array({report, without_estimates, error})));
  std::vector<std::string> lines;
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 8);
  EXPECT_NE(lines[0].find("GPU GiB/s"), std::string::npos);
  EXPECT_NE(lines[0].find("GPU GFLOPS"), std::string::npos);
  for (size_t i : {2u, 4u, 6u}) {
    EXPECT_EQ(lines[i].substr(lines[i].size() - 3), "N/A");
    EXPECT_EQ(lines[i].find("111.625"), std::string::npos);
  }
  EXPECT_NE(lines[7].find("ERROR error_case: Bad input"), std::string::npos);
}

TEST(BenchmarkReport, OperatorTensorMetadataIsInlineWithoutValues) {
  const Json arg_set = Json::array(
      {{{"tensor",
         {{"name", "x"},
          {"sizes", {2}},
          {"dtype", "float16"},
          {"data", {1, 2}}}}},
       {{"tensor",
         {{"name", "w"},
          {"sizes", {2}},
          {"storage", "constant"},
          {"fill", 0.5}}}},
       nullptr,
       {{"int", 32}},
       {{"tensor", {{"ref", "x"}}}},
       {{"tensor", {{"ref", "x"}, {"access", "write"}}}}});
  const Json spec = parse_configs({{"operator", "aten.linear.default"},
                                   {"args_sweep", Json::array({arg_set})}})
                        .at(0);
  const Report report =
      operator_report(spec, {{"x", "float32"}, {"w", "float32"}});
  EXPECT_FALSE(report.contains("tensors"));
  EXPECT_FALSE(report.contains("inputs"));
  EXPECT_FALSE(report.contains("outputs"));
  EXPECT_EQ(report.at("sweep_index"), 0);
  const auto& args = report.at("args");
  ASSERT_EQ(args.size(), 6);
  EXPECT_EQ(args.at(0).at("tensor").at("name"), "x");
  EXPECT_EQ(args.at(0).at("tensor").at("dtype"), "float32");
  EXPECT_EQ(args.at(0).at("tensor").at("requested_dtype"), "float16");
  EXPECT_EQ(args.at(0).at("tensor").at("access"), "read");
  EXPECT_EQ(args.at(1).at("tensor").at("type"), "constant");
  EXPECT_FALSE(args.at(1).at("tensor").contains("constant"));
  EXPECT_TRUE(args.at(2).is_null());
  EXPECT_EQ(args.at(3), (Report{{"int", 32}}));
  EXPECT_EQ(args.at(4).at("tensor"), args.at(0).at("tensor"));
  Report output = args.at(0).at("tensor");
  output["access"] = "write";
  EXPECT_EQ(args.at(5).at("tensor"), output);
  EXPECT_EQ(report.dump().find("\"data\""), std::string::npos);
  EXPECT_EQ(report.dump().find("\"fill\""), std::string::npos);
}

TEST(BenchmarkReport, InferredTensorRolesAndResolvedVariablesAreVisible) {
  const Json args = Json::array(
      {{{"tensor", {{"name", "x"}, {"sizes", {2}}, {"dtype", "v_DTYPE"}}}},
       {{"tensor", {{"name", "y"}, {"sizes", {2}}, {"dtype", "v_DTYPE"}}}}});
  const auto cases = parse_configs(
      {{"operator", "aten.relu.default"},
       {"sweep_variables", {{"v_DTYPE", {"float32", "float16"}}}},
       {"args_sweep", Json::array({args})}});
  const Report report = operator_report(
      cases[1],
      {{"x", "float16"}, {"y", "float16"}},
      {{"x", "read"}, {"y", "write"}});
  EXPECT_EQ(report.at("sweep_index"), 1);
  EXPECT_EQ(report.at("args_sweep_index"), 0);
  EXPECT_EQ(report.at("sweep_variables").at("v_DTYPE"), "float16");
  EXPECT_EQ(report.at("args")[0].at("tensor").at("access"), "read");
  EXPECT_EQ(report.at("args")[1].at("tensor").at("access"), "write");
}

TEST(BenchmarkReport, SweepIndexesIdentifyResultsAndErrors) {
  Report result = {
      {"case_name", "linear"},
      {"operator_name", "aten.linear.default"},
      {"sweep_index", 1},
      {"args", Report::array()}};
  result.update(measurement_report(1, true, {5}, {6}));
  const Report error = error_report(
      {{"name", "linear"},
       {"operator", "aten.linear.default"},
       {"sweep_index", 0}},
      "missing argument");
  EXPECT_EQ(error.at("sweep_index"), 0);
  const std::string table = format_table(Report::array({error, result}));
  EXPECT_NE(table.find("aten.linear.default [0]"), std::string::npos);
  EXPECT_NE(table.find("aten.linear.default [1]"), std::string::npos);
  EXPECT_NE(
      table.find("ERROR linear [0]: missing argument"), std::string::npos);
}

TEST(BenchmarkReport, TablePreservesCaseErrors) {
  const Report report = error_report(
      {{"shader", "missing"}, {"name", "bad\ncase"}}, "Unknown\tshader");
  const std::string table = format_table(Report::array({report}));
  EXPECT_NE(table.find("ERROR bad case: Unknown shader"), std::string::npos);
  EXPECT_NE(table.find("missing"), std::string::npos);
}

TEST(BenchmarkReport, TableShowsObservedClockRangeAcrossAllPasses) {
  Report report = {
      {"operator_name", "op"},
      {"args", Report::array()},
      {"gpu_batch_observed_gpu_clock_freq_MHz", {{300, 400}, {nullptr, 350}}},
      {"wall_time_batch_observed_gpu_clock_freq_MHz", {{200, 250}, {0, 100}}},
      {"shader_profile_batch_observed_gpu_clock_freq_MHz",
       {{600, 700}, {500, nullptr}}},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "child"},
             {"gwg_size", {{"value", {1, 1, 1}}}},
             {"lwg_size", {{"value", {1, 1, 1}}}},
             {"gpu_timestamp_shader_us", {{"mean", 1}, {"p90", 2}}}}})}};
  report.update(measurement_report(1, true, {2, 3, 4}, {}));
  Report missing = {{"operator_name", "missing"}, {"args", Report::array()}};
  missing.update(measurement_report(1, true, {2}, {}));
  const std::string table = format_table(Report::array({report, missing}));
  EXPECT_NE(table.find("Min GPU MHz"), std::string::npos);
  EXPECT_NE(table.find("Max GPU MHz"), std::string::npos);
  std::istringstream lines(table);
  for (std::string line; std::getline(lines, line);) {
    std::istringstream words(line);
    std::vector<std::string> fields;
    for (std::string word; words >> word;)
      fields.push_back(word);
    if (fields.empty())
      continue;
    if (fields[0] == "op") {
      EXPECT_EQ(fields[fields.size() - 2], "0");
      EXPECT_EQ(fields.back(), "700");
    } else if (fields[0] == "child") {
      EXPECT_EQ(fields[fields.size() - 2], "500");
      EXPECT_EQ(fields.back(), "700");
    } else if (fields[0] == "missing") {
      EXPECT_EQ(fields[fields.size() - 2], "N/A");
      EXPECT_EQ(fields.back(), "N/A");
    }
  }
}

TEST(BenchmarkReport, TableHandlesUnavailableAndFractionalClockReadings) {
  Report report = {
      {"shader_name", "copy"},
      {"gwg_size", {{"value", {1, 1, 1}}}},
      {"lwg_size", {{"value", {1, 1, 1}}}},
      {"bindings", Report::object()},
      {"gpu_batch_observed_gpu_clock_freq_MHz", {{nullptr, nullptr}}}};
  report.update(measurement_report(1, false, {}, {10, 20}));
  std::string table = format_table(Report::array({report}));
  EXPECT_NE(table.find("Min GPU MHz"), std::string::npos);
  EXPECT_EQ(table.substr(table.size() - 3), "N/A");
  report["wall_time_batch_observed_gpu_clock_freq_MHz"] = {{333.125, 444.75}};
  table = format_table(Report::array({report}));
  EXPECT_NE(table.find("333.125"), std::string::npos);
  EXPECT_NE(table.find("444.75"), std::string::npos);
}

TEST(BenchmarkReport, GpuClockPairsStayOnOneLine) {
  Report report = {
      {"shader_name", "copy"},
      {"gwg_size", {{"value", {1, 1, 1}}}},
      {"lwg_size", {{"value", {1, 1, 1}}}},
      {"bindings", Report::object()}};
  report.update(measurement_report(1, false, {2}, {3}));
  for (const char* field :
       {"gpu_batch_observed_gpu_clock_freq_MHz",
        "wall_time_batch_observed_gpu_clock_freq_MHz",
        "shader_profile_batch_observed_gpu_clock_freq_MHz"}) {
    Report clocked = report;
    clocked[field] = {{903, 950}, {nullptr, 800}};
    const std::string text = format_report(clocked);
    EXPECT_NE(text.find("[903, 950]"), std::string::npos);
    EXPECT_NE(text.find("[null, 800]"), std::string::npos);
    EXPECT_EQ(Report::parse(text), clocked);
  }
}

TEST(BenchmarkReport, DimensionArraysStayOnOneLine) {
  const Report report = Report::array(
      {{{"gwg_size", {{"method", "explicit"}, {"value", {8, 8, 1}}}},
        {"lwg_size", {{"method", "default_pick_lwg"}, {"value", {4, 4, 1}}}},
        {"bindings",
         {{"input", {{"type", "texture3d"}, {"sizes", {1, 3, 224, 224}}}}}},
        {"wall_time_batch_us", {12.5, 13.5}}}});
  const std::string text = format_report(report);
  EXPECT_NE(text.find("\"value\": [8, 8, 1]"), std::string::npos);
  EXPECT_NE(text.find("\"value\": [4, 4, 1]"), std::string::npos);
  EXPECT_NE(text.find("\"sizes\": [1, 3, 224, 224]"), std::string::npos);
  EXPECT_NE(text.find("\"wall_time_batch_us\": [\n"), std::string::npos);
  EXPECT_EQ(Report::parse(text), report);
}

TEST(BenchmarkReport, FormattingPreservesEscapedStringsAndFieldOrder) {
  const Report report = {
      {"case_name", "quoted \"name\"\nline"},
      {"labels", {"[", "\"", "\\", "\n"}},
      {"tensors", {{"weight", {{"sizes", {8, 32}}}}}},
      {"empty_object", Report::object()},
      {"empty_array", Report::array()}};
  const std::string text = format_report(report);
  EXPECT_EQ(Report::parse(text), report);
  EXPECT_NE(text.find("\"sizes\": [8, 32]"), std::string::npos);
}

TEST(BenchmarkReport, CondensedJsonUsesInputsAndSummarizesBatchTimings) {
  Report report = {
      {"operator_name", "aten.linear.default"},
      {"args",
       Report::array(
           {{{"tensor",
              {{"type", "buffer"},
               {"sizes", {4, 32}},
               {"dtype", "float32"},
               {"layout", "width"},
               {"access", "read"}}}},
            {{"tensor",
              {{"type", "constant"},
               {"sizes", {16, 32}},
               {"dtype", "float32"},
               {"layout", "width"}}}},
            {{"int_list", {1, 2}}},
            {{"tensor",
              {{"type", "buffer"},
               {"sizes", {4, 16}},
               {"dtype", "float32"},
               {"access", "write"}}}}})},
      {"benchmark_runs", 4},
      {"shader_breakdown",
       Report::array(
           {{{"shader_name", "linear"},
             {"gpu_timestamp_shader_us", latency_statistics({1, 2, 3})}}})},
      {"validation", {{"status", "PASS"}}}};
  report.update(
      measurement_report(10, true, {60, 20, 90, 30}, {80, 40, 20, 60}));
  const Report original = report;
  const Report condensed = Report::parse(format_condensed_report(report));
  EXPECT_FALSE(condensed.contains("args"));
  EXPECT_EQ(condensed.at("args_str"), "b[4,32]f32.width, c[16,32]f32, [1,2]");
  EXPECT_EQ(
      condensed.at("gpu_timestamp_operator_us"),
      (Report{{"min", 2}, {"median", 4.5}, {"max", 9}}));
  EXPECT_EQ(
      condensed.at("gpu_timestamp_batch_us"),
      (Report{{"min", 20}, {"median", 45}, {"max", 90}}));
  EXPECT_EQ(
      condensed.at("wall_time_operator_us"),
      (Report{{"min", 2}, {"median", 5}, {"max", 8}}));
  EXPECT_EQ(condensed.at("shader_breakdown"), report.at("shader_breakdown"));
  EXPECT_EQ(condensed.at("validation"), report.at("validation"));
  EXPECT_EQ(report, original);
}

TEST(BenchmarkReport, CondensedJsonSummarizesPassClockRanges) {
  const Report report = {
      {"gpu_batch_observed_gpu_clock_freq_MHz", {{903, 800}, {nullptr, 950}}},
      {"wall_time_batch_observed_gpu_clock_freq_MHz", {{0, 700}, {600, 650}}},
      {"shader_profile_batch_observed_gpu_clock_freq_MHz",
       {{nullptr, nullptr}}},
      {"gpu_timestamp_operator_us", Report::array()},
      {"gpu_timestamp_batch_us", Report::array()},
      {"wall_time_batch_us", {3, 1, 2}}};
  const Report condensed = Report::parse(format_condensed_report(report));
  EXPECT_EQ(
      condensed.at("gpu_batch_observed_gpu_clock_freq_MHz"),
      (Report{{"min", 800}, {"max", 950}}));
  EXPECT_EQ(
      condensed.at("wall_time_batch_observed_gpu_clock_freq_MHz"),
      (Report{{"min", 0}, {"max", 700}}));
  EXPECT_TRUE(condensed.at("shader_profile_batch_observed_gpu_clock_freq_MHz")
                  .is_null());
  EXPECT_TRUE(condensed.at("gpu_timestamp_operator_us").is_null());
  EXPECT_TRUE(condensed.at("gpu_timestamp_batch_us").is_null());
  EXPECT_EQ(
      condensed.at("wall_time_batch_us"),
      (Report{{"min", 1}, {"median", 2}, {"max", 3}}));
}

TEST(BenchmarkReport, CondensedJsonPreservesShaderStatisticsAndCaseGroups) {
  Report report = {
      {"shader_name", "copy"},
      {"bindings", {{"input", {{"sizes", {8}}, {"dtype", "float32"}}}}}};
  report.update(measurement_report(10, false, {10, 20, 30}, {30, 10, 20}));
  const Report error = error_report({{"operator", "bad"}}, "test error");
  const Report condensed =
      Report::parse(format_condensed_report(Report::array({report, error})));
  ASSERT_EQ(condensed.size(), 2);
  EXPECT_EQ(condensed[0].at("bindings"), report.at("bindings"));
  EXPECT_EQ(
      condensed[0].at("gpu_timestamp_shader_us"),
      report.at("gpu_timestamp_shader_us"));
  EXPECT_EQ(
      condensed[0].at("wall_time_shader_us"),
      (Report{{"min", 1}, {"median", 2}, {"max", 3}}));
  EXPECT_EQ(condensed[1].at("error"), "test error");
}

TEST(BenchmarkReport, StatisticsUseInterpolatedPercentiles) {
  const Report stats = latency_statistics({5, 1, 4, 2, 3});
  ASSERT_TRUE(stats.is_object());
  EXPECT_EQ(stats.at("min"), 1);
  EXPECT_EQ(stats.at("max"), 5);
  EXPECT_EQ(stats.at("mean"), 3);
  EXPECT_EQ(stats.at("p50"), 3);
  EXPECT_NEAR(stats.at("p90").get<double>(), 4.6, 1e-12);
  EXPECT_NEAR(stats.at("p95").get<double>(), 4.8, 1e-12);
  EXPECT_TRUE(latency_statistics({}).is_null());
}

TEST(BenchmarkReport, PerInvocationSamplesAreDividedByRepetitions) {
  const Report report =
      measurement_report(10, false, {10, 20, 30}, {20, 40, 60});
  ASSERT_TRUE(report.contains("gpu_timestamp_shader_us"));
  EXPECT_EQ(report.at("gpu_timestamp_shader_us").at("mean"), 2);
  EXPECT_EQ(report.at("gpu_timestamp_batch_us"), Report::array({10, 20, 30}));
  EXPECT_EQ(report.at("wall_time_shader_us"), Report::array({2, 4, 6}));
  EXPECT_EQ(report.at("wall_time_batch_us"), Report::array({20, 40, 60}));
  EXPECT_THROW(measurement_report(0, false, {}, {1}), std::invalid_argument);
}

TEST(BenchmarkReport, OperatorGpuSamplesPreserveRunOrderAndBatchAverages) {
  Report report = {{"operator_name", "op"}, {"args", Report::array()}};
  report.update(measurement_report(10, true, {30, 10, 20}, {60, 20, 40}));
  EXPECT_EQ(report.at("gpu_timestamp_batch_us"), Report::array({30, 10, 20}));
  EXPECT_EQ(report.at("gpu_timestamp_operator_us"), Report::array({3, 1, 2}));
  EXPECT_EQ(report.at("wall_time_operator_us"), Report::array({6, 2, 4}));
  const std::string table = format_table(Report::array({report}));
  EXPECT_NE(table.find("2.000"), std::string::npos);
  EXPECT_NE(table.find("2.800"), std::string::npos);
}

TEST(BenchmarkReport, UnavailableTimestampsAndOperatorsUseExplicitLabels) {
  const Report report = measurement_report(2, true, {}, {4, 8});
  ASSERT_TRUE(report.contains("gpu_timestamp_operator_us"));
  EXPECT_EQ(report.at("gpu_timestamp_operator_us"), Report::array());
  EXPECT_EQ(report.at("gpu_timestamp_batch_us"), Report::array());
  EXPECT_EQ(report.at("wall_time_operator_us"), Report::array({2, 4}));
  EXPECT_FALSE(report.contains("gpu_timestamp_shader_us"));
}

TEST(BenchmarkReport, ShaderDetailsHaveStableOrderAndNoTensorValues) {
  const Json spec =
      parse_configs(
          {{"name", "copy"},
           {"shader", "buffer_to_buffer_float"},
           {"gwg_size", {8, 1, 1}},
           {"lwg_size", "default_pick_lwg"},
           {"bindings",
            Json::array(
                {{{"buffer",
                   {{"name", "output"}, {"sizes", {8}}, {"access", "write"}}}},
                 {{"buffer",
                   {{"name", "input"},
                    {"sizes", {8}},
                    {"data", {1, 2, 3, 4, 5, 6, 7, 8}}}}},
                 {{"meta_ubo", "output"}, {"field", "numel"}}})},
           {"outputs", {"output"}},
           {"dump_outputs", true}})
          .at(0);
  const Report report = shader_report(
      spec,
      {8, 1, 1},
      {8, 1, 1},
      {{"input", "float32"}, {"output", "float32"}},
      Report::object());
  std::vector<std::string> keys;
  for (const auto& item : report.items()) {
    keys.push_back(item.key());
  }
  EXPECT_EQ(
      keys,
      (std::vector<std::string>{
          "case_name",
          "shader_name",
          "gwg_size",
          "lwg_size",
          "bindings",
          "specialization_constants"}));
  ASSERT_TRUE(report.contains("bindings"));
  EXPECT_EQ(report.at("gwg_size").at("method"), "explicit");
  EXPECT_EQ(report.at("lwg_size").at("method"), "default_pick_lwg");
  EXPECT_EQ(report.at("bindings").at("input").at("type"), "buffer");
  EXPECT_EQ(report.at("bindings").at("input").at("binding_index"), 1);
  EXPECT_FALSE(report.at("bindings").at("input").contains("data"));
  EXPECT_FALSE(report.contains("outputs"));
}

TEST(BenchmarkReport, SharedBindingsAndPrecisionPromotionRemainVisible) {
  const Json spec =
      parse_configs({{"shader", "buffer_to_buffer_float"},
                     {"gwg_size", {8, 1, 1}},
                     {"lwg_size", {8, 1, 1}},
                     {"bindings",
                      Json::array(
                          {{{"buffer",
                             {{"name", "shared"},
                              {"sizes", {8}},
                              {"dtype", "float16"},
                              {"access", "write"}}}},
                           {{"buffer", {{"ref", "shared"}}}},
                           {{"meta_ubo", "shared"}, {"field", "numel"}}})}})
          .at(0);
  const Report report = shader_report(
      spec, {8, 1, 1}, {8, 1, 1}, {{"shared", "float32"}}, Report::object());
  const auto& bindings = report.at("bindings");
  ASSERT_EQ(bindings.size(), 3);
  EXPECT_EQ(bindings.at("shared").at("dtype"), "float32");
  EXPECT_EQ(bindings.at("shared").at("requested_dtype"), "float16");
  std::vector<uint32_t> indices;
  for (const auto& info : bindings) {
    indices.push_back(info.at("binding_index"));
    if (info.at("binding_index") == 1) {
      EXPECT_EQ(info.at("ref"), "shared");
    }
  }
  EXPECT_EQ(indices, (std::vector<uint32_t>{0, 1, 2}));
}

} // namespace
} // namespace vkcompute::shader_profiling
