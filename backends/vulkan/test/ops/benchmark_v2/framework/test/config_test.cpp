// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

Json config() {
  return {
      {"shader", "buffer_to_buffer_float"},
      {"gwg_size", {8, 1, 1}},
      {"lwg_size", {8, 1, 1}},
      {"bindings",
       {{{"buffer", {{"name", "out"}, {"sizes", {8}}, {"access", "write"}}}}}}};
}

Json operator_config() {
  const Json args = Json::array(
      {{{"tensor", {{"name", "x"}, {"sizes", {2, 8}}}}},
       {{"tensor", {{"name", "y"}, {"sizes", {2, 4}}}}}});
  return {
      {"operator", "aten.linear.default"}, {"args_sweep", Json::array({args})}};
}

TEST(ShaderProfilingConfig, AcceptsSingleAndMultipleConfigs) {
  EXPECT_EQ(parse_configs(config()).size(), 1);
  EXPECT_EQ(parse_configs(Json::array({config(), config()})).size(), 2);
  EXPECT_THROW(parse_configs(Json::array()), std::invalid_argument);
}

TEST(ShaderProfilingConfig, ThroughputFlagsMustBeBooleans) {
  for (const char* key :
       {"show_memory_throughput", "show_compute_throughput"}) {
    Json spec = config();
    spec[key] = true;
    EXPECT_EQ(parse_configs(spec)[0].at(key), true);
    spec[key] = false;
    EXPECT_EQ(parse_configs(spec)[0].at(key), false);
    for (const Json& invalid : {Json("true"), Json(1), Json(), Json::array()}) {
      spec[key] = invalid;
      EXPECT_THROW(parse_configs(spec), std::invalid_argument);
    }
  }
}

TEST(ShaderProfilingConfig, ExpandsShaderNamesWithSharedIndependentInputs) {
  Json spec = config();
  spec.erase("shader");
  spec["shader_sweep"] = {"shader_b", "shader_a", "shader_b"};
  spec["name"] = "shared inputs";
  spec["target_ms"] = 7;
  spec["reference"] = "copy";
  auto parsed = parse_configs(spec);
  ASSERT_EQ(parsed.size(), 3);
  for (size_t i = 0; i < parsed.size(); ++i) {
    EXPECT_EQ(parsed[i].at("shader"), spec.at("shader_sweep")[i]);
    EXPECT_EQ(parsed[i].at("sweep_index"), i);
    EXPECT_EQ(parsed[i].at("name"), "shared inputs");
    EXPECT_EQ(parsed[i].at("target_ms"), 7);
    EXPECT_EQ(parsed[i].at("reference"), "copy");
    EXPECT_FALSE(parsed[i].contains("shader_sweep"));
    EXPECT_EQ(parsed[i].at("bindings"), parsed[0].at("bindings"));
    EXPECT_EQ(parsed[i].at("tensors"), parsed[0].at("tensors"));
  }
  parsed[0]["tensors"]["out"]["sizes"] = {16};
  EXPECT_EQ(parsed[1].at("tensors").at("out").at("sizes"), Json::array({8}));
}

TEST(ShaderProfilingConfig, CombinesShaderAndLocalWorkgroupListsInOrder) {
  Json spec = config();
  spec.erase("shader");
  spec["shader_sweep"] = {"shader_a", "shader_b"};
  spec["lwg_size"] = {{8, 1, 1}, {10, 2, 1}};
  const auto parsed = parse_configs(spec);
  ASSERT_EQ(parsed.size(), 4);
  for (size_t i = 0; i < parsed.size(); ++i) {
    EXPECT_EQ(parsed[i].at("shader"), spec.at("shader_sweep")[i / 2]);
    EXPECT_EQ(parsed[i].at("lwg_size"), spec.at("lwg_size")[i % 2]);
    EXPECT_EQ(parsed[i].at("sweep_index"), i);
  }
}

TEST(ShaderProfilingConfig, CombinesShaderListAndCartesianWorkgroupRange) {
  Json spec = config();
  spec.erase("shader");
  spec["shader_sweep"] = {"shader_a", "shader_b"};
  spec.erase("lwg_size");
  spec["lwg_size_sweep"] = {{"x", {8, 12, 4}}, {"y", {1, 2, 1}}};
  const auto parsed = parse_configs(spec);
  ASSERT_EQ(parsed.size(), 8);
  const Json sizes = {{8, 1, 1}, {8, 2, 1}, {12, 1, 1}, {12, 2, 1}};
  for (size_t i = 0; i < parsed.size(); ++i) {
    EXPECT_EQ(parsed[i].at("shader"), spec.at("shader_sweep")[i / 4]);
    EXPECT_EQ(parsed[i].at("lwg_size"), sizes[i % 4]);
    EXPECT_EQ(parsed[i].at("sweep_index"), i);
    EXPECT_FALSE(parsed[i].contains("lwg_size_sweep"));
  }
}

TEST(ShaderProfilingConfig, RejectsMalformedOrAmbiguousShaderSweeps) {
  Json spec = config();
  spec["shader_sweep"] = {"shader_a"};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  Json op = operator_config();
  op["shader_sweep"] = {"shader_a"};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  spec.erase("shader");
  for (const Json& invalid :
       {Json::array(),
        Json("shader_a"),
        Json::object(),
        Json::array({""}),
        Json::array({"shader_a", 1}),
        Json::array({nullptr})}) {
    spec["shader_sweep"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
}

TEST(ShaderProfilingConfig, CapsCombinedShaderAndWorkgroupSweep) {
  Json spec = config();
  spec.erase("shader");
  spec["shader_sweep"] = {"shader_a", "shader_b"};
  spec.erase("lwg_size");
  spec["lwg_size_sweep"] = {{"x", {1, 32769, 1}}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec["shader_sweep"] = std::vector<std::string>(65537, "shader_a");
  spec.erase("lwg_size_sweep");
  spec["lwg_size"] = {8, 1, 1};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, PreservesMixedDefinitionOrderWithShaderSweeps) {
  Json sweep = config();
  sweep.erase("shader");
  sweep["shader_sweep"] = {"shader_a", "shader_b"};
  const auto parsed = parse_configs(
      Json::array({config(), sweep, operator_config(), config()}));
  ASSERT_EQ(parsed.size(), 5);
  EXPECT_EQ(parsed[0].at("shader"), "buffer_to_buffer_float");
  EXPECT_EQ(parsed[1].at("shader"), "shader_a");
  EXPECT_EQ(parsed[2].at("shader"), "shader_b");
  EXPECT_TRUE(parsed[3].contains("operator"));
  EXPECT_EQ(parsed[4].at("shader"), "buffer_to_buffer_float");
  EXPECT_FALSE(parsed[0].contains("sweep_index"));
  EXPECT_FALSE(parsed[4].contains("sweep_index"));
}

TEST(ShaderProfilingConfig, RejectsInvalidGridAndTensorSizes) {
  Json spec = config();
  for (const Json& invalid :
       {Json{0, 1, 1}, Json{-1, 1, 1}, Json{1.5, 1, 1}, Json{1, 1}}) {
    spec["gwg_size"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
  EXPECT_EQ(tensor_numel({2, 3, 4}), 24);
  EXPECT_THROW(tensor_numel({2, 0}), std::invalid_argument);
  EXPECT_THROW(tensor_numel({2147483647, 2}), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsUnknownTensorAndInvalidOptions) {
  Json spec = config();
  spec["bindings"][0] = {{"buffer", {{"ref", "missing"}}}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["benchmark_runs"] = 0;
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, PacksTypedInputsAndRejectsTruncation) {
  auto bytes = make_data({{"dtype", "int32"}, {"data", {-7, 42}}}, 2);
  int32_t values[2];
  std::memcpy(values, bytes.data(), sizeof(values));
  EXPECT_EQ(values[0], -7);
  EXPECT_EQ(values[1], 42);
  EXPECT_THROW(
      make_data({{"dtype", "int8"}, {"data", {128}}}, 1),
      std::invalid_argument);
  EXPECT_THROW(make_data({{"data", {1, 2}}}, 3), std::invalid_argument);
}

TEST(ShaderProfilingConfig, StructuredDataInitializersAreTypedAndReproducible) {
  const auto constants =
      make_data({{"dtype", "int8"}, {"data", {{"fill", -7}}}}, 4);
  EXPECT_EQ(constants, std::vector<uint8_t>(4, 249));
  const Json spec = {
      {"dtype", "float32"}, {"data", {{"rand_uniform_float", {0, 1}}}}};
  const auto random = make_data(spec, 64, 7);
  EXPECT_EQ(random, make_data(spec, 64, 7));
  EXPECT_NE(random, make_data(spec, 64, 8));
  Json half = spec;
  half["dtype"] = "float16";
  EXPECT_EQ(random, make_data(half, 64, 7));
  float values[64];
  std::memcpy(values, random.data(), sizeof(values));
  for (float value : values) {
    EXPECT_GE(value, 0);
    EXPECT_LE(value, 1);
  }
  Json seeded = spec;
  seeded["data"]["seed"] = 42;
  EXPECT_EQ(make_data(seeded, 64, 1), make_data(seeded, 64, 2));
}

TEST(ShaderProfilingConfig, UniformIntegerDataIsInclusiveAndReproducible) {
  const Json spec = {
      {"dtype", "int8"}, {"data", {{"rand_uniform_int", {-4, 4}}}}};
  const auto data = make_data(spec, 512, 7);
  EXPECT_EQ(data, make_data(spec, 512, 7));
  EXPECT_NE(data, make_data(spec, 512, 8));
  int8_t values[512];
  std::memcpy(values, data.data(), sizeof(values));
  bool saw_min = false;
  bool saw_max = false;
  for (int8_t value : values) {
    EXPECT_GE(value, -4);
    EXPECT_LE(value, 4);
    saw_min |= value == -4;
    saw_max |= value == 4;
  }
  EXPECT_TRUE(saw_min);
  EXPECT_TRUE(saw_max);
  const auto large = make_data(
      {{"dtype", "uint32"},
       {"data", {{"rand_uniform_int", {0, 2147483648LL}}}}},
      256);
  uint32_t wide[256];
  std::memcpy(wide, large.data(), sizeof(wide));
  for (uint32_t value : wide) {
    EXPECT_LE(value, 2147483648u);
  }
  EXPECT_THROW(
      make_data({{"data", {{"rand_uniform_int", {0, 1.5}}}}}, 4),
      std::invalid_argument);
  EXPECT_THROW(
      make_data(
          {{"dtype", "int8"}, {"data", {{"rand_uniform_int", {-129, 127}}}}},
          4),
      std::invalid_argument);
  EXPECT_THROW(
      make_data(
          {{"dtype", "int32"}, {"data", {{"rand_uniform_int", {0, 1.5}}}}}, 4),
      std::invalid_argument);
}

TEST(
    ShaderProfilingConfig,
    RowSumsTrackRandomWeightsAndResolveForwardReferences) {
  const Json tensors = {
      {"a_sums",
       {{"sizes", {2}},
        {"dtype", "int32"},
        {"data", {{"row_sum", "z_weights"}}}}},
      {"z_weights",
       {{"sizes", {2, 3}},
        {"dtype", "int8"},
        {"data", {{"rand_uniform_int", {-4, 4}}}}}}};
  const auto data = make_tensor_data(tensors);
  int8_t weights[6];
  int32_t sums[2];
  std::memcpy(weights, data.at("z_weights").data(), sizeof(weights));
  std::memcpy(sums, data.at("a_sums").data(), sizeof(sums));
  EXPECT_EQ(sums[0], weights[0] + weights[1] + weights[2]);
  EXPECT_EQ(sums[1], weights[3] + weights[4] + weights[5]);
  EXPECT_EQ(data, make_tensor_data(tensors));
  Json invalid = tensors;
  invalid["a_sums"]["data"]["row_sum"] = "missing";
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  invalid["a_sums"]["data"]["row_sum"] = "a_sums";
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  invalid = tensors;
  invalid["a_sums"]["sizes"] = {1};
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsMalformedDataInitializers) {
  for (const Json& data :
       {Json::object(),
        Json{{"unknown", 0}},
        Json{{"constant", 0}},
        Json{{"fill", "bad"}},
        Json{{"rand_uniform_float", {1, 0}}},
        Json{{"rand_uniform_float", {0}}},
        Json{{"rand_uniform_float", {0, 1}}, {"seed", -1}},
        Json{{"fill", 0}, {"rand_uniform_float", {0, 1}}}}) {
    EXPECT_THROW(make_data({{"data", data}}, 4), std::invalid_argument);
  }
  EXPECT_THROW(
      make_data(
          {{"dtype", "int8"}, {"data", {{"rand_uniform_float", {0, 1}}}}}, 4),
      std::invalid_argument);
  EXPECT_THROW(
      make_data({{"dtype", "int8"}, {"data", {{"fill", 128}}}}, 4),
      std::invalid_argument);
  EXPECT_THROW(
      make_data({{"fill", 1}, {"data", {{"fill", 0}}}}, 4),
      std::invalid_argument);
}

TEST(ShaderProfilingConfig, BoundsAdaptiveDispatchCount) {
  EXPECT_EQ(choose_dispatch_count(10, 1000, 4096), 100);
  EXPECT_EQ(choose_dispatch_count(2000, 1000, 4096), 1);
  EXPECT_EQ(choose_dispatch_count(0, 1000, 128), 128);
}

TEST(ShaderProfilingConfig, AcceptsOperatorAndMixedCases) {
  Json op = operator_config();
  op["args_sweep"][0].push_back(nullptr);
  op["args_sweep"][0].push_back({{"int_list", {1, 2}}});
  EXPECT_EQ(parse_configs(op).size(), 1);
  EXPECT_EQ(parse_configs(Json::array({config(), op})).size(), 2);
}

TEST(ShaderProfilingConfig, ConstantStorageIsOperatorOnlyAndReplacesTheFlag) {
  Json op = operator_config();
  op["args_sweep"][0][0]["tensor"]["storage"] = "constant";
  op["args_sweep"][0][0]["tensor"]["sizes"] = {1, 2, 3, 4, 5};
  const Json parsed = parse_configs(op).at(0);
  EXPECT_EQ(parsed.at("tensors").at("x").at("storage"), "constant");
  EXPECT_FALSE(parsed.at("tensors").at("x").contains("constant"));
  op = operator_config();
  op["args_sweep"][0][0]["tensor"]["constant"] = true;
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  Json shader = config();
  shader["bindings"][0]["buffer"]["storage"] = "constant";
  EXPECT_THROW(parse_configs(shader), std::invalid_argument);
}

TEST(ShaderProfilingConfig, CartesianVariablesPreserveTypesAndTemplateOrder) {
  Json op = operator_config();
  op["sweep_variables"] = {
      {"v_DTYPE", {"float32", "float16"}},
      {"v_STORAGE", {"buffer", "texture3d"}},
      {"v_M", {4}}};
  for (auto& arg : op["args_sweep"][0]) {
    arg["tensor"]["dtype"] = "v_DTYPE";
    arg["tensor"]["storage"] = "v_STORAGE";
    arg["tensor"]["sizes"][0] = "v_M";
    arg["tensor"]["data"] = {{"rand_uniform_float", {0, 1}}};
  }
  op["args_sweep"].push_back(op["args_sweep"][0]);
  const auto parsed = parse_configs(op);
  ASSERT_EQ(parsed.size(), 8);
  for (size_t i = 0; i < parsed.size(); ++i) {
    EXPECT_EQ(parsed[i].at("sweep_index"), i);
    EXPECT_EQ(parsed[i].at("args_sweep_index"), i / 4);
    EXPECT_EQ(parsed[i].at("tensors").at("x").at("sizes")[0], 4);
    EXPECT_EQ(
        parsed[i].at("tensors").at("x").at("dtype"),
        i % 4 < 2 ? "float32" : "float16");
    EXPECT_EQ(
        parsed[i].at("tensors").at("x").at("storage"),
        i % 2 == 0 ? "buffer" : "texture3d");
    EXPECT_EQ(parsed[i].at("sweep_variables").at("v_M"), 4);
    EXPECT_FALSE(parsed[i].at("args")[1].contains("access"));
  }
}

TEST(ShaderProfilingConfig, RejectsInvalidOrUnboundSweepVariables) {
  for (const Json& variables :
       {Json::array(),
        Json{{"v_DTYPE", Json::array()}},
        Json{{"v_DTYPE", "float32"}},
        Json{{"DTYPE", {"float32"}}}}) {
    Json op = operator_config();
    op["sweep_variables"] = variables;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  Json op = operator_config();
  op["args_sweep"][0][0]["tensor"]["dtype"] = "v_MISSING";
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, NormalizesExplicitOperatorRoutes) {
  Json op = operator_config();
  EXPECT_FALSE(parse_configs(op).at(0).contains("route"));
  op["route"] = "direct";
  EXPECT_EQ(
      parse_configs(op).at(0).at("route"),
      (Json{{"name", "direct"}, {"options", Json::object()}}));
  op["route"] = {{"name", "tiled"}};
  EXPECT_EQ(
      parse_configs(op).at(0).at("route"),
      (Json{{"name", "tiled"}, {"options", Json::object()}}));
  op["route"]["options"] = {{"tile", {2, 4}}, {"custom_option", true}};
  EXPECT_EQ(parse_configs(op).at(0).at("route"), op.at("route"));
}

TEST(ShaderProfilingConfig, ExpandsRouteSweepInsideArgumentVariables) {
  Json op = operator_config();
  op["sweep_variables"] = {{"v_DTYPE", {"float32", "float16"}}};
  op["args_sweep"][0][0]["tensor"]["dtype"] = "v_DTYPE";
  op["args_sweep"][0][0]["tensor"]["data"] = {{"rand_uniform_float", {-1, 1}}};
  op["args_sweep"].push_back(op["args_sweep"][0]);
  op["args_sweep"][1][0]["tensor"]["sizes"] = {4, 8};
  op["route_sweep"] = Json::array(
      {"auto", {{"name", "tiled"}, {"options", {{"tile_rows", 2}}}}});
  auto cases = parse_configs(op);
  ASSERT_EQ(cases.size(), 8);
  for (size_t i = 0; i < cases.size(); ++i) {
    EXPECT_EQ(cases[i].at("sweep_index"), i);
    EXPECT_EQ(cases[i].at("args_sweep_index"), i / 4);
    EXPECT_EQ(
        cases[i].at("sweep_variables").at("v_DTYPE"),
        i % 4 < 2 ? "float32" : "float16");
    EXPECT_EQ(cases[i].at("route").at("name"), i % 2 == 0 ? "auto" : "tiled");
    EXPECT_FALSE(cases[i].contains("route_sweep"));
    EXPECT_EQ(cases[i].at("tensors").at("x").at("sizes")[0], i < 4 ? 2 : 4);
  }
  EXPECT_EQ(
      make_tensor_data(cases[0].at("tensors")),
      make_tensor_data(cases[1].at("tensors")));
  cases[0]["tensors"]["x"]["sizes"] = {8, 8};
  EXPECT_EQ(cases[1].at("tensors").at("x").at("sizes"), (Json{2, 8}));
}

TEST(ShaderProfilingConfig, SubstitutesRouteNamesAndTypedOptions) {
  Json op = operator_config();
  op["sweep_variables"] = {
      {"v_ROUTE", {"direct", "tiled"}}, {"v_TILE", {{2, 4}, {4, 8}}}};
  op["route"] = {{"name", "v_ROUTE"}, {"options", {{"tile", "v_TILE"}}}};
  const auto cases = parse_configs(op);
  ASSERT_EQ(cases.size(), 4);
  EXPECT_EQ(cases[0].at("route").at("name"), "direct");
  EXPECT_EQ(cases[2].at("route").at("name"), "tiled");
  EXPECT_EQ(cases[0].at("route").at("options").at("tile"), (Json{2, 4}));
  EXPECT_EQ(cases[1].at("route").at("options").at("tile"), (Json{4, 8}));
  op.erase("route");
  op["route_sweep"] =
      Json::array({"v_ROUTE", {{"name", "auto"}, {"options", "v_OPTIONS"}}});
  op["sweep_variables"] = {
      {"v_ROUTE", {"direct", "tiled"}},
      {"v_OPTIONS", Json::array({Json{{"tile_rows", 2}}})}};
  const auto swept = parse_configs(op);
  ASSERT_EQ(swept.size(), 4);
  EXPECT_EQ(swept[0].at("route").at("name"), "direct");
  EXPECT_EQ(swept[2].at("route").at("name"), "tiled");
  EXPECT_EQ(swept[1].at("route").at("options").at("tile_rows"), 2);
}

TEST(ShaderProfilingConfig, RejectsMalformedOrShaderRouteSelections) {
  for (const Json& invalid :
       {Json(nullptr),
        Json(1),
        Json(false),
        Json(""),
        Json::array({"direct"}),
        Json::object(),
        Json{{"options", Json::object()}},
        Json{{"name", 1}},
        Json{{"name", ""}},
        Json{{"name", "direct"}, {"unknown", 1}},
        Json{{"name", "direct"}, {"options", nullptr}},
        Json{{"name", "direct"}, {"options", Json::array()}},
        Json{{"name", "direct"}, {"options", 2}}}) {
    Json op = operator_config();
    op["route"] = invalid;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
    op.erase("route");
    op["route_sweep"] = Json::array({"auto", invalid});
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  for (const Json& invalid :
       {Json(nullptr), Json("direct"), Json::object(), Json::array()}) {
    Json op = operator_config();
    op["route_sweep"] = invalid;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  Json op = operator_config();
  op["route"] = "direct";
  op["route_sweep"] = {"auto"};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  for (const char* key : {"route", "route_sweep"}) {
    Json shader = config();
    shader[key] =
        key == std::string("route") ? Json("direct") : Json::array({"direct"});
    EXPECT_THROW(parse_configs(shader), std::invalid_argument);
  }
  op = operator_config();
  op["route"] = {{"name", "v_MISSING"}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op["sweep_variables"] = {{"v_MISSING", {1}}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsLegacyRouteSelectionKeys) {
  for (const char* key : {"implementation", "implementation_sweep"}) {
    for (Json spec : {operator_config(), config()}) {
      spec[key] = key == std::string("implementation")
          ? Json("direct")
          : Json::array({"direct"});
      EXPECT_THROW(parse_configs(spec), std::invalid_argument);
    }
  }
}

TEST(ShaderProfilingConfig, CapsCombinedRouteCartesianExpansion) {
  Json op = operator_config();
  op["args_sweep"] = Json::array_t(32, Json::array({{{"int", 1}}}));
  op["sweep_variables"] = {{"v_VALUE", Json::array_t(128, 1)}};
  op["route_sweep"] = Json::array_t(32, "auto");
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op["args_sweep"] = Json::array_t(17, Json::array({{{"int", 1}}}));
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(
    ShaderProfilingConfig,
    ExpandsSweepsInOrderWithIndependentTensorDefinitions) {
  Json op = operator_config();
  op["name"] = "linear";
  op["target_ms"] = 7;
  Json half_args = op.at("args_sweep").at(0);
  half_args[0]["tensor"]["dtype"] = "float16";
  half_args[0]["tensor"]["sizes"] = {4, 8};
  half_args[1]["tensor"]["dtype"] = "float16";
  half_args[1]["tensor"]["sizes"] = {4, 4};
  op["args_sweep"].push_back(half_args);
  const auto parsed = parse_configs(Json::array({config(), op, config()}));
  ASSERT_EQ(parsed.size(), 4);
  EXPECT_TRUE(parsed[0].contains("shader"));
  EXPECT_TRUE(parsed[3].contains("shader"));
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(parsed[i + 1].at("name"), "linear");
    EXPECT_EQ(parsed[i + 1].at("target_ms"), 7);
    EXPECT_EQ(parsed[i + 1].at("sweep_index"), i);
    EXPECT_FALSE(parsed[i + 1].contains("args_sweep"));
    EXPECT_EQ(parsed[i + 1].at("tensors").size(), 2);
  }
  EXPECT_EQ(parsed[1].at("tensors").at("x").at("sizes"), Json::array({2, 8}));
  EXPECT_EQ(parsed[2].at("tensors").at("x").at("sizes"), Json::array({4, 8}));
  EXPECT_EQ(
      parsed[1].at("tensors").at("x").value("dtype", "float32"), "float32");
  EXPECT_EQ(parsed[2].at("tensors").at("x").at("dtype"), "float16");
  op["args_sweep"][1][0]["tensor"] = {{"ref", "x"}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsEmptyAndMalformedSweeps) {
  for (const Json& invalid :
       {Json::array(),
        Json::object(),
        Json(1),
        Json::array({Json::object()}),
        Json::array({Json::array()})}) {
    Json op = operator_config();
    op["args_sweep"] = invalid;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  Json op = operator_config();
  op.erase("args_sweep");
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  Json shader = config();
  shader["args_sweep"] = Json::array({Json::array()});
  EXPECT_THROW(parse_configs(shader), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsAmbiguousCasesAndOperatorArguments) {
  Json op = operator_config();
  op["shader"] = "buffer_to_buffer_float";
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op.erase("shader");
  op["args_sweep"][0][0]["tensor"] = {{"ref", "missing"}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op["args_sweep"][0] = Json::array({{{"int", 1.5}}});
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, SupportsPackedInt8Activations) {
  Json op = operator_config();
  op["operator"] = "et_vk.q8ta_conv2d.default";
  op["args_sweep"][0] = Json::array(
      {{{"tensor",
         {{"sizes", {1, 4, 8, 8}}, {"dtype", "int8x4"}, {"layout", "4w4c"}}}},
       {{"tensor",
         {{"sizes", {1, 8, 8, 8}},
          {"dtype", "int8x4"},
          {"layout", "4w4c"},
          {"access", "write"}}}}});
  EXPECT_EQ(parse_configs(op).size(), 1);
  EXPECT_EQ(
      make_data({{"dtype", "int8x4"}, {"fill", -7}}, 4),
      std::vector<uint8_t>(4, 249));
}

TEST(ShaderProfilingConfig, InlineOperatorTensorsPreserveReferencesAndData) {
  Json op = operator_config();
  op["args_sweep"][0] = Json::array(
      {{{"tensor", {{"ref", "state"}}}},
       {{"tensor",
         {{"name", "weight"},
          {"sizes", {2}},
          {"dtype", "int32"},
          {"storage", "constant"},
          {"data", {-7, 42}}}}},
       nullptr,
       {{"tensor", {{"sizes", {2}}, {"fill", 0.5}}}},
       {{"tensor",
         {{"name", "state"},
          {"sizes", {2}},
          {"storage", "texture3d"},
          {"access", "write"}}}},
       {{"tensor", {{"ref", "arg_3"}, {"access", "read_write"}}}}});
  const Json parsed = parse_configs(op).at(0);
  EXPECT_EQ(parsed.at("tensors").size(), 3);
  EXPECT_FALSE(parsed.contains("inputs"));
  EXPECT_FALSE(parsed.contains("outputs"));
  EXPECT_EQ(parsed.at("args").size(), 6);
  EXPECT_EQ(parsed.at("args").at(0).at("tensor"), "state");
  EXPECT_TRUE(parsed.at("args").at(2).is_null());
  EXPECT_EQ(parsed.at("args").at(4).at("tensor"), "state");
  EXPECT_EQ(parsed.at("args").at(4).at("access"), "write");
  EXPECT_EQ(parsed.at("args").at(5).at("tensor"), "arg_3");
  EXPECT_EQ(parsed.at("args").at(5).at("access"), "read_write");
  EXPECT_EQ(
      parsed.at("tensors").at("weight").at("data"), Json::array({-7, 42}));
  EXPECT_EQ(parsed.at("tensors").at("weight").at("storage"), "constant");
  EXPECT_EQ(parsed.at("tensors").at("arg_3").at("fill"), 0.5);
}

TEST(
    ShaderProfilingConfig,
    RejectsExternalOperatorTensorsAndInvalidDefinitions) {
  Json op = operator_config();
  op["tensors"] = {{"old", {{"sizes", {8}}}}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  for (const Json& invalid :
       {Json("x"),
        Json{{"name", "x"}},
        Json{{"sizes", {0}}},
        Json{{"name", ""}, {"sizes", {2}}},
        Json{{"name", "y"}, {"sizes", {2}}},
        Json{{"ref", "missing"}},
        Json{{"ref", "y"}, {"sizes", {2}}},
        Json{{"sizes", {2}}, {"dtype", "unsupported"}}}) {
    op = operator_config();
    op["args_sweep"][0][0]["tensor"] = invalid;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  op = operator_config();
  op["args_sweep"][0][1]["tensor"]["storage"] = "constant";
  op["args_sweep"][0][1]["tensor"]["access"] = "write";
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op["args_sweep"][0] = Json::array({{{"float", "bad"}}});
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
  op["args_sweep"][0] = Json::array();
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsOldOperatorListsAndInvalidAccess) {
  for (const char* key : {"args", "inputs", "outputs"}) {
    Json op = operator_config();
    op[key] = Json::array();
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  for (const Json& access : {Json("unknown"), Json(1), Json(false)}) {
    Json op = operator_config();
    op["args_sweep"][0][1]["tensor"]["access"] = access;
    EXPECT_THROW(parse_configs(op), std::invalid_argument);
  }
  Json op = operator_config();
  op["args_sweep"][0][0]["tensor"]["storage"] = "constant";
  op["args_sweep"][0][1]["tensor"] = {{"ref", "x"}, {"access", "write"}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, ReadWriteTensorKeepsOneArgument) {
  Json op = operator_config();
  op["operator"] = "test.inplace.default";
  op["args_sweep"][0] =
      Json::array({{{"tensor", {{"sizes", {8}}, {"access", "read_write"}}}}});
  const Json parsed = parse_configs(op).at(0);
  EXPECT_EQ(parsed.at("args").size(), 1);
  EXPECT_EQ(parsed.at("args").at(0).at("tensor"), "arg_0");
  EXPECT_EQ(parsed.at("args").at(0).at("access"), "read_write");
}

TEST(ShaderProfilingConfig, InlineBindingsPreserveDataAndSharedReferences) {
  Json spec = config();
  spec["bindings"] = Json::array(
      {{{"meta_ubo", "shared"}, {"field", "numel"}},
       {{"buffer",
         {{"name", "shared"},
          {"sizes", {2}},
          {"dtype", "int32"},
          {"data", {-7, 42}}}}},
       {{"buffer", {{"ref", "shared"}, {"access", "read_write"}}}},
       {{"buffer", {{"sizes", {2}}, {"fill", 0.5}}}}});
  const Json parsed = parse_configs(spec).at(0);
  EXPECT_EQ(parsed.at("tensors").size(), 2);
  EXPECT_EQ(
      parsed.at("tensors").at("shared").at("data"), Json::array({-7, 42}));
  EXPECT_EQ(parsed.at("tensors").at("shared").at("dtype"), "int32");
  EXPECT_EQ(parsed.at("bindings").at(1).at("tensor"), "shared");
  EXPECT_EQ(parsed.at("bindings").at(2).at("tensor"), "shared");
}

TEST(
    ShaderProfilingConfig,
    RejectsExternalTensorListsAndDuplicateBindingNames) {
  Json spec = config();
  spec["tensors"] = {{"old", {{"sizes", {8}}}}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec.erase("tensors");
  spec["bindings"].push_back({{"buffer", {{"name", "out"}, {"sizes", {8}}}}});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsInvalidInlineTensorAndLayoutReferences) {
  Json spec = config();
  spec["bindings"][0]["buffer"]["sizes"] = {0};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["specialization_constants"] = Json::array({{{"layout", "missing"}}});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["outputs"] = {"missing"};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, BindingTypeDeterminesTensorStorage) {
  for (const char* type : {"buffer", "texture2d", "texture3d"}) {
    Json spec = config();
    spec["bindings"] =
        Json::array({{{type, {{"name", "input"}, {"sizes", {8}}}}}});
    EXPECT_EQ(
        parse_configs(spec).at(0).at("tensors").at("input").at("storage"),
        type);
  }
}

TEST(ShaderProfilingConfig, RejectsUntypedAndConflictingBindings) {
  Json spec = config();
  spec["bindings"] = Json::array({{{"name", "input"}, {"sizes", {8}}}});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["bindings"][0]["texture3d"] = {{"sizes", {8}}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["bindings"][0]["buffer"]["storage"] = "texture3d";
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["bindings"].push_back({{"texture3d", {{"ref", "out"}}}});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, AcceptsNamedWorkgroupPickers) {
  Json spec = config();
  spec["gwg_size"] = "default_pick_gwg";
  spec["lwg_size"] = "default_pick_lwg";
  EXPECT_EQ(parse_configs(spec).at(0).at("gwg_size"), "default_pick_gwg");
  spec["lwg_size"] = {8, 1, 1};
  EXPECT_EQ(parse_configs(spec).size(), 1);
  spec["gwg_size"] = {8, 1, 1};
  spec["lwg_size"] = "pick_xy_square_lwg";
  EXPECT_EQ(parse_configs(spec).size(), 1);
}

TEST(ShaderProfilingConfig, ExpandsLocalWorkgroupListInSpecifiedOrder) {
  Json spec = config();
  spec["lwg_size"] = {{8, 8, 1}, {10, 10, 1}};
  const auto cases = parse_configs(spec);
  ASSERT_EQ(cases.size(), 2);
  EXPECT_EQ(cases[0].at("lwg_size"), (Json{8, 8, 1}));
  EXPECT_EQ(cases[1].at("lwg_size"), (Json{10, 10, 1}));
  EXPECT_EQ(cases[0].at("sweep_index"), 0);
  EXPECT_EQ(cases[1].at("sweep_index"), 1);
  EXPECT_EQ(cases[0].at("tensors"), cases[1].at("tensors"));
  spec["lwg_size"] = {{8, 1, 1}, {0, 1, 1}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec["lwg_size"] = Json::array({Json{8, 1, 1}, "default_pick_lwg"});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(
    ShaderProfilingConfig,
    ExpandsInclusiveLocalWorkgroupRangeCartesianProduct) {
  Json spec = config();
  spec.erase("lwg_size");
  spec["lwg_size_sweep"] = {{"x", {8, 10, 2}}, {"y", {1, 3, 2}}};
  const auto cases = parse_configs(spec);
  ASSERT_EQ(cases.size(), 4);
  const Json expected = {{8, 1, 1}, {8, 3, 1}, {10, 1, 1}, {10, 3, 1}};
  for (size_t i = 0; i < cases.size(); ++i) {
    EXPECT_EQ(cases[i].at("lwg_size"), expected[i]);
    EXPECT_EQ(cases[i].at("sweep_index"), i);
    EXPECT_FALSE(cases[i].contains("lwg_size_sweep"));
  }
  spec["lwg_size_sweep"] = {{"x", {8, 11, 2}}};
  EXPECT_EQ(parse_configs(spec).size(), 2);
  spec["lwg_size_sweep"] = {{"x", {4294967294ULL, 4294967295ULL, 2}}};
  EXPECT_EQ(parse_configs(spec).size(), 1);
}

TEST(ShaderProfilingConfig, RejectsInvalidLocalWorkgroupSweeps) {
  Json spec = config();
  spec["lwg_size_sweep"] = {{"x", {1, 2, 1}}};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec.erase("lwg_size");
  for (const Json& invalid :
       {Json::array(),
        Json{{"w", {1, 2, 1}}},
        Json{{"x", {2, 1, 1}}},
        Json{{"x", {1, 2, 0}}},
        Json{{"x", {1, 2, -1}}},
        Json{{"x", {1, 2}}},
        Json{{"x", {1, 2, 0.5}}},
        Json{{"x", {0, 2, 1}}},
        Json{{"x", {1, 4294967296ULL, 1}}},
        Json{{"x", {1, 65537, 1}}},
        Json{{"x", {1, 257, 1}}, {"y", {1, 256, 1}}}}) {
    spec["lwg_size_sweep"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
  Json op = operator_config();
  op["lwg_size_sweep"] = {{"x", {1, 2, 1}}};
  EXPECT_THROW(parse_configs(op), std::invalid_argument);
}

TEST(ShaderProfilingConfig, ValidatesReferenceComparisonOptions) {
  Json spec = config();
  spec["reference"] = "copy";
  spec["validation_method"] = {
      {"function", "is_close"}, {"atol", 0.01}, {"rtol", 0.02}};
  EXPECT_EQ(parse_configs(spec)[0].at("reference"), "copy");
  spec["reference"] = "custom_reference";
  EXPECT_EQ(parse_configs(spec).size(), 1);
  for (const Json& invalid : {Json(0), Json("")}) {
    spec["reference"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
  spec["reference"] = "copy";
  for (const Json& invalid :
       {Json("is_close"),
        Json{{"function", "unknown"}},
        Json{{"atol", 0.01}},
        Json{{"function", "is_close"}, {"atol", -1}},
        Json{{"function", "is_close"}, {"rtol", "0.1"}},
        Json{
            {"function", "is_close"},
            {"atol", std::numeric_limits<double>::infinity()}},
        Json{{"function", "is_close"}, {"extra", 1}}}) {
    spec["validation_method"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
}

TEST(ShaderProfilingConfig, SelectsNamedValidationOutputs) {
  Json spec = config();
  spec["validation_method"] = {{"function", "is_close"}, {"outputs", {"out"}}};
  EXPECT_EQ(
      parse_configs(spec)[0].at("validation_method").at("outputs"),
      Json::array({"out"}));
  for (const Json& invalid :
       {Json("out"),
        Json::array(),
        Json::array({""}),
        Json::array({0}),
        Json::array({"missing"}),
        Json::array({"out", "out"})}) {
    spec["validation_method"]["outputs"] = invalid;
    EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  }
  spec = operator_config();
  spec["validation_method"] = {{"function", "is_close"}, {"outputs", {"y"}}};
  EXPECT_EQ(parse_configs(spec).size(), 1);
  spec["args_sweep"][0][1]["tensor"]["storage"] = "constant";
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

TEST(ShaderProfilingConfig, RejectsInvalidWorkgroupPickerValues) {
  for (const char* key : {"gwg_size", "lwg_size"}) {
    for (const Json& invalid :
         {Json(""), Json(1), Json(false), Json::object()}) {
      Json spec = config();
      spec[key] = invalid;
      EXPECT_THROW(parse_configs(spec), std::invalid_argument);
    }
  }
}

TEST(ShaderProfilingConfig, RejectsOldWorkgroupAndMetadataKeys) {
  Json spec = config();
  spec["global_size"] = {8, 1, 1};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec.erase("global_size");
  spec["local_size"] = {8, 1, 1};
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
  spec = config();
  spec["bindings"].push_back({{"metadata", "out"}});
  EXPECT_THROW(parse_configs(spec), std::invalid_argument);
}

} // namespace
TEST(ShaderProfilingConfig, RejectsRemovedNarrowInt4Setting) {
  for (Json spec :
       {config(),
        operator_config(),
        Json{{"check", "unsigned_dot_selection"}}}) {
    for (bool value : {false, true}) {
      spec["force_narrow_int4_tile"] = value;
      EXPECT_THROW(parse_configs(spec), std::invalid_argument);
    }
  }
}

TEST(ShaderProfilingConfig, RejectsRemovedBenchmarkControls) {
  const Json removed = {
      {"force_resize", true},
      {"invocations_per_batch", 17},
      {"gpu_timing", "batch"},
      {"shader_filter", Json::array()},
      {"requirements", {{"features", {"int8_dot_product"}}}},
      {"device_variables", {{"v_SIZE", {{"property", "subgroup_size"}}}}}};
  for (const Json& base :
       {config(),
        operator_config(),
        Json{{"check", "unsigned_dot_selection"}}}) {
    for (const auto& option : removed.items()) {
      Json spec = base;
      spec[option.key()] = option.value();
      EXPECT_THROW(parse_configs(spec), std::invalid_argument);
    }
  }
}

} // namespace vkcompute::shader_profiling
