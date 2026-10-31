// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

template <typename T>
std::vector<T> unpack(const std::vector<uint8_t>& data) {
  std::vector<T> result(data.size() / sizeof(T));
  std::memcpy(result.data(), data.data(), data.size());
  return result;
}
Json tensor(const char* dtype, const Json& sizes, const Json& data) {
  return {{"dtype", dtype}, {"sizes", sizes}, {"data", data}};
}

TEST(DataInitializer, GeneratesReproducibleIntegerValuedFloatAndHalfInputs) {
  const Json fp =
      tensor("float32", {512}, {{"rand_uniform_int", {-10, 10}}, {"seed", 23}});
  Json half = fp;
  half["dtype"] = "float16";
  const auto data = make_data(fp, 512, 1);
  EXPECT_EQ(data, make_data(fp, 512, 99));
  EXPECT_EQ(data, make_data(half, 512, 2));
  bool minimum = false, maximum = false;
  for (float value : unpack<float>(data)) {
    EXPECT_EQ(std::trunc(value), value);
    EXPECT_GE(value, -10);
    EXPECT_LE(value, 10);
    minimum |= value == -10;
    maximum |= value == 10;
  }
  EXPECT_TRUE(minimum);
  EXPECT_TRUE(maximum);
}
TEST(DataInitializer, RejectsFractionalOrOversizedIntegerRandomBounds) {
  for (const Json& bounds :
       {Json{0, 1.5},
        Json{2, 1},
        Json{0, 4294967296ULL},
        Json{-2147483648LL, 4294967295ULL}}) {
    EXPECT_THROW(
        validate_data_initializer(
            tensor("float32", {8}, {{"rand_uniform_int", bounds}}), 8),
        std::invalid_argument);
  }
}
TEST(DataInitializer, RepeatsPatternsAndValidatesEveryValue) {
  EXPECT_EQ(
      unpack<int8_t>(
          make_data(tensor("int8", {8}, {{"repeat", {-128, -1, 127}}}), 8)),
      (std::vector<int8_t>{-128, -1, 127, -128, -1, 127, -128, -1}));
  EXPECT_EQ(
      unpack<float>(
          make_data(tensor("float16", {5}, {{"repeat", {.5, -1}}}), 5)),
      (std::vector<float>{.5f, -1, .5f, -1, .5f}));
  for (const Json& pattern :
       {Json::array(),
        Json("bad"),
        Json{1, 128},
        Json{1, 1.5},
        Json{1, nullptr}}) {
    EXPECT_THROW(
        validate_data_initializer(
            tensor("int8", {1}, {{"repeat", pattern}}), 1),
        std::invalid_argument);
  }
}
TEST(DataInitializer, AbsoluteModuloMatchesRawSignedDraws) {
  const Json source =
      tensor("int32", {512}, {{"rand_uniform_int", {-100, 100}}, {"seed", 3}});
  const Json derived = tensor(
      "int32",
      {512},
      {{"abs_mod", {{"bounds", {-100, 100}}, {"mod", 7}, {"seed", 3}}}});
  const auto raw = unpack<int32_t>(make_data(source, 512));
  const auto values = unpack<int32_t>(make_data(derived, 512));
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(values[i], std::abs(raw[i]) % 7);
  EXPECT_EQ(
      unpack<int32_t>(make_data(
          tensor(
              "int32",
              {2},
              {{"abs_mod",
                {{"bounds", {-2147483648LL, -2147483648LL}}, {"mod", 3}}}}),
          2)),
      (std::vector<int32_t>{2, 2}));
  EXPECT_EQ(
      unpack<uint8_t>(make_data(
          tensor(
              "uint8",
              {2},
              {{"abs_mod", {{"bounds", {-1000, -1000}}, {"mod", 7}}}}),
          2)),
      (std::vector<uint8_t>{6, 6}));
}
TEST(DataInitializer, RejectsInvalidAbsoluteModuloDefinitions) {
  for (const Json& definition :
       {Json{{"bounds", {0, 1}}, {"mod", 0}},
        Json{{"bounds", {0, 1}}, {"mod", 1.5}},
        Json{{"bounds", {0, 1.5}}, {"mod", 2}},
        Json{{"bounds", {0, 1}}, {"mod", 2}, {"extra", 1}},
        Json{{"bounds", {0, 1}}, {"mod", 2}, {"seed", -1}}}) {
    EXPECT_THROW(
        validate_data_initializer(
            tensor("int32", {4}, {{"abs_mod", definition}}), 4),
        std::invalid_argument);
  }
}
TEST(DataInitializer, GeneratesEntirePaddedRowsBeforeZeroingColumns) {
  const Json initializer = {{"rand_uniform_int", {-128, 127}}, {"seed", 11}};
  const auto raw =
      unpack<int8_t>(make_data(tensor("int8", {3, 8}, initializer), 24));
  const Json padded = tensor(
      "int8",
      {3, 8},
      {{"pad_columns", {{"source", initializer}, {"valid_columns", 5}}}});
  const auto values = unpack<int8_t>(make_data(padded, 24));
  for (size_t i = 0; i < 24; ++i)
    EXPECT_EQ(values[i], i % 8 < 5 ? raw[i] : 0);
  const auto repeat = unpack<float>(make_data(
      tensor(
          "float32",
          {2, 4},
          {{"pad_columns",
            {{"source", {{"repeat", {1, 2, 3}}}}, {"valid_columns", 2}}}}),
      8));
  EXPECT_EQ(repeat, (std::vector<float>{1, 2, 0, 0, 2, 3, 0, 0}));
}
TEST(DataInitializer, RejectsInvalidPaddingShapesAndSources) {
  for (const Json& definition :
       {Json{{"source", {{"repeat", {1}}}}, {"valid_columns", -1}},
        Json{{"source", {{"repeat", {1}}}}, {"valid_columns", 5}},
        Json{{"source", {{"repeat", {1}}}}, {"valid_columns", 1.5}},
        Json{{"source", {{"row_sum", "other"}}}, {"valid_columns", 1}},
        Json{
            {"source", {{"repeat", {1}}}},
            {"valid_columns", 1},
            {"extra", 1}}}) {
    EXPECT_THROW(
        validate_data_initializer(
            tensor("float32", {2, 4}, {{"pad_columns", definition}}), 8),
        std::invalid_argument);
  }
  EXPECT_THROW(
      make_data(
          tensor(
              "float32",
              {2, 4},
              {{"pad_columns",
                {{"source", {{"repeat", {1}}}}, {"valid_columns", 1}}}}),
          7),
      std::invalid_argument);
}
TEST(DataInitializer, PadsRowSumsAndReducesAllLeadingColumnDimensions) {
  const Json specs = {
      {"weight",
       tensor("int8", {2, 2, 3}, {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6})},
      {"rows", tensor("int32", {4}, {{"row_sum", "weight"}})},
      {"columns", tensor("int32", {4}, {{"column_sum", "weight"}})}};
  const auto values = make_tensor_data(specs);
  EXPECT_EQ(
      unpack<int32_t>(values.at("rows")),
      (std::vector<int32_t>{21, -21, 0, 0}));
  EXPECT_EQ(
      unpack<int32_t>(values.at("columns")),
      (std::vector<int32_t>{0, 0, 0, 0}));
  Json nonzero = specs;
  nonzero["weight"]["data"] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_EQ(
      unpack<int32_t>(make_tensor_data(nonzero).at("columns")),
      (std::vector<int32_t>{22, 26, 30, 0}));
}
TEST(DataInitializer, RejectsShortSumsUnknownSourcesCyclesAndOverflow) {
  Json specs = {
      {"weight", tensor("int32", {2, 2}, {1, 2, 3, 4})},
      {"sum", tensor("int32", {1}, {{"row_sum", "weight"}})}};
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
  specs["sum"]["data"] = {{"column_sum", "weight"}};
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
  specs["sum"]["data"] = {{"row_sum", "missing"}};
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
  specs = {
      {"a", tensor("int32", {1}, {{"column_sum", "b"}})},
      {"b", tensor("int32", {1}, {{"row_sum", "a"}})}};
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
  specs = {
      {"weight", tensor("int32", {1, 2}, {2147483647, 1})},
      {"sum", tensor("int32", {1}, {{"row_sum", "weight"}})}};
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
  specs["weight"]["dtype"] = "float32";
  EXPECT_THROW(make_tensor_data(specs), std::invalid_argument);
}
TEST(DataInitializer, PackedQ4GroupSumsUseGroupMajorOrderAndPadOutputChannels) {
  const Json packed = {0x98, 0x76, 0xAF, 0x80};
  const Json base = {
      {"weight", tensor("uint8", {2, 2}, packed)},
      {"sum",
       tensor(
           "int32",
           {2, 4},
           {{"q4_group_sum", {{"tensor", "weight"}, {"group_size", 2}}}})}};
  const auto values = unpack<int32_t>(make_tensor_data(base).at("sum"));
  EXPECT_EQ(values, (std::vector<int32_t>{1, 9, 0, 0, -3, -8, 0, 0}));
  Json signed_source = base;
  signed_source["weight"]["dtype"] = "int8";
  signed_source["weight"]["data"] = {-104, 118, -81, -128};
  EXPECT_EQ(unpack<int32_t>(make_tensor_data(signed_source).at("sum")), values);
  for (const char* dtype : {"float32", "float16"}) {
    Json fp = base;
    fp["sum"]["dtype"] = dtype;
    EXPECT_EQ(
        unpack<float>(make_tensor_data(fp).at("sum")),
        (std::vector<float>{1, 9, 0, 0, -3, -8, 0, 0}));
  }
}
TEST(DataInitializer, RejectsInvalidQ4GroupShapesSourcesAndCycles) {
  Json specs = {
      {"weight", tensor("uint8", {2, 2}, {0, 0, 0, 0})},
      {"sum",
       tensor(
           "int32",
           {2, 2},
           {{"q4_group_sum", {{"tensor", "weight"}, {"group_size", 2}}}})}};
  for (int size : {0, 3, 8}) {
    Json invalid = specs;
    invalid["sum"]["data"]["q4_group_sum"]["group_size"] = size;
    EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  }
  Json invalid = specs;
  invalid["sum"]["sizes"] = {1, 2};
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  invalid = specs;
  invalid["sum"]["sizes"] = {2, 1};
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  invalid = specs;
  invalid["weight"]["dtype"] = "int32";
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
  invalid = specs;
  invalid["sum"]["data"]["q4_group_sum"]["tensor"] = "sum";
  EXPECT_THROW(make_tensor_data(invalid), std::invalid_argument);
}
} // namespace
} // namespace vkcompute::shader_profiling
