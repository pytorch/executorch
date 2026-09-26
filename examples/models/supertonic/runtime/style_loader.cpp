/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "style_loader.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace supertonic {
namespace {

using Json = nlohmann::json;

constexpr double kMaximumFp16 = 65504.0;

std::string shape_string(const std::vector<int>& dimensions) {
  std::string result = "[";
  for (size_t index = 0; index < dimensions.size(); ++index) {
    result += (index == 0 ? "" : ", ") + std::to_string(dimensions[index]);
  }
  return result + "]";
}

void read_exact_nested_data(
    const Json& value,
    const std::vector<int>& dimensions,
    size_t depth,
    const std::string& name,
    std::vector<float>& output) {
  if (depth == dimensions.size()) {
    if (!value.is_number()) {
      throw std::runtime_error(name + ".data must contain only numbers");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > kMaximumFp16) {
      throw std::runtime_error(
          name + ".data values must be within the finite FP16 range");
    }
    output.push_back(static_cast<float>(number));
    return;
  }
  if (!value.is_array() ||
      value.size() != static_cast<size_t>(dimensions[depth])) {
    throw std::runtime_error(
        name + ".data must have nested shape " + shape_string(dimensions));
  }
  for (const auto& child : value) {
    read_exact_nested_data(child, dimensions, depth + 1, name, output);
  }
}

std::vector<float> read_style_tensor(
    const Json& root,
    const char* name,
    const std::vector<int>& expected_dimensions,
    size_t expected_values) {
  if (!root.contains(name) || !root.at(name).is_object()) {
    throw std::runtime_error(std::string("missing ") + name);
  }
  const auto& tensor = root.at(name);
  if (!tensor.contains("dims") ||
      tensor.at("dims").get<std::vector<int>>() != expected_dimensions) {
    throw std::runtime_error(std::string(name) + ".dims has an invalid shape");
  }
  if (!tensor.contains("data") || !tensor.at("data").is_array()) {
    throw std::runtime_error(std::string(name) + ".data must be an array");
  }
  std::vector<float> values;
  values.reserve(expected_values);
  read_exact_nested_data(
      tensor.at("data"), expected_dimensions, 0, name, values);
  if (values.size() != expected_values) {
    throw std::runtime_error(
        std::string(name) + ".data must contain exactly " +
        std::to_string(expected_values) + " values");
  }
  return values;
}

} // namespace

std::string require_single_voice_style_path(
    const std::vector<std::string>& style_paths) {
  if (style_paths.size() != 1 || style_paths.front().empty()) {
    throw std::invalid_argument("expected exactly one voice style path");
  }
  return style_paths.front();
}

VoiceStyle load_voice_style(const std::string& style_path) {
  std::ifstream file(style_path);
  if (!file) {
    throw std::runtime_error("failed to open voice style: " + style_path);
  }
  Json style;
  try {
    style = Json::parse(file);
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "failed to parse voice style " + style_path + ": " + error.what());
  }
  return {
      read_style_tensor(style, "style_ttl", {1, 50, 256}, 50 * 256),
      read_style_tensor(style, "style_dp", {1, 8, 16}, 8 * 16)};
}

} // namespace supertonic
