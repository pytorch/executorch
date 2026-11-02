// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Sweep.h>

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace vkcompute::shader_profiling {
namespace {

constexpr size_t kMaxSweepCases = 65536;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

Json substitute(const Json& value, const Json& variables) {
  if (value.is_string()) {
    const std::string name = value.get<std::string>();
    if (name.compare(0, 2, "v_") == 0) {
      require(variables.contains(name), "Undefined sweep variable: " + name);
      return variables.at(name);
    }
  }
  Json result = value;
  if (result.is_structured()) {
    for (auto& item : result) {
      item = substitute(item, variables);
    }
  }
  return result;
}

std::vector<Json> combinations(const Json& variables, size_t max_cases) {
  require(variables.is_object(), "sweep_variables must be an object");
  std::vector<Json> result{Json::object()};
  for (const auto& variable : variables.items()) {
    const std::string& name = variable.key();
    require(
        name.size() > 2 && name.compare(0, 2, "v_") == 0,
        "Sweep variable names must start with v_");
    require(
        variable.value().is_array() && !variable.value().empty(),
        "Sweep variable choices must be a nonempty array: " + name);
    require(
        variable.value().size() <= max_cases / result.size(),
        "Sweep must not exceed 65536 cases");
    std::vector<Json> next;
    for (const Json& combination : result) {
      for (const Json& value : variable.value()) {
        substitute(value, Json::object());
        Json entry = combination;
        entry[name] = value;
        next.push_back(std::move(entry));
      }
    }
    result = std::move(next);
  }
  return result;
}

Json normalize_route(Json route) {
  if (route.is_string()) {
    route = {{"name", route}};
  }
  require(route.is_object(), "route must be a name or an object");
  for (const auto& field : route.items()) {
    require(
        field.key() == "name" || field.key() == "options",
        "route accepts only name and options");
  }
  require(
      route.contains("name") && route.at("name").is_string() &&
          !route.at("name").get<std::string>().empty(),
      "route.name must be a nonempty string");
  route["options"] = route.value("options", Json::object());
  require(route.at("options").is_object(), "route.options must be an object");
  return route;
}

} // namespace

std::vector<Json> expand_operator_sweep(const Json& definition) {
  require(!definition.contains("args"), "Use args_sweep instead of args");
  require(
      definition.contains("args_sweep") &&
          definition.at("args_sweep").is_array() &&
          !definition.at("args_sweep").empty(),
      "args_sweep must be a nonempty list of argument sets");
  require(
      !(definition.contains("route") && definition.contains("route_sweep")),
      "Specify only one of route and route_sweep");
  const bool has_route =
      definition.contains("route") || definition.contains("route_sweep");
  Json routes = Json::array({nullptr});
  if (definition.contains("route_sweep")) {
    routes = definition.at("route_sweep");
    require(
        routes.is_array() && !routes.empty(),
        "route_sweep must be a nonempty list of routes");
  } else if (definition.contains("route")) {
    routes = Json::array({definition.at("route")});
  }
  const Json& sweep = definition.at("args_sweep");
  require(
      routes.size() <= kMaxSweepCases &&
          sweep.size() <= kMaxSweepCases / routes.size(),
      "Sweep must not exceed 65536 cases");
  const auto choices = combinations(
      definition.value("sweep_variables", Json::object()),
      kMaxSweepCases / (sweep.size() * routes.size()));
  Json common = definition;
  common.erase("args_sweep");
  common.erase("sweep_variables");
  common.erase("route");
  common.erase("route_sweep");
  std::vector<Json> result;
  result.reserve(sweep.size() * choices.size() * routes.size());
  for (size_t i = 0; i < sweep.size(); ++i) {
    require(
        sweep.at(i).is_array(),
        "Each args_sweep entry must be an argument array");
    common["args"] = sweep.at(i);
    for (const Json& choice : choices) {
      Json selected = substitute(common, choice);
      if (!choice.empty()) {
        selected["args_sweep_index"] = i;
        selected["sweep_variables"] = choice;
      }
      for (const Json& route : routes) {
        Json entry = selected;
        if (has_route) {
          entry["route"] = normalize_route(substitute(route, choice));
        }
        entry["sweep_index"] = result.size();
        result.push_back(std::move(entry));
      }
    }
  }
  return result;
}

namespace {

std::vector<Json> expand_lwg_sweep(const Json& definition, size_t max_cases) {
  require(
      !(definition.contains("lwg_size") &&
        definition.contains("lwg_size_sweep")),
      "Specify only one of lwg_size and lwg_size_sweep");
  Json sizes;
  if (definition.contains("lwg_size_sweep")) {
    const Json& sweep = definition.at("lwg_size_sweep");
    require(sweep.is_object(), "lwg_size_sweep must be an object");
    for (const auto& axis : sweep.items()) {
      require(
          axis.key() == "x" || axis.key() == "y" || axis.key() == "z",
          "lwg_size_sweep accepts only x, y, z");
    }
    std::array<uint32_t, 3> starts{1, 1, 1};
    std::array<uint32_t, 3> steps{1, 1, 1};
    std::array<size_t, 3> counts{1, 1, 1};
    size_t count = 1;
    for (size_t i = 0; i < 3; ++i) {
      const char* axis = i == 0 ? "x" : i == 1 ? "y" : "z";
      if (!sweep.contains(axis)) {
        continue;
      }
      const Json& range = sweep.at(axis);
      require(
          range.is_array() && range.size() == 3,
          std::string("lwg_size_sweep.") + axis +
              " must be [start, stop, step]");
      for (const Json& value : range) {
        require(
            value.is_number_integer() && value.get<double>() > 0 &&
                value.get<double>() <= UINT32_MAX,
            "Local workgroup range values must be positive uint32 integers");
      }
      starts[i] = range.at(0).get<uint32_t>();
      const uint32_t stop = range.at(1).get<uint32_t>();
      steps[i] = range.at(2).get<uint32_t>();
      require(starts[i] <= stop, "Local workgroup range start exceeds stop");
      const uint64_t length = (uint64_t(stop) - starts[i]) / steps[i] + 1;
      require(length <= max_cases / count, "Sweep must not exceed 65536 cases");
      counts[i] = static_cast<size_t>(length);
      count *= counts[i];
    }
    sizes = Json::array();
    for (size_t x = 0; x < counts[0]; ++x) {
      for (size_t y = 0; y < counts[1]; ++y) {
        for (size_t z = 0; z < counts[2]; ++z) {
          sizes.push_back(
              {starts[0] + uint64_t(x) * steps[0],
               starts[1] + uint64_t(y) * steps[1],
               starts[2] + uint64_t(z) * steps[2]});
        }
      }
    }
  } else if (
      definition.contains("lwg_size") && definition.at("lwg_size").is_array() &&
      !definition.at("lwg_size").empty() &&
      definition.at("lwg_size").front().is_array()) {
    sizes = definition.at("lwg_size");
  } else {
    return {definition};
  }
  require(sizes.size() <= max_cases, "Sweep must not exceed 65536 cases");
  std::vector<Json> result;
  result.reserve(sizes.size());
  for (const Json& size : sizes) {
    require(
        size.is_array() && size.size() == 3,
        "Each lwg_size sweep entry must contain three positive integers");
    Json entry = definition;
    entry.erase("lwg_size_sweep");
    entry["lwg_size"] = size;
    entry["sweep_index"] = result.size();
    result.push_back(std::move(entry));
  }
  return result;
}

} // namespace

std::vector<Json> expand_shader_sweep(const Json& definition) {
  if (!definition.contains("shader_sweep")) {
    return expand_lwg_sweep(definition, kMaxSweepCases);
  }
  require(
      !definition.contains("shader"),
      "Specify only one of shader and shader_sweep");
  const Json& shaders = definition.at("shader_sweep");
  require(
      shaders.is_array() && !shaders.empty(),
      "shader_sweep must be a nonempty list of shader names");
  require(
      shaders.size() <= kMaxSweepCases, "Sweep must not exceed 65536 cases");
  for (const Json& shader : shaders) {
    require(
        shader.is_string() && !shader.get<std::string>().empty(),
        "shader_sweep entries must be nonempty shader names");
  }
  Json common = definition;
  common.erase("shader_sweep");
  const auto workgroups =
      expand_lwg_sweep(common, kMaxSweepCases / shaders.size());
  std::vector<Json> result;
  result.reserve(shaders.size() * workgroups.size());
  for (const Json& shader : shaders) {
    for (const Json& workgroup : workgroups) {
      Json entry = workgroup;
      entry["shader"] = shader;
      entry["sweep_index"] = result.size();
      result.push_back(std::move(entry));
    }
  }
  return result;
}

} // namespace vkcompute::shader_profiling
