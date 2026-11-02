// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace vkcompute::shader_profiling {

namespace reference {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void parallel_for(
    size_t count,
    size_t threads,
    const std::function<void(size_t)>& function) {
  if (threads == 0) {
    threads = std::max(1u, std::thread::hardware_concurrency());
  }
  threads = std::min({threads, size_t(8), count});
  std::vector<std::thread> workers;
  for (size_t worker = 1; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      for (size_t i = count * worker / threads;
           i < count * (worker + 1) / threads;
           ++i) {
        function(i);
      }
    });
  }
  for (size_t i = 0; threads > 0 && i < count / threads; ++i) {
    function(i);
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

const Tensor&
tensor_arg(const Json& spec, size_t index, const TensorMap& tensors) {
  return tensors.at(spec.at("args").at(index).at("tensor").get<std::string>());
}

} // namespace reference

namespace {

using Half = executorch::runtime::etensor::Half;

std::string tensor_dtype(
    const Json& spec,
    const Json& actual_dtypes,
    const std::string& name) {
  return actual_dtypes.contains(name)
      ? actual_dtypes.at(name).get<std::string>()
      : spec.at("tensors").at(name).value("dtype", "float32");
}

template <typename T>
double read_element(const std::vector<uint8_t>& bytes, size_t index) {
  T value;
  std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
  return double(value);
}

reference::Tensor decode_tensor(
    const Json& description,
    const std::string& actual_dtype,
    const std::vector<uint8_t>& bytes) {
  const std::string source_dtype = description.value("dtype", "float32");
  const bool constant_half = source_dtype == "float16" &&
      description.value("storage", "buffer") == "constant";
  const size_t element_size = constant_half                    ? sizeof(Half)
      : source_dtype == "float32" || source_dtype == "float16" ? sizeof(float)
      : source_dtype == "int32"                                ? sizeof(int32_t)
                                : sizeof(uint8_t);
  const size_t count = tensor_numel(description.at("sizes"));
  reference::require(
      bytes.size() == count * element_size,
      "Reference tensor byte count does not match its shape and dtype");
  reference::Tensor result{
      description.at("sizes").get<std::vector<int64_t>>(),
      std::vector<double>(count)};
  for (size_t i = 0; i < count; ++i) {
    double value;
    if (constant_half) {
      value = read_element<Half>(bytes, i);
    } else if (source_dtype == "float32" || source_dtype == "float16") {
      value = read_element<float>(bytes, i);
    } else if (source_dtype == "int32") {
      value = read_element<int32_t>(bytes, i);
    } else if (source_dtype == "uint8") {
      value = read_element<uint8_t>(bytes, i);
    } else {
      value = read_element<int8_t>(bytes, i);
    }
    result.values[i] =
        actual_dtype == "float16" ? float(Half(float(value))) : value;
  }
  return result;
}

std::string output_name(const Json& spec) {
  if (spec.contains("operator")) {
    return spec.at("args").back().at("tensor");
  }
  for (const Json& binding : spec.at("bindings")) {
    if (binding.contains("tensor")) {
      return binding.at("tensor");
    }
  }
  throw std::invalid_argument("Reference requires an output tensor binding");
}

} // namespace

std::string reference_name(const Json& spec) {
  const auto& registry = reference_registry();
  if (spec.contains("reference")) {
    const std::string name = spec.at("reference");
    reference::require(
        registry.has_reference(name), "Unknown reference function: " + name);
    return name;
  }
  return spec.contains("operator")
      ? registry.operator_reference(spec.at("operator").get<std::string>())
      : registry.shader_reference(spec.at("shader").get<std::string>());
}

ReferenceOutputs reference_outputs(
    const Json& spec,
    const Json& actual_dtypes,
    const std::map<std::string, std::vector<uint8_t>>& host_data,
    size_t threads) {
  const std::string function = reference_name(spec);
  reference::TensorMap tensors;
  for (const auto& item : host_data) {
    const Json& description = spec.at("tensors").at(item.first);
    tensors.emplace(
        item.first,
        decode_tensor(
            description,
            tensor_dtype(spec, actual_dtypes, item.first),
            item.second));
  }
  ReferenceOutputs outputs;
  if (const auto callback =
          reference_registry().get_reference_outputs(function)) {
    outputs = callback(spec, tensors, threads);
  } else {
    reference::Tensor output =
        reference_registry().get_reference(function)(spec, tensors, threads);
    const std::string name = output_name(spec);
    reference::require(
        output.sizes ==
            spec.at("tensors").at(name).at("sizes").get<std::vector<int64_t>>(),
        "Reference output shape does not match declared output tensor");
    outputs.emplace(name, std::move(output.values));
  }
  for (auto& output : outputs) {
    reference::require(
        spec.at("tensors").contains(output.first),
        "Reference produced unknown output tensor: " + output.first);
    reference::require(
        output.second.size() ==
            tensor_numel(spec.at("tensors").at(output.first).at("sizes")),
        "Reference output element count does not match declared shape: " +
            output.first);
    if (tensor_dtype(spec, actual_dtypes, output.first) == "float16") {
      for (double& value : output.second)
        value = float(Half(float(value)));
    }
  }
  return outputs;
}

Report compare_outputs(
    const Json& spec,
    const ReferenceOutputs& expected,
    const ReferenceOutputs& actual,
    const Json& actual_dtypes) {
  const Json method = spec.value("validation_method", Json::object());
  reference::require(
      method.value("function", "is_close") == "is_close",
      "Unknown validation method");
  Report result = {
      {"status", "PASS"},
      {"reference", reference_name(spec)},
      {"method", "is_close"},
      {"outputs", Report::object()}};
  const auto selected = validation_output_names(spec);
  if (!selected.empty()) {
    result["output_selection"] = selected;
    for (const auto& name : selected) {
      if (!expected.count(name)) {
        result["status"] = "FAIL";
        result["outputs"][name] = {
            {"status", "FAIL"}, {"error", "Missing reference output"}};
      }
    }
  }
  size_t total_mismatches = 0;
  if (expected.empty()) {
    result["status"] = "FAIL";
    result["error"] = "Reference produced no outputs";
  }
  for (const auto& item : expected) {
    if (!selected.empty() && !selected.count(item.first))
      continue;
    const std::string dtype = tensor_dtype(spec, actual_dtypes, item.first);
    const double default_tolerance = dtype == "float16" ? 1e-2
        : dtype == "float32"                            ? 1e-4
                                                        : 0;
    const double atol = method.value("atol", default_tolerance);
    const double rtol = method.value("rtol", default_tolerance);
    reference::require(
        std::isfinite(atol) && atol >= 0 && std::isfinite(rtol) && rtol >= 0,
        "Validation tolerances must be finite and nonnegative");
    Report output = {
        {"status", "PASS"},
        {"element_count", item.second.size()},
        {"atol", atol},
        {"rtol", rtol}};
    const auto computed = actual.find(item.first);
    if (computed == actual.end() ||
        computed->second.size() != item.second.size() ||
        item.second.size() !=
            tensor_numel(spec.at("tensors").at(item.first).at("sizes"))) {
      output["status"] = "FAIL";
      output["error"] = computed == actual.end()
          ? "Missing output"
          : "Output element count does not match declared shape";
      result["status"] = "FAIL";
      result["outputs"][item.first] = std::move(output);
      continue;
    }
    size_t mismatches = 0;
    size_t nonfinite = 0;
    double max_abs_error = 0;
    double max_rel_error = 0;
    Report mismatch_indices = Report::array();
    for (size_t i = 0; i < item.second.size(); ++i) {
      const double reference_value = item.second[i];
      const double actual_value = computed->second[i];
      const bool finite =
          std::isfinite(reference_value) && std::isfinite(actual_value);
      const double error = std::abs(actual_value - reference_value);
      if (finite) {
        max_abs_error = std::max(max_abs_error, error);
        const double relative = reference_value != 0
            ? error / std::abs(reference_value)
            : error == 0 ? 0
                         : std::numeric_limits<double>::infinity();
        max_rel_error = std::max(max_rel_error, relative);
      } else {
        ++nonfinite;
      }
      if (!finite || error > atol + rtol * std::abs(reference_value)) {
        ++mismatches;
        if (mismatch_indices.size() < 8) {
          mismatch_indices.push_back(i);
        }
      }
    }
    output["status"] = mismatches == 0 ? "PASS" : "FAIL";
    output["mismatch_count"] = mismatches;
    output["nonfinite_count"] = nonfinite;
    output["max_abs_error"] = max_abs_error;
    output["max_rel_error"] =
        std::isfinite(max_rel_error) ? Report(max_rel_error) : Report();
    output["first_mismatch_indices"] = std::move(mismatch_indices);
    total_mismatches += mismatches;
    if (mismatches > 0) {
      result["status"] = "FAIL";
    }
    result["outputs"][item.first] = std::move(output);
  }
  for (const auto& item : actual) {
    if (!selected.empty() && !selected.count(item.first))
      continue;
    if (!expected.count(item.first)) {
      result["status"] = "FAIL";
      result["outputs"][item.first] = {
          {"status", "FAIL"}, {"error", "Unexpected output"}};
    }
  }
  result["mismatch_count"] = total_mismatches;
  return result;
}

} // namespace vkcompute::shader_profiling
