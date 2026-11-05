// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ShaderBenchmark.h>

#include <gflags/gflags.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

DEFINE_string(
    config,
    "",
    "JSON file containing shader/operator configs, singly or as an array.");
DEFINE_bool(
    validate_specs,
    false,
    "Validate case schemas and registry bindings without executing operators.");
DEFINE_bool(
    validation_only,
    false,
    "Execute correctness checks without calibration or latency measurements.");
DEFINE_string(
    validation_mode,
    "all",
    "Cases to validate: all, first, or none.");
DEFINE_string(
    output_format,
    "json",
    "Output format: json (default), json_full, or table.");
DEFINE_string(
    error_mode,
    "report",
    "Error handling: report or throw; skip disables validation for compatibility.");
DEFINE_uint32(
    table_name_max_chars,
    32,
    "Table name width limit; 0 means unlimited.");
DEFINE_uint32(
    table_inputs_max_chars,
    64,
    "Table Inputs width limit; 0 means unlimited.");

namespace {

vkcompute::shader_profiling::Report skip_report(
    const vkcompute::shader_profiling::Json& spec,
    const std::string& reason,
    bool validation_enabled) {
  using namespace vkcompute::shader_profiling;
  Json accesses = Json::object();
  for (const auto& tensor : spec.at("tensors").items()) {
    accesses[tensor.key()] = "unknown";
  }
  Report result = spec.contains("operator")
      ? operator_report(spec, Json::object(), accesses)
      : shader_report(spec, Json(), Json(), Json::object(), Report::array());
  result["status"] = "SKIP";
  result["skip_reason"] = reason;
  if (validation_enabled) {
    result["validation"] = {{"status", "SKIP"}};
  }
  Report throughput = error_report(spec, reason);
  if (throughput.contains("throughput")) {
    result["throughput"] = std::move(throughput["throughput"]);
  }
  return result;
}

} // namespace

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Replay registered ET-VK shaders or operators from --config=<file.json>.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_config.empty() || argc != 1) {
    std::cerr << "Usage: etvk_ops_bench --config=<file.json> "
                 "[--output_format=json|json_full|table] "
                 "[--validation_mode=all|first|none]\n";
    return 2;
  }
  using namespace vkcompute::shader_profiling;
  try {
    if (FLAGS_output_format != "json" && FLAGS_output_format != "table" &&
        FLAGS_output_format != "json_full") {
      throw std::invalid_argument(
          "output_format must be json, json_full, or table");
    }
    if (FLAGS_validation_mode != "all" && FLAGS_validation_mode != "first" &&
        FLAGS_validation_mode != "none") {
      throw std::invalid_argument(
          "validation_mode must be all, first, or none");
    }
    if (FLAGS_error_mode != "report" && FLAGS_error_mode != "throw" &&
        FLAGS_error_mode != "skip") {
      throw std::invalid_argument("error_mode must be report, throw, or skip");
    }
    if (FLAGS_validation_only &&
        (FLAGS_validation_mode == "none" || FLAGS_error_mode == "skip")) {
      throw std::invalid_argument(
          "validation_only requires validation_mode=all or first and "
          "error_mode=report or throw");
    }
    std::ifstream file(FLAGS_config);
    if (!file) {
      throw std::invalid_argument("Cannot open config: " + FLAGS_config);
    }
    Json document;
    file >> document;
    const auto configs = parse_configs(document);
    if (FLAGS_validate_specs) {
      Report errors = Report::array();
      for (const auto& config : configs) {
        try {
          if (config.contains("check")) {
            case_check_registry().get_check(config.at("check"));
          } else if (config.contains("operator")) {
            const std::string op = config.at("operator");
            VK_GET_OP_FN(op);
            if (config.contains("route") &&
                config.at("route").at("name") != "auto") {
              route_registry().operator_route(
                  op, config.at("route").at("name"));
            }
            const auto reference = reference_name(config);
            if (!reference_registry().has_reference(reference))
              throw std::invalid_argument(
                  "Unknown reference function: " + reference);
          }
        } catch (const std::exception& error) {
          errors.push_back(
              {{"case_name", config.value("name", "")},
               {"error", error.what()}});
        }
      }
      const Report summary{
          {"status", errors.empty() ? "PASS" : "FAIL"},
          {"case_count", configs.size()},
          {"errors", errors}};
      std::cout << summary.dump(2) << std::endl;
      return errors.empty() ? 0 : 1;
    }
    Report results = Report::array();
    bool failed = false;
    for (size_t case_index = 0; case_index < configs.size(); ++case_index) {
      const Json& config = configs.at(case_index);
      const bool validation_enabled = FLAGS_error_mode != "skip" &&
          (FLAGS_validation_mode == "all" ||
           (FLAGS_validation_mode == "first" && case_index == 0));
      if (config.contains("check")) {
        Report result;
        if (!validation_enabled) {
          result = {
              {"case_name",
               config.value("name", config.at("check").get<std::string>())},
              {"check_name", config.at("check")},
              {"status", "SKIP"},
              {"skip_reason",
               FLAGS_error_mode == "skip"
                   ? "Validation-only case; error_mode=skip"
                   : "Validation-only case; validation_mode=" +
                       FLAGS_validation_mode}};
        } else {
          result = run_case_check(config);
          if (result.at("validation").at("status") == "FAIL") {
            failed = true;
            if (FLAGS_error_mode == "throw") {
              std::cerr << result.at("validation").dump() << std::endl;
              return 1;
            }
          }
        }
        results.push_back(std::move(result));
        continue;
      }
      if (FLAGS_validation_only && !validation_enabled) {
        results.push_back(skip_report(
            config,
            "Validation not selected by validation_mode=" +
                FLAGS_validation_mode,
            true));
        continue;
      }
      Report validation;
      Report validation_result;
      TensorData host_data;
      try {
        host_data = prepare_tensor_data(config);
      } catch (const std::exception& e) {
        failed = true;
        if (FLAGS_error_mode == "throw") {
          std::cerr << e.what() << std::endl;
          return 1;
        }
        Report result = error_report(config, e.what());
        if (validation_enabled) {
          result["validation"] = {{"status", "FAIL"}, {"error", e.what()}};
        }
        results.push_back(std::move(result));
        continue;
      }
      if (validation_enabled) {
        try {
          validation_result = config.contains("operator")
              ? benchmark_operator(config, host_data, true)
              : benchmark_shader(config, host_data, true);
          validation = validation_result.at("validation");
        } catch (const UnsupportedRoute& e) {
          results.push_back(skip_report(config, e.what(), validation_enabled));
          continue;
        } catch (const vkcompute::vkapi::ShaderNotSupportedError& e) {
          results.push_back(skip_report(config, e.what(), validation_enabled));
          continue;
        } catch (const std::exception& e) {
          validation = {{"status", "FAIL"}, {"error", e.what()}};
        }
        if (validation.at("status") == "FAIL") {
          failed = true;
          if (FLAGS_error_mode == "throw") {
            std::cerr << "Validation failed for "
                      << config.value(
                             "name",
                             config.value(
                                 "operator", config.value("shader", "")))
                      << ": " << validation.dump() << std::endl;
            return 1;
          }
        }
      }
      if (FLAGS_validation_only) {
        if (validation_result.is_null()) {
          validation_result = error_report(config, validation.dump());
        }
        validation_result["validation"] = std::move(validation);
        results.push_back(std::move(validation_result));
        continue;
      }
      Report result;
      try {
        result = config.contains("operator")
            ? benchmark_operator(config, host_data)
            : benchmark_shader(config, host_data);
      } catch (const UnsupportedRoute& e) {
        results.push_back(skip_report(config, e.what(), validation_enabled));
        continue;
      } catch (const vkcompute::vkapi::ShaderNotSupportedError& e) {
        results.push_back(skip_report(config, e.what(), validation_enabled));
        continue;
      } catch (const std::exception& e) {
        failed = true;
        if (FLAGS_error_mode == "throw") {
          std::cerr << e.what() << std::endl;
          return 1;
        }
        result = error_report(config, e.what());
      }
      if (!validation.is_null()) {
        result["validation"] = std::move(validation);
      }
      results.push_back(std::move(result));
    }
    const TableOptions table_options = {
        FLAGS_table_name_max_chars, FLAGS_table_inputs_max_chars};
    std::cout << (FLAGS_output_format == "table"
                      ? format_table(results, table_options)
                      : FLAGS_output_format == "json"
                      ? format_condensed_report(results)
                      : format_report(results))
              << std::endl;
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
}
