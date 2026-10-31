// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace vkcompute::shader_profiling {
namespace {

void write_report_value(
    std::ostream& output,
    const Report& value,
    size_t depth,
    std::string_view field = {},
    bool inline_array = false) {
  if ((!value.is_object() && !value.is_array()) || value.empty()) {
    output << value.dump();
    return;
  }
  if (value.is_array() && inline_array) {
    output << '[';
    for (size_t i = 0; i < value.size(); ++i) {
      if (i > 0) {
        output << ", ";
      }
      output << value.at(i).dump();
    }
    output << ']';
    return;
  }

  const bool object = value.is_object();
  output << (object ? '{' : '[') << '\n';
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it != value.begin()) {
      output << ",\n";
    }
    output << std::string((depth + 1) * 2, ' ');
    const std::string key = object ? it.key() : "";
    if (object) {
      output << Report(key).dump() << ": ";
    }
    constexpr std::string_view clock_suffix = "_observed_gpu_clock_freq_MHz";
    const bool clock_pairs = field.size() >= clock_suffix.size() &&
        field.substr(field.size() - clock_suffix.size()) == clock_suffix;
    const bool compact = key == "sizes" || clock_pairs ||
        (key == "value" && (field == "gwg_size" || field == "lwg_size"));
    write_report_value(output, it.value(), depth + 1, key, compact);
  }
  output << '\n' << std::string(depth * 2, ' ') << (object ? '}' : ']');
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const double position = fraction * (sorted.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

Report case_header(const Json& spec) {
  const bool op = spec.contains("operator");
  const std::string name = spec.at(op ? "operator" : "shader");
  Report report = {
      {"case_name", spec.value("name", name)},
      {op ? "operator_name" : "shader_name", name}};
  if (spec.contains("sweep_index")) {
    report["sweep_index"] = spec.at("sweep_index");
  }
  if (op && spec.contains("sweep_variables")) {
    report["args_sweep_index"] = spec.at("args_sweep_index");
    report["sweep_variables"] = spec.at("sweep_variables");
  }
  if (op && spec.contains("route")) {
    report["route"] = {{"requested", spec.at("route")}, {"resolved", nullptr}};
  }
  return report;
}

Report tensor_description(const Json& tensor, const std::string& actual_dtype) {
  const std::string requested = tensor.value("dtype", "float32");
  Report info = {
      {"type", tensor.value("storage", "buffer")},
      {"sizes", tensor.at("sizes")},
      {"dtype", actual_dtype},
      {"layout", tensor.value("layout", "width")}};
  if (actual_dtype != requested) {
    info["requested_dtype"] = requested;
  }
  return info;
}

Report describe_bindings(const Json& spec, const Json& actual_dtypes) {
  Report result = Report::object();
  const Json& tensors = spec.at("tensors");
  for (size_t i = 0; i < spec.at("bindings").size(); ++i) {
    const Json& binding = spec.at("bindings").at(i);
    Report info;
    std::string name;
    if (binding.contains("tensor")) {
      name = binding.at("tensor");
      const Json& tensor = tensors.at(name);
      info = tensor_description(
          tensor, actual_dtypes.value(name, tensor.value("dtype", "float32")));
      info["access"] = binding.value("access", "read");
      if (result.contains(name)) {
        info["ref"] = name;
      }
    } else if (binding.contains("meta_ubo")) {
      const std::string tensor = binding.at("meta_ubo");
      const std::string field = binding.value("field", "meta");
      name = tensor + "." + field;
      info = {{"type", "meta_ubo"}, {"tensor", tensor}, {"field", field}};
    } else {
      name = "binding_" + std::to_string(i);
      info = {{"type", "uniform"}, {"values", binding.at("uniform")}};
    }
    info["binding_index"] = i;
    std::string key = name;
    for (size_t suffix = i; result.contains(key); ++suffix) {
      key = name + "_" + std::to_string(suffix);
    }
    result[key] = std::move(info);
  }
  return result;
}

Report workgroup_report(const Json& setting, const Json& resolved) {
  return {
      {"method", setting.is_string() ? setting.get<std::string>() : "explicit"},
      {"value", resolved}};
}

std::string compact_tensor(const Report& tensor) {
  const std::string storage = tensor.value("type", "buffer");
  std::string prefix = storage;
  if (storage == "buffer") {
    prefix = "b";
  } else if (storage == "texture2d") {
    prefix = "t2";
  } else if (storage == "texture3d") {
    prefix = "t3";
  } else if (storage == "constant") {
    prefix = "c";
  }
  const std::string dtype = tensor.at("dtype");
  std::string suffix = dtype;
  if (dtype == "float32") {
    suffix = "f32";
  } else if (dtype == "float16") {
    suffix = "f16";
  } else if (dtype == "int32") {
    suffix = "i32";
  } else if (dtype == "int8") {
    suffix = "i8";
  } else if (dtype == "uint8") {
    suffix = "u8";
  } else if (dtype == "int8x4") {
    suffix = "i8x4";
  }
  if (storage != "constant") {
    suffix += "." + tensor.value("layout", "width");
  }
  return prefix + tensor.at("sizes").dump() + suffix;
}

std::string compact_input_value(const Report& value) {
  if (value.is_number_float()) {
    std::ostringstream output;
    output << std::setprecision(3) << value.get<double>();
    return output.str();
  }
  if (value.is_array()) {
    std::string output = "[";
    for (size_t i = 0; i < value.size(); ++i) {
      if (i > 0) {
        output += ',';
      }
      output += compact_input_value(value.at(i));
    }
    return output + ']';
  }
  return value.dump();
}

std::string table_inputs(const Report& report, bool op) {
  std::string inputs;
  auto append = [&](const std::string& value) {
    if (!inputs.empty()) {
      inputs += ", ";
    }
    inputs += value;
  };
  if (op) {
    for (const auto& arg : report.at("args")) {
      if (arg.contains("tensor")) {
        if (arg.at("tensor").value("access", "read") != "write") {
          append(compact_tensor(arg.at("tensor")));
        }
      } else {
        append(
            compact_input_value(arg.is_object() ? arg.begin().value() : arg));
      }
    }
  } else {
    auto append_blocks = [&](const Report& blocks) {
      for (const auto& block : blocks) {
        append(compact_input_value(block.at("data")));
      }
    };
    for (const auto& binding : report.at("bindings")) {
      if (binding.contains("sizes") &&
          binding.value("access", "read") != "write") {
        append(compact_tensor(binding));
      } else if (binding.value("type", "") == "uniform") {
        append_blocks(binding.at("values"));
      }
    }
    if (report.contains("push_constants")) {
      append_blocks(report.at("push_constants"));
    }
  }
  return inputs.empty() ? "-" : inputs;
}

std::string table_latency(const Report& stats, const char* key) {
  if (stats.is_null()) {
    return "N/A";
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << stats.at(key).get<double>();
  return output.str();
}

std::string table_gpu_throughput(const Report& report, const char* metric) {
  if (!report.contains("throughput") ||
      !report.at("throughput").contains("gpu_timestamp")) {
    return "N/A";
  }
  const Report& rates = report.at("throughput").at("gpu_timestamp");
  return rates.contains(metric) ? table_latency(rates.at(metric), "mean")
                                : "N/A";
}

constexpr std::array<const char*, 3> kObservedGpuClockFields = {
    "gpu_batch_observed_gpu_clock_freq_MHz",
    "wall_time_batch_observed_gpu_clock_freq_MHz",
    "shader_profile_batch_observed_gpu_clock_freq_MHz"};

void observe_gpu_clock_range(
    std::array<std::optional<double>, 2>& range,
    const Report& samples) {
  if (!samples.is_array())
    return;
  for (const auto& pair : samples) {
    if (!pair.is_array())
      continue;
    for (const auto& observation : pair) {
      if (!observation.is_number())
        continue;
      const double frequency = observation.get<double>();
      if (!std::isfinite(frequency) || frequency < 0)
        continue;
      range[0] = range[0] ? std::min(*range[0], frequency) : frequency;
      range[1] = range[1] ? std::max(*range[1], frequency) : frequency;
    }
  }
}

std::array<std::string, 2> table_gpu_clock_range(
    const Report& report,
    bool profile_only = false) {
  std::array<std::optional<double>, 2> range{};
  const size_t first = profile_only ? kObservedGpuClockFields.size() - 1 : 0;
  for (size_t i = first; i < kObservedGpuClockFields.size(); ++i) {
    const char* field = kObservedGpuClockFields[i];
    if (report.contains(field))
      observe_gpu_clock_range(range, report.at(field));
  }
  std::array<std::string, 2> result{"N/A", "N/A"};
  for (size_t i = 0; i < result.size(); ++i) {
    if (range[i]) {
      std::ostringstream output;
      output << std::setprecision(9) << *range[i];
      result[i] = output.str();
    }
  }
  return result;
}

std::string table_text(std::string text, size_t max_chars = 0) {
  for (char& ch : text) {
    if (ch == '\n' || ch == '\r' || ch == '\t') {
      ch = ' ';
    }
  }
  if (max_chars != 0 && text.size() > max_chars) {
    const size_t indentation = std::min(text.find_first_not_of(' '), max_chars);
    const size_t dots = std::min<size_t>(3, max_chars - indentation);
    text = text.substr(0, max_chars - dots) + std::string(dots, '.');
  }
  return text;
}

std::string table_route(const Report& report) {
  if (!report.contains("route")) {
    return "-";
  }
  const Report& route = report.at("route");
  const Report& requested = route.at("requested");
  const Report& resolved = route.at("resolved");
  const Report& selected = resolved.is_null() ? requested : resolved;
  const std::string requested_name = requested.at("name");
  const std::string selected_name = selected.at("name");
  std::string label = requested_name == "auto" && selected_name != "auto"
      ? "auto->" + selected_name
      : selected_name;
  const Report selected_options = selected.value("options", Report::object());
  if (selected_options.contains("intermediate_storage")) {
    label += "[" +
        selected_options.at("intermediate_storage").get<std::string>() + "]";
  }
  return table_text(label, 32);
}

} // namespace

std::string format_report(const Report& report) {
  std::ostringstream output;
  write_report_value(output, report, 0);
  return output.str();
}

std::string format_condensed_report(const Report& report) {
  auto condense_case = [](const Report& original) {
    if (!original.is_object())
      return original;
    Report condensed = Report::object();
    for (const auto& item : original.items()) {
      const std::string& key = item.key();
      const Report& value = item.value();
      const bool batch_timing = key == "gpu_timestamp_operator_us" ||
          key == "gpu_timestamp_batch_us" || key == "wall_time_operator_us" ||
          key == "wall_time_batch_us" || key == "wall_time_shader_us";
      if (key == "args") {
        condensed["args_str"] = table_text(table_inputs(original, true));
      } else if (batch_timing && value.is_array()) {
        const Report stats =
            latency_statistics(value.get<std::vector<double>>());
        condensed[key] = stats.is_null() ? Report()
                                         : Report{
                                               {"min", stats.at("min")},
                                               {"median", stats.at("p50")},
                                               {"max", stats.at("max")}};
      } else if (
          std::find(
              kObservedGpuClockFields.begin(),
              kObservedGpuClockFields.end(),
              key) != kObservedGpuClockFields.end()) {
        std::array<std::optional<double>, 2> range{};
        observe_gpu_clock_range(range, value);
        condensed[key] = range[0]
            ? Report{{"min", *range[0]}, {"max", *range[1]}}
            : Report();
      } else {
        condensed[key] = value;
      }
    }
    return condensed;
  };
  if (!report.is_array())
    return format_report(condense_case(report));
  Report condensed = Report::array();
  for (const auto& item : report)
    condensed.push_back(condense_case(item));
  return format_report(condensed);
}

std::string format_table(const Report& reports, const TableOptions& options) {
  const bool show_route =
      std::any_of(reports.begin(), reports.end(), [](const Report& report) {
        return report.contains("route");
      });
  const bool show_validation =
      std::any_of(reports.begin(), reports.end(), [](const Report& report) {
        return report.contains("validation");
      });
  const bool show_clocks =
      std::any_of(reports.begin(), reports.end(), [](const Report& report) {
        return std::any_of(
            kObservedGpuClockFields.begin(),
            kObservedGpuClockFields.end(),
            [&](const char* field) { return report.contains(field); });
      });
  auto has_throughput = [&](const char* metric) {
    return std::any_of(
        reports.begin(), reports.end(), [&](const Report& report) {
          return report.contains("throughput") &&
              report.at("throughput").contains(metric);
        });
  };
  std::vector<size_t> columns = {0};
  if (show_route) {
    columns.push_back(9);
  }
  columns.insert(columns.end(), {1, 2, 3, 4, 5});
  if (show_clocks) {
    columns.insert(columns.end(), {10, 11});
  }
  if (show_validation) {
    columns.push_back(6);
  }
  if (has_throughput("memory_bytes")) {
    columns.push_back(7);
  }
  if (has_throughput("compute_flops")) {
    columns.push_back(8);
  }
  using Row = std::array<std::string, 12>;
  std::vector<Row> rows = {
      {"Shader / operator",
       "GWG",
       "LWG",
       "Inputs",
       "Mean GPU us",
       "P90 GPU us",
       "Validation",
       "GPU GiB/s",
       "GPU GFLOPS",
       "Route",
       "Min GPU MHz",
       "Max GPU MHz"}};
  std::vector<size_t> case_rows;
  std::vector<std::string> errors;
  for (const auto& report : reports) {
    const bool op = report.contains("operator_name");
    const bool check = report.contains("check_name");
    const std::string name = report.at(
        check    ? "check_name"
            : op ? "operator_name"
                 : "shader_name");
    const std::string suffix = report.contains("sweep_index")
        ? " [" + std::to_string(report.at("sweep_index").get<size_t>()) + "]"
        : "";
    const bool skipped = report.value("status", "") == "SKIP";
    const std::string label = table_text(
        name + suffix + (skipped ? " (SKIP)" : ""), options.name_max_chars);
    const std::string route = table_route(report);
    const std::string validation = !skipped && report.contains("validation")
        ? table_text(report.at("validation").at("status").get<std::string>())
        : "SKIP";
    case_rows.push_back(rows.size());
    if (skipped) {
      const std::string inputs = report.contains(op ? "args" : "bindings")
          ? table_inputs(report, op)
          : "-";
      rows.push_back(
          {label,
           "-",
           "-",
           table_text(inputs, options.inputs_max_chars),
           "N/A",
           "N/A",
           "SKIP",
           "N/A",
           "N/A",
           route,
           "N/A",
           "N/A"});
      errors.push_back(table_text(
          "SKIP " + report.value("case_name", name) + suffix + ": " +
          report.value("skip_reason", "")));
      continue;
    }
    if (report.contains("error")) {
      rows.push_back(
          {label,
           "-",
           "-",
           "ERROR",
           "N/A",
           "N/A",
           validation,
           "N/A",
           "N/A",
           route,
           "N/A",
           "N/A"});
      errors.push_back(table_text(
          "ERROR " + report.value("case_name", name) + suffix + ": " +
          report.at("error").get<std::string>()));
      continue;
    }
    if (check) {
      rows.push_back(
          {label,
           "-",
           "-",
           "-",
           "N/A",
           "N/A",
           validation,
           "N/A",
           "N/A",
           "-",
           "N/A",
           "N/A"});
      if (report.at("validation").value("status", "") == "FAIL")
        errors.push_back(table_text(
            "FAIL " + report.value("case_name", name) + ": " +
            report.at("validation").value("error", "Case check failed")));
      continue;
    }
    const auto& timing =
        report.at(op ? "gpu_timestamp_operator_us" : "gpu_timestamp_shader_us");
    const Report stats = timing.is_array()
        ? latency_statistics(timing.get<std::vector<double>>())
        : timing;
    const auto clocks = table_gpu_clock_range(report);
    rows.push_back(
        {label,
         op ? "-" : report.at("gwg_size").at("value").dump(),
         op ? "-" : report.at("lwg_size").at("value").dump(),
         table_text(table_inputs(report, op), options.inputs_max_chars),
         table_latency(stats, "mean"),
         table_latency(stats, "p90"),
         validation,
         table_gpu_throughput(report, "memory_gib_per_s"),
         table_gpu_throughput(report, "compute_gflops"),
         route,
         clocks[0],
         clocks[1]});
    if (op && report.contains("shader_breakdown")) {
      const auto profile_clocks = table_gpu_clock_range(report, true);
      for (const auto& shader : report.at("shader_breakdown")) {
        const auto& shader_stats = shader.at("gpu_timestamp_shader_us");
        rows.push_back(
            {table_text(
                 "  " + shader.at("shader_name").get<std::string>(),
                 options.name_max_chars),
             shader.at("gwg_size").at("value").dump(),
             shader.at("lwg_size").at("value").dump(),
             "-",
             table_latency(shader_stats, "mean"),
             table_latency(shader_stats, "p90"),
             "-",
             table_gpu_throughput(shader, "memory_gib_per_s"),
             table_gpu_throughput(shader, "compute_gflops"),
             "-",
             profile_clocks[0],
             profile_clocks[1]});
      }
    }
  }
  std::array<size_t, 12> widths{};
  for (const auto& row : rows) {
    for (size_t i : columns) {
      widths[i] = std::max(widths[i], row[i].size());
    }
  }
  const size_t line_width =
      std::accumulate(widths.begin(), widths.end(), size_t{0}) +
      2 * (columns.size() - 1);
  const std::string separator(line_width, '-');
  std::ostringstream output;
  for (size_t r = 0; r < rows.size(); ++r) {
    if (r > 0) {
      output << '\n';
    }
    if (std::binary_search(case_rows.begin(), case_rows.end(), r)) {
      output << separator << '\n';
    }
    for (size_t i : columns) {
      if (i != columns.front()) {
        output << "  ";
      }
      output << (i < 4 || i == 6 || i == 9 ? std::left : std::right)
             << std::setw(static_cast<int>(widths[i])) << rows[r][i];
    }
  }
  for (const auto& error : errors) {
    output << '\n' << error;
  }
  return output.str();
}

Report latency_statistics(std::vector<double> samples) {
  if (samples.empty()) {
    return nullptr;
  }
  std::sort(samples.begin(), samples.end());
  return {
      {"min", samples.front()},
      {"max", samples.back()},
      {"mean",
       std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()},
      {"p50", percentile(samples, 0.5)},
      {"p90", percentile(samples, 0.9)},
      {"p95", percentile(samples, 0.95)}};
}

Report measurement_report(
    uint32_t repetitions,
    bool operator_case,
    const std::vector<double>& gpu_batches,
    const std::vector<double>& wall_batches) {
  if (repetitions == 0) {
    throw std::invalid_argument("Repetitions must be positive");
  }
  std::vector<double> gpu_average = gpu_batches;
  std::vector<double> wall_average = wall_batches;
  for (double& value : gpu_average) {
    value /= repetitions;
  }
  for (double& value : wall_average) {
    value /= repetitions;
  }
  const Report gpu_latency = operator_case
      ? Report(gpu_average)
      : latency_statistics(std::move(gpu_average));
  return {
      {operator_case ? "gpu_timestamp_operator_us" : "gpu_timestamp_shader_us",
       gpu_latency},
      {"gpu_timestamp_batch_us", gpu_batches},
      {operator_case ? "wall_time_operator_us" : "wall_time_shader_us",
       wall_average},
      {"wall_time_batch_us", wall_batches}};
}

Report throughput_report(
    const Json& spec,
    const WorkAmounts& work,
    uint32_t repetitions,
    const std::vector<double>& gpu_batches,
    const std::vector<double>& wall_batches) {
  const bool memory =
      spec.is_object() && spec.value("show_memory_throughput", false);
  const bool compute =
      spec.is_object() && spec.value("show_compute_throughput", false);
  Report result = Report::object();
  if (!memory && !compute) {
    return result;
  }
  if (repetitions == 0) {
    throw std::invalid_argument("Repetitions must be positive");
  }
  auto set_work = [&](const char* key, const std::optional<double>& amount) {
    if (amount && (!std::isfinite(*amount) || *amount < 0)) {
      throw std::invalid_argument("Work counts must be finite and nonnegative");
    }
    result[key] = amount ? Report(*amount) : Report();
  };
  if (memory) {
    set_work("memory_bytes", work.memory_bytes);
  }
  if (compute) {
    set_work("compute_flops", work.compute_flops);
  }
  auto metrics = [&](const std::vector<double>& batches, bool statistics) {
    Report output = Report::object();
    auto add_metric = [&](const char* key,
                          const std::optional<double>& amount,
                          double units) {
      if (!amount) {
        output[key] = nullptr;
        return;
      }
      Report rates = Report::array();
      std::vector<double> samples;
      bool valid = true;
      for (double batch_us : batches) {
        const double invocation_us = batch_us / repetitions;
        if (invocation_us <= 0 || !std::isfinite(invocation_us)) {
          rates.push_back(nullptr);
          valid = false;
          continue;
        }
        const double rate = *amount / units * (1e6 / invocation_us);
        if (std::isfinite(rate)) {
          rates.push_back(rate);
          samples.push_back(rate);
        } else {
          rates.push_back(nullptr);
          valid = false;
        }
      }
      output[key] = statistics
          ? (valid ? latency_statistics(std::move(samples)) : Report())
          : std::move(rates);
    };
    if (memory) {
      add_metric("memory_gib_per_s", work.memory_bytes, 1073741824.0);
      add_metric("memory_mib_per_s", work.memory_bytes, 1048576.0);
    }
    if (compute) {
      add_metric("compute_gflops", work.compute_flops, 1e9);
    }
    return output;
  };
  result["gpu_timestamp"] = metrics(gpu_batches, true);
  result["gpu_timestamp_batch"] = result.at("gpu_timestamp");
  result["wall_time"] = metrics(wall_batches, false);
  result["wall_time_batch"] = result.at("wall_time");
  if (!work.diagnostics.empty()) {
    result["diagnostics"] = work.diagnostics;
  }
  return result;
}

Report shader_report(
    const Json& spec,
    const Json& gwg,
    const Json& lwg,
    const Json& actual_dtypes,
    const Report& constants) {
  Report report = case_header(spec);
  report["gwg_size"] = workgroup_report(spec.at("gwg_size"), gwg);
  report["lwg_size"] = workgroup_report(spec.at("lwg_size"), lwg);
  report["bindings"] = describe_bindings(spec, actual_dtypes);
  report["specialization_constants"] = constants;
  if (spec.contains("push_constants")) {
    report["push_constants"] = spec.at("push_constants");
  }
  return report;
}

Report shader_breakdown_report(
    const Report& dispatches,
    const std::vector<std::vector<double>>& samples,
    uint32_t repetitions,
    const Json& throughput_flags,
    const std::vector<WorkAmounts>& work) {
  if (repetitions == 0) {
    throw std::invalid_argument("Repetitions must be positive");
  }
  for (const auto& sample : samples) {
    if (sample.size() != dispatches.size() * repetitions) {
      throw std::invalid_argument(
          "Shader timestamp count changed between runs");
    }
  }
  Report result = dispatches;
  if (!work.empty() && work.size() != dispatches.size()) {
    throw std::invalid_argument(
        "Shader work count does not match dispatch count");
  }
  for (size_t i = 0; i < dispatches.size(); ++i) {
    std::vector<double> durations;
    durations.reserve(samples.size());
    for (const auto& sample : samples) {
      double total = 0;
      for (uint32_t r = 0; r < repetitions; ++r) {
        total += sample[r * dispatches.size() + i];
      }
      durations.push_back(total / repetitions);
    }
    result[i]["gpu_timestamp_shader_us"] = latency_statistics(durations);
    Report throughput = throughput_report(
        throughput_flags,
        work.empty() ? WorkAmounts{} : work[i],
        1,
        durations,
        {});
    if (!throughput.empty()) {
      throughput.erase("gpu_timestamp_batch");
      throughput.erase("wall_time");
      throughput.erase("wall_time_batch");
      result[i]["throughput"] = std::move(throughput);
    }
  }
  return result;
}

Report operator_report(
    const Json& spec,
    const Json& actual_dtypes,
    const Json& actual_accesses) {
  Report report = case_header(spec);
  report["args"] = Report::array();
  for (const Json& arg : spec.at("args")) {
    if (arg.contains("tensor")) {
      const std::string name = arg.at("tensor");
      const Json& tensor = spec.at("tensors").at(name);
      Report description = {{"name", name}};
      description.update(tensor_description(
          tensor, actual_dtypes.value(name, tensor.value("dtype", "float32"))));
      description["access"] =
          arg.value("access", actual_accesses.value(name, Json("read")));
      report["args"].push_back({{"tensor", std::move(description)}});
    } else {
      report["args"].push_back(Report(arg));
    }
  }
  return report;
}

Report error_report(const Json& spec, const std::string& error) {
  Report report = case_header(spec);
  report["error"] = error;
  Report throughput = throughput_report(
      spec,
      {std::nullopt, std::nullopt, {"No timing measurements"}},
      1,
      {},
      {});
  if (!throughput.empty()) {
    report["throughput"] = std::move(throughput);
  }
  return report;
}

} // namespace vkcompute::shader_profiling
