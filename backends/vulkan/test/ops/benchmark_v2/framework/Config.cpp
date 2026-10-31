// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Sweep.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace vkcompute::shader_profiling {

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

uint32_t positive_integer(const Json& value, const std::string& name) {
  require(value.is_number_integer(), name + " must be an integer");
  const double number = value.get<double>();
  require(number > 0 && number <= UINT32_MAX, name + " is out of range");
  return value.get<uint32_t>();
}

std::string
normalize_tensor(Json& tensors, Json tensor, const std::string& default_name) {
  require(tensor.is_object(), "Tensor properties must be an object");
  if (tensor.contains("ref")) {
    require(tensor.size() == 1, "Tensor references accept only ref");
    require(
        tensor.at("ref").is_string() &&
            !tensor.at("ref").get<std::string>().empty(),
        "ref must be a nonempty tensor name");
    return tensor.at("ref").get<std::string>();
  }
  require(tensor.contains("sizes"), "Tensor requires sizes or ref");
  const Json name = tensor.value("name", Json(default_name));
  require(
      name.is_string() && !name.get<std::string>().empty(),
      "Tensor name must be a nonempty string");
  const std::string tensor_name = name.get<std::string>();
  require(
      !tensors.contains(tensor_name), "Duplicate tensor name: " + tensor_name);
  tensor.erase("name");
  tensors[tensor_name] = std::move(tensor);
  return tensor_name;
}

Json normalize_shader_config(const Json& spec) {
  require(
      !spec.contains("tensors"),
      "Declare shader tensor properties in bindings, not tensors");
  require(
      spec.contains("bindings") && spec.at("bindings").is_array(),
      "bindings must be an array in shader binding order");
  Json normalized = spec;
  normalized["tensors"] = Json::object();
  auto& bindings = normalized["bindings"];
  for (size_t i = 0; i < bindings.size(); ++i) {
    Json& binding = bindings.at(i);
    require(binding.is_object(), "Each binding must be an object");
    int kinds = binding.contains("meta_ubo") + binding.contains("uniform");
    const char* type = nullptr;
    for (const char* candidate : {"buffer", "texture2d", "texture3d"}) {
      if (binding.contains(candidate)) {
        ++kinds;
        type = candidate;
      }
    }
    require(
        kinds == 1,
        "Each binding requires exactly one of buffer, texture2d, texture3d, meta_ubo, uniform");
    if (type == nullptr) {
      continue;
    }
    require(
        binding.size() == 1,
        "Put tensor properties inside the binding type object");
    Json tensor = binding.at(type);
    require(tensor.is_object(), "Tensor binding properties must be an object");
    require(
        !tensor.contains("storage"),
        "Binding type determines storage; omit storage");
    const Json access = tensor.value("access", Json("read"));
    tensor.erase("access");
    if (!tensor.contains("ref")) {
      tensor["storage"] = type;
    }
    const std::string tensor_name = normalize_tensor(
        normalized["tensors"],
        std::move(tensor),
        "binding_" + std::to_string(i));
    binding = {{"tensor", tensor_name}, {"storage", type}, {"access", access}};
  }
  return normalized;
}

Json normalize_operator_config(const Json& spec) {
  require(
      !spec.contains("tensors"),
      "Declare operator tensor properties in args_sweep, not tensors");
  require(
      !spec.contains("inputs") && !spec.contains("outputs"),
      "Use args_sweep instead of inputs and outputs for operators");
  require(
      spec.contains("args") && spec.at("args").is_array() &&
          !spec.at("args").empty(),
      "Operator args must be a nonempty ordered array");
  Json normalized = spec;
  normalized["tensors"] = Json::object();
  auto& args = normalized["args"];
  for (size_t i = 0; i < args.size(); ++i) {
    Json& arg = args.at(i);
    if (!arg.contains("tensor")) {
      continue;
    }
    require(arg.size() == 1, "Each argument requires a single typed value");
    Json tensor = arg.at("tensor");
    require(tensor.is_object(), "Tensor properties must be an object");
    const bool has_access = tensor.contains("access");
    const Json access = tensor.value("access", Json("read"));
    tensor.erase("access");
    const std::string name = normalize_tensor(
        normalized["tensors"], std::move(tensor), "arg_" + std::to_string(i));
    arg = {{"tensor", name}};
    if (has_access) {
      arg["access"] = access;
    }
  }
  return normalized;
}

void validate_config(const Json& spec) {
  require(spec.is_object(), "Each config must be an object");
  require(
      spec.contains("shader") != spec.contains("operator"),
      "Specify exactly one of shader or operator");
  const bool is_operator = spec.contains("operator");
  const char* kind = is_operator ? "operator" : "shader";
  require(
      !spec.contains("global_size") && !spec.contains("local_size"),
      "Use gwg_size and lwg_size for workgroup settings");
  require(
      spec.at(kind).is_string(),
      std::string(kind) + " must be a registry name");
  if (!is_operator) {
    for (const char* key : {"gwg_size", "lwg_size"}) {
      require(spec.contains(key), std::string("Missing ") + key);
      if (spec.at(key).is_string()) {
        require(
            !spec.at(key).get<std::string>().empty(),
            std::string(key) + " picker name must not be empty");
        continue;
      }
      require(
          spec.contains(key) && spec.at(key).is_array() &&
              spec.at(key).size() == 3,
          std::string(key) +
              " must be a picker name or three positive integers");
      for (const Json& dim : spec.at(key)) {
        positive_integer(dim, key);
      }
    }
  }
  const Json tensors = spec.value("tensors", Json::object());
  require(tensors.is_object(), "tensors must be an object keyed by name");
  for (const auto& item : tensors.items()) {
    const Json& tensor = item.value();
    require(
        tensor.is_object() && tensor.contains("sizes"),
        "Tensor requires sizes");
    tensor_numel(tensor.at("sizes"));
    const std::string dtype = tensor.value("dtype", "float32");
    require(
        dtype == "float32" || dtype == "float16" || dtype == "int32" ||
            dtype == "int8" || dtype == "uint8" || dtype == "int8x4",
        "Unsupported tensor dtype: " + dtype);
    const std::string storage = tensor.value("storage", "buffer");
    require(
        storage == "buffer" || storage == "texture2d" ||
            storage == "texture3d" || (is_operator && storage == "constant"),
        "Unsupported tensor storage: " + storage);
    if (storage != "buffer" && storage != "constant") {
      require(
          tensor.at("sizes").size() <= 4,
          "Textures support at most four dimensions");
    }
    const std::string layout = tensor.value("layout", "width");
    require(
        layout == "width" || layout == "height" || layout == "channels" ||
            layout == "4w" || layout == "4c" || layout == "4w4c" ||
            layout == "4h4w" || layout == "4c1w",
        "Unsupported tensor layout: " + layout);
    const bool packed_layout =
        layout != "width" && layout != "height" && layout != "channels";
    require(
        (dtype == "int8x4") == packed_layout,
        "int8x4 requires a packed int8 layout; other dtypes require width, height, or channels");
    if (dtype == "int8x4") {
      require(
          (storage == "buffer" || storage == "constant") &&
              tensor.at("sizes").size() <= 4,
          "int8x4 requires buffer or constant storage with at most four dimensions");
    }
    require(
        !tensor.contains("constant"),
        "Use storage: constant instead of the tensor constant flag");
    validate_data_initializer(tensor, tensor_numel(tensor.at("sizes")));
  }
  if (is_operator) {
    for (const char* key :
         {"gwg_size",
          "lwg_size",
          "lwg_size_sweep",
          "bindings",
          "specialization_constants",
          "push_constants"}) {
      require(
          !spec.contains(key),
          std::string(key) + " is only valid for raw shaders");
    }
    require(
        spec.contains("args") && spec.at("args").is_array(),
        "Operator args must be an ordered argument array");
    for (const Json& arg : spec.at("args")) {
      if (arg.is_null()) {
        continue;
      }
      require(arg.is_object(), "Each argument must be null or a typed value");
      if (arg.contains("tensor")) {
        require(
            arg.size() == 1 || (arg.size() == 2 && arg.contains("access")),
            "Tensor argument accepts only tensor and optional access");
        const Json& value = arg.at("tensor");
        require(
            value.is_string() && tensors.contains(value.get<std::string>()),
            "Unknown argument tensor");
        const Json access = arg.value("access", Json("read"));
        require(
            access == "read" || access == "write" || access == "read_write",
            "Tensor access must be read, write, or read_write");
        if (access != "read") {
          require(
              tensors.at(value.get<std::string>()).value("storage", "buffer") !=
                  "constant",
              "Writable tensors cannot be constant");
        }
        continue;
      }
      require(
          arg.size() == 1,
          "Each argument must be null or a single typed value");
      const auto item = arg.items().begin();
      const std::string key = item.key();
      const Json& value = item.value();
      if (key == "int") {
        require(
            value.is_number_integer() && value.get<double>() >= INT32_MIN &&
                value.get<double>() <= INT32_MAX,
            "int argument must fit int32");
      } else if (key == "float") {
        require(
            value.is_number() && std::isfinite(value.get<double>()),
            "float argument must be finite");
      } else if (key == "bool") {
        require(value.is_boolean(), "bool argument must be boolean");
      } else if (key == "string") {
        require(value.is_string(), "string argument must be a string");
      } else if (key == "int_list") {
        require(value.is_array(), "int_list must be an array");
        for (const Json& number : value) {
          require(
              number.is_number_integer() && number.get<double>() >= INT32_MIN &&
                  number.get<double>() <= INT32_MAX,
              "int_list values must fit int32");
        }
      } else {
        require(false, "Unsupported operator argument type: " + key);
      }
    }
  } else {
    require(
        !spec.contains("route") && !spec.contains("route_sweep"),
        "route and route_sweep are only valid for operators");
    require(
        !spec.contains("args") && !spec.contains("args_sweep") &&
            !spec.contains("sweep_variables"),
        "args_sweep is only valid for operators");
    require(
        spec.contains("bindings") && spec.at("bindings").is_array(),
        "bindings must be an array in shader binding order");
    for (const Json& binding : spec.at("bindings")) {
      require(binding.is_object(), "Each binding must be an object");
      const int kinds = binding.contains("tensor") +
          binding.contains("meta_ubo") + binding.contains("uniform");
      require(
          kinds == 1,
          "Each binding requires exactly one of tensor, meta_ubo, uniform");
      for (const char* key : {"tensor", "meta_ubo"}) {
        if (binding.contains(key)) {
          require(
              binding.at(key).is_string() &&
                  tensors.contains(binding.at(key).get<std::string>()),
              std::string(key) + " references an unknown tensor");
        }
      }
      if (binding.contains("tensor")) {
        require(
            tensors.at(binding.at("tensor").get<std::string>()).at("storage") ==
                binding.at("storage"),
            "Tensor reference binding type does not match the referenced tensor");
      }
      const std::string access = binding.value("access", "read");
      require(
          access == "read" || access == "write" || access == "read_write",
          "Unsupported binding access: " + access);
    }
    if (spec.contains("specialization_constants")) {
      require(
          spec.at("specialization_constants").is_array(),
          "specialization_constants must be an array");
      for (const Json& constant : spec.at("specialization_constants")) {
        require(
            constant.is_object(),
            "Each specialization constant must be an object");
        if (constant.contains("layout")) {
          require(
              constant.at("layout").is_string() &&
                  tensors.contains(constant.at("layout").get<std::string>()),
              "layout references an unknown tensor binding");
        }
      }
    }
    if (spec.contains("outputs")) {
      require(
          spec.at("outputs").is_array(),
          "outputs must be an array of tensor binding names");
      for (const Json& name : spec.at("outputs")) {
        require(
            name.is_string() && tensors.contains(name.get<std::string>()),
            "Unknown output tensor binding");
      }
    }
  }
  if (spec.contains("dump_outputs")) {
    require(
        spec.at("dump_outputs").is_boolean(), "dump_outputs must be boolean");
  }
  for (const char* key :
       {"show_memory_throughput", "show_compute_throughput"}) {
    if (spec.contains(key)) {
      require(spec.at(key).is_boolean(), std::string(key) + " must be boolean");
    }
  }
  if (spec.contains("reference")) {
    require(
        spec.at("reference").is_string() &&
            !spec.at("reference").get<std::string>().empty(),
        "reference must be a nonempty function name");
  }
  if (spec.contains("validation_method")) {
    const Json& method = spec.at("validation_method");
    require(method.is_object(), "validation_method must be an object");
    require(
        method.contains("function") && method.at("function") == "is_close",
        "validation_method.function must be is_close");
    for (const auto& option : method.items()) {
      require(
          option.key() == "function" || option.key() == "atol" ||
              option.key() == "rtol" || option.key() == "outputs",
          "validation_method accepts only function, atol, rtol, outputs");
      if (option.key() == "atol" || option.key() == "rtol") {
        require(
            option.value().is_number() &&
                std::isfinite(option.value().get<double>()) &&
                option.value().get<double>() >= 0,
            "Validation tolerance must be finite and nonnegative");
      }
    }
    validation_output_names(spec);
  }
  for (const char* key : {"benchmark_runs", "max_dispatches"}) {
    if (spec.contains(key)) {
      const uint32_t number = positive_integer(spec.at(key), key);
      require(number <= 65536, std::string(key) + " must not exceed 65536");
    }
  }
  if (spec.contains("warmup_runs")) {
    require(
        spec.at("warmup_runs").is_number_integer() &&
            spec.at("warmup_runs").get<double>() >= 0 &&
            spec.at("warmup_runs").get<double>() <= 65536,
        "warmup_runs must be in [0, 65536]");
  }
  if (spec.contains("target_ms")) {
    require(
        spec.at("target_ms").is_number() &&
            spec.at("target_ms").get<double>() > 0 &&
            spec.at("target_ms").get<double>() <= 10000,
        "target_ms must be in (0, 10000]");
  }
}

} // namespace

std::set<std::string> validation_output_names(const Json& spec) {
  const Json method = spec.value("validation_method", Json::object());
  std::set<std::string> names;
  if (!method.contains("outputs"))
    return names;
  const Json& selected = method.at("outputs");
  require(
      selected.is_array() && !selected.empty(),
      "validation_method.outputs must be a nonempty list");
  for (const Json& name : selected) {
    require(
        name.is_string() && !name.get<std::string>().empty() &&
            spec.at("tensors").contains(name.get<std::string>()),
        "validation_method.outputs references an unknown tensor");
    const std::string value = name;
    require(
        names.insert(value).second, "Duplicate validation output: " + value);
    require(
        spec.at("tensors").at(value).value("storage", "buffer") != "constant",
        "Validation output must be a GPU tensor: " + value);
  }
  return names;
}

std::vector<Json> parse_configs(const Json& document) {
  std::vector<Json> definitions;
  if (document.is_array()) {
    require(!document.empty(), "Config list must not be empty");
    require(
        document.size() <= 65536, "Config list must not exceed 65536 cases");
    definitions = document.get<std::vector<Json>>();
  } else {
    definitions.push_back(document);
  }
  std::vector<Json> configs;
  for (const Json& definition : definitions) {
    require(definition.is_object(), "Each config must be an object");
    require(
        !definition.contains("force_narrow_int4_tile"),
        "force_narrow_int4_tile is not supported by benchmark_v2");
    for (const char* key :
         {"force_resize",
          "invocations_per_batch",
          "gpu_timing",
          "shader_filter",
          "requirements",
          "device_variables"}) {
      require(
          !definition.contains(key),
          std::string(key) + " is not supported by benchmark_v2");
    }
    require(
        !definition.contains("implementation") &&
            !definition.contains("implementation_sweep"),
        "Use route and route_sweep for operator route selection");
    require(
        definition.contains("operator") + definition.contains("shader") +
                definition.contains("shader_sweep") +
                definition.contains("check") ==
            1,
        "Specify exactly one of operator, shader, shader_sweep, or check");
    if (definition.contains("check")) {
      require(
          definition.at("check").is_string() &&
              !definition.at("check").get<std::string>().empty(),
          "check must be a nonempty registry name");
      require(
          !definition.contains("args_sweep") && !definition.contains("args") &&
              !definition.contains("bindings") &&
              !definition.contains("tensors"),
          "Standalone checks define their inputs in parameters or operator_case");
      configs.push_back(definition);
      continue;
    }
    const bool is_operator = definition.contains("operator");
    const auto expanded = is_operator ? expand_operator_sweep(definition)
                                      : expand_shader_sweep(definition);
    require(
        expanded.size() <= 65536 - configs.size(),
        "Config list must not exceed 65536 cases");
    for (Json config : expanded) {
      if (is_operator) {
        config = normalize_operator_config(config);
      } else {
        config = normalize_shader_config(config);
      }
      validate_config(config);
      configs.push_back(std::move(config));
    }
  }
  return configs;
}

size_t tensor_numel(const Json& sizes) {
  require(
      sizes.is_array() && !sizes.empty() && sizes.size() <= 8,
      "sizes must contain one to eight positive integers");
  size_t count = 1;
  for (const Json& size : sizes) {
    const uint32_t dim = positive_integer(size, "Tensor dimension");
    require(count <= INT32_MAX / dim, "Tensor element count must fit int32");
    count *= dim;
  }
  return count;
}

uint32_t
choose_dispatch_count(double probe_us, double target_us, uint32_t limit) {
  require(
      limit > 0 && std::isfinite(probe_us) && probe_us >= 0 &&
          std::isfinite(target_us) && target_us > 0,
      "Invalid calibration parameters");
  const double count = std::ceil(target_us / std::max(probe_us, 0.001));
  return static_cast<uint32_t>(
      std::clamp(count, 1.0, static_cast<double>(limit)));
}

} // namespace vkcompute::shader_profiling
