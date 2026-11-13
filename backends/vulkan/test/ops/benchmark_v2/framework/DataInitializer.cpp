// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/DataInitializer.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace vkcompute::shader_profiling {
namespace {
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::invalid_argument(message);
}

template <typename T>
double numeric_value(const Json& value) {
  require(value.is_number(), "Input values must be numbers");
  const double number = value.get<double>();
  require(
      std::isfinite(number) &&
          number >= double(std::numeric_limits<T>::lowest()) &&
          number <= double(std::numeric_limits<T>::max()),
      "Input value is out of dtype range");
  if constexpr (std::is_integral_v<T>)
    require(
        std::trunc(number) == number, "Integer input must not have a fraction");
  return number;
}
uint32_t random_seed(const Json& value) {
  require(
      value.is_number_integer() && value.get<double>() >= 0 &&
          value.get<double>() <= UINT32_MAX,
      "seed must fit uint32");
  return value.get<uint32_t>();
}
void integer_bounds(double lower, double upper) {
  require(
      std::trunc(lower) == lower && std::trunc(upper) == upper,
      "Integer random bounds must not have a fraction");
  require(
      lower >= INT32_MIN && upper <= UINT32_MAX && upper - lower <= UINT32_MAX,
      "Integer random bounds require at most a uint32-sized range");
}

template <typename T>
std::vector<uint8_t>
pack_data(const Json& spec, size_t count, uint32_t seed, bool generate) {
  require(count <= SIZE_MAX / sizeof(T), "Input byte size overflows");
  const Json data = spec.value("data", Json());
  const Json* values = nullptr;
  bool repeat = false, random = false, random_integer = false;
  double lower = 0, upper = 0;
  uint64_t modulus = 0;
  Json fill = spec.value("fill", Json(0));
  if (spec.contains("data")) {
    require(!spec.contains("fill"), "Specify either data or fill");
    require(
        data.is_array() || data.is_object(),
        "data must be an initializer object or array");
    if (data.is_array()) {
      require(
          data.size() == count,
          "data must contain exactly one value per element");
      values = &data;
    } else if (data.contains("repeat")) {
      require(
          data.size() == 1 && data.at("repeat").is_array() &&
              !data.at("repeat").empty(),
          "repeat requires a nonempty array");
      values = &data.at("repeat");
      repeat = true;
    } else if (
        data.contains("row_sum") || data.contains("column_sum") ||
        data.contains("q4_group_sum")) {
      require(data.size() == 1, "Derived initializers require a single method");
      if (data.contains("q4_group_sum")) {
        const Json& method = data.at("q4_group_sum");
        require(
            method.is_object() && method.size() == 2 &&
                method.contains("tensor") && method.contains("group_size"),
            "q4_group_sum requires tensor and group_size");
        require(
            method.at("tensor").is_string() &&
                !method.at("tensor").get<std::string>().empty(),
            "q4_group_sum requires a source tensor name");
        const Json& group = method.at("group_size");
        require(
            group.is_number_integer() && group.get<double>() > 0 &&
                group.get<double>() <= INT32_MAX,
            "q4_group_sum group_size must be a positive int32");
        require(
            (std::is_same_v<T, int32_t> || std::is_same_v<T, float>),
            "q4_group_sum requires int32 or floating point output data");
      } else {
        const char* method =
            data.contains("row_sum") ? "row_sum" : "column_sum";
        require(
            data.at(method).is_string() &&
                !data.at(method).get<std::string>().empty(),
            std::string(method) + " requires a source tensor name");
        require(
            (std::is_same_v<T, int32_t>),
            std::string(method) + " requires int32 output data");
      }
      require(!generate, "Derived initializers require tensor data context");
      return {};
    } else if (data.contains("fill")) {
      require(data.size() == 1, "fill initializer accepts only fill");
      fill = data.at("fill");
    } else if (data.contains("abs_mod")) {
      require(data.size() == 1, "abs_mod accepts only its method object");
      const Json& method = data.at("abs_mod");
      require(
          method.is_object() && method.contains("bounds") &&
              method.contains("mod") &&
              (method.size() == 2 ||
               (method.size() == 3 && method.contains("seed"))),
          "abs_mod requires bounds, mod, and optional seed");
      const Json& bounds = method.at("bounds");
      require(
          bounds.is_array() && bounds.size() == 2,
          "abs_mod bounds require [min, max]");
      lower = numeric_value<int64_t>(bounds.at(0));
      upper = numeric_value<int64_t>(bounds.at(1));
      const Json& mod = method.at("mod");
      require(
          mod.is_number_integer() && mod.get<double>() > 0 &&
              mod.get<double>() <= UINT32_MAX,
          "abs_mod mod must be a positive uint32");
      modulus = mod.get<uint64_t>();
      numeric_value<T>(Json(modulus - 1));
      if (method.contains("seed"))
        seed = random_seed(method.at("seed"));
      random = true;
      random_integer = true;
    } else {
      random_integer = data.contains("rand_uniform_int");
      require(
          random_integer || data.contains("rand_uniform_float"),
          "Unknown data initializer");
      const char* method =
          random_integer ? "rand_uniform_int" : "rand_uniform_float";
      require(
          data.size() == 1 || (data.size() == 2 && data.contains("seed")),
          "Random initializers accept only bounds and optional seed");
      require(
          random_integer || std::is_floating_point_v<T>,
          "Floating point random initializer requires floating point dtype");
      const Json& bounds = data.at(method);
      require(
          bounds.is_array() && bounds.size() == 2,
          "Random initializer requires [min, max]");
      lower = numeric_value<T>(bounds.at(0));
      upper = numeric_value<T>(bounds.at(1));
      if (data.contains("seed"))
        seed = random_seed(data.at("seed"));
      random = true;
    }
  }
  if (values)
    for (const Json& value : *values)
      numeric_value<T>(value);
  else if (!random)
    numeric_value<T>(fill);
  if (random) {
    require(lower <= upper, "Random initializer min must not exceed max");
    if (random_integer)
      integer_bounds(lower, upper);
  }
  if (!generate)
    return {};
  std::mt19937 engine(seed);
  std::vector<uint8_t> bytes(count * sizeof(T));
  for (size_t i = 0; i < count; ++i) {
    double number;
    if (random_integer) {
      const uint64_t span = uint64_t(int64_t(upper) - int64_t(lower)) + 1;
      const uint64_t range = uint64_t{1} << 32, limit = range - range % span;
      uint32_t draw;
      do {
        draw = engine();
      } while (draw >= limit);
      int64_t value = int64_t(lower) + int64_t(uint64_t(draw) % span);
      if (modulus) {
        const uint64_t absolute =
            value < 0 ? uint64_t(-value) : uint64_t(value);
        value = int64_t(absolute % modulus);
      }
      number = double(value);
    } else if (random) {
      const double unit = double(engine()) / (double(UINT32_MAX) + 1);
      number = lower + (upper - lower) * unit;
    } else
      number = numeric_value<T>(
          values ? values->at(repeat ? i % values->size() : i) : fill);
    const T value = static_cast<T>(number);
    std::memcpy(bytes.data() + i * sizeof(T), &value, sizeof(T));
  }
  return bytes;
}

std::vector<uint8_t> build_data(
    const Json& spec,
    size_t count,
    uint32_t seed,
    bool generate,
    size_t depth = 0) {
  require(depth <= 64, "Data initializer nesting exceeds 64 levels");
  const Json data = spec.value("data", Json());
  if (data.is_object() && data.contains("pad_columns")) {
    require(!spec.contains("fill"), "Specify either data or fill");
    require(data.size() == 1, "pad_columns accepts only its method object");
    const Json& method = data.at("pad_columns");
    require(
        method.is_object() && method.size() == 2 && method.contains("source") &&
            method.contains("valid_columns"),
        "pad_columns requires source and valid_columns");
    require(
        spec.contains("sizes") && tensor_numel(spec.at("sizes")) == count,
        "pad_columns requires sizes matching element count");
    const size_t columns = spec.at("sizes").back().get<size_t>();
    const Json& valid = method.at("valid_columns");
    require(
        valid.is_number_integer() && valid.get<double>() >= 0 &&
            valid.get<double>() <= double(columns),
        "pad_columns valid_columns must be between zero and row width");
    const Json& source = method.at("source");
    require(
        source.is_array() || source.is_object(),
        "pad_columns source must be an initializer");
    require(
        !source.is_object() ||
            (!source.contains("row_sum") && !source.contains("column_sum") &&
             !source.contains("q4_group_sum")),
        "pad_columns source must not require tensor context");
    Json nested = spec;
    nested["data"] = source;
    auto result = build_data(nested, count, seed, generate, depth + 1);
    if (generate) {
      const size_t element_size = result.size() / count,
                   valid_columns = valid.get<size_t>();
      for (size_t row = 0; row < count / columns; ++row)
        std::memset(
            result.data() + (row * columns + valid_columns) * element_size,
            0,
            (columns - valid_columns) * element_size);
    }
    return result;
  }
  const std::string dtype = spec.value("dtype", "float32");
  if (dtype == "float32" || dtype == "float16")
    return pack_data<float>(spec, count, seed, generate);
  if (dtype == "int32")
    return pack_data<int32_t>(spec, count, seed, generate);
  if (dtype == "uint32")
    return pack_data<uint32_t>(spec, count, seed, generate);
  if (dtype == "int8" || dtype == "int8x4")
    return pack_data<int8_t>(spec, count, seed, generate);
  if (dtype == "uint8")
    return pack_data<uint8_t>(spec, count, seed, generate);
  throw std::invalid_argument("Unsupported data dtype: " + dtype);
}

template <typename T>
std::vector<uint8_t> integer_sums(
    const std::vector<uint8_t>& source,
    size_t segments,
    size_t count,
    bool columns) {
  require(
      segments > 0 && count >= segments &&
          source.size() % (segments * sizeof(T)) == 0,
      "Derived sum output is too short or source shape is invalid");
  const size_t contributions = source.size() / (segments * sizeof(T));
  std::vector<uint8_t> result(count * sizeof(int32_t));
  for (size_t segment = 0; segment < segments; ++segment) {
    int64_t sum = 0;
    for (size_t i = 0; i < contributions; ++i) {
      const size_t index =
          columns ? i * segments + segment : segment * contributions + i;
      T value;
      std::memcpy(&value, source.data() + index * sizeof(T), sizeof(T));
      sum += value;
    }
    require(
        sum >= INT32_MIN && sum <= INT32_MAX, "Derived sum overflows int32");
    const int32_t value = int32_t(sum);
    std::memcpy(result.data() + segment * sizeof(value), &value, sizeof(value));
  }
  return result;
}
std::vector<uint8_t> sum_integer_tensor(
    const Json& spec,
    const std::vector<uint8_t>& source,
    size_t segments,
    size_t count,
    bool columns) {
  const std::string dtype = spec.value("dtype", "float32");
  if (dtype == "int8" || dtype == "int8x4")
    return integer_sums<int8_t>(source, segments, count, columns);
  if (dtype == "uint8")
    return integer_sums<uint8_t>(source, segments, count, columns);
  if (dtype == "int32")
    return integer_sums<int32_t>(source, segments, count, columns);
  if (dtype == "uint32")
    return integer_sums<uint32_t>(source, segments, count, columns);
  throw std::invalid_argument("Derived sum source must have integer dtype");
}
std::vector<uint8_t> q4_group_sums(
    const Json& spec,
    const Json& source_spec,
    const std::vector<uint8_t>& source) {
  const std::string source_dtype = source_spec.value("dtype", "float32");
  require(
      source_dtype == "int8" || source_dtype == "uint8",
      "q4_group_sum source must contain packed int8/uint8 weights");
  const Json& input_sizes = source_spec.at("sizes");
  const Json& output_sizes = spec.at("sizes");
  require(
      input_sizes.size() == 2 && output_sizes.size() == 2,
      "q4_group_sum requires [N,K/2] weights and [K/group_size,N] sums");
  const size_t channels = input_sizes.at(0).get<size_t>(),
               packed_width = input_sizes.at(1).get<size_t>(),
               width = packed_width * 2;
  const size_t group_size =
      spec.at("data").at("q4_group_sum").at("group_size").get<size_t>();
  require(
      width % group_size == 0,
      "q4_group_sum group_size must divide unpacked weight width");
  const size_t groups = width / group_size,
               output_channels = output_sizes.at(1).get<size_t>();
  require(
      output_sizes.at(0).get<size_t>() == groups && output_channels >= channels,
      "q4_group_sum output shape does not match groups and channels");
  require(
      source.size() == channels * packed_width,
      "q4_group_sum source byte size does not match weights");
  const bool floating = spec.value("dtype", "float32") != "int32";
  std::vector<uint8_t> result(groups * output_channels * sizeof(int32_t));
  for (size_t group = 0; group < groups; ++group)
    for (size_t channel = 0; channel < channels; ++channel) {
      int64_t sum = 0;
      for (size_t k = group * group_size; k < (group + 1) * group_size; ++k) {
        const uint8_t value = source.at(channel * packed_width + k / 2);
        sum += int((value >> ((k % 2) * 4)) & 15) - 8;
      }
      const size_t index = group * output_channels + channel;
      if (floating) {
        const float value = float(sum);
        std::memcpy(
            result.data() + index * sizeof(value), &value, sizeof(value));
      } else {
        require(
            sum >= INT32_MIN && sum <= INT32_MAX,
            "q4_group_sum overflows int32");
        const int32_t value = int32_t(sum);
        std::memcpy(
            result.data() + index * sizeof(value), &value, sizeof(value));
      }
    }
  return result;
}
} // namespace

void validate_data_initializer(const Json& spec, size_t count) {
  build_data(spec, count, 0, false);
}
std::vector<uint8_t> make_data(const Json& spec, size_t count, uint32_t seed) {
  return build_data(spec, count, seed, true);
}
std::map<std::string, std::vector<uint8_t>> make_tensor_data(
    const Json& tensors) {
  std::map<std::string, std::vector<uint8_t>> result;
  std::set<std::string> active;
  std::function<void(const std::string&)> initialize =
      [&](const std::string& name) {
        if (result.count(name))
          return;
        require(tensors.contains(name), "Unknown data source tensor: " + name);
        require(
            active.insert(name).second,
            "Cyclic data initializer reference: " + name);
        const Json& spec = tensors.at(name);
        const size_t count = tensor_numel(spec.at("sizes"));
        validate_data_initializer(spec, count);
        const Json data = spec.value("data", Json());
        if (data.is_object() &&
            (data.contains("row_sum") || data.contains("column_sum") ||
             data.contains("q4_group_sum"))) {
          const bool q4 = data.contains("q4_group_sum"),
                     columns = data.contains("column_sum");
          const std::string source = q4
              ? data.at("q4_group_sum").at("tensor").get<std::string>()
              : data.at(columns ? "column_sum" : "row_sum").get<std::string>();
          initialize(source);
          const Json& source_spec = tensors.at(source);
          if (q4)
            result[name] = q4_group_sums(spec, source_spec, result.at(source));
          else {
            const size_t segments =
                source_spec.at("sizes")
                    .at(columns ? source_spec.at("sizes").size() - 1 : 0)
                    .get<size_t>();
            result[name] = sum_integer_tensor(
                source_spec, result.at(source), segments, count, columns);
          }
        } else {
          // FNV-1a preserves per-tensor seeds across platforms and sweep
          // entries.
          uint32_t seed = 2166136261u;
          for (unsigned char ch : name)
            seed = (seed ^ ch) * 16777619u;
          result[name] = make_data(spec, count, seed);
        }
        active.erase(name);
      };
  for (const auto& tensor : tensors.items())
    initialize(tensor.key());
  return result;
}
} // namespace vkcompute::shader_profiling
