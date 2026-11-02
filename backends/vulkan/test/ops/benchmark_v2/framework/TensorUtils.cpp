// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/DataInitializer.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Int8x4Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace vkcompute::shader_profiling {

namespace {

const std::map<std::string, vkapi::ScalarType> kScalarTypes = {
    {"float32", vkapi::kFloat},
    {"float16", vkapi::kHalf},
    {"int32", vkapi::kInt},
    {"int8", vkapi::kChar},
    {"uint8", vkapi::kByte},
    {"int8x4", vkapi::kInt8x4}};

const std::map<std::string, utils::GPUMemoryLayout> kMemoryLayouts = {
    {"width", utils::kWidthPacked},
    {"height", utils::kHeightPacked},
    {"channels", utils::kChannelsPacked},
    {"4w", utils::kPackedInt8_4W},
    {"4c", utils::kPackedInt8_4C},
    {"4w4c", utils::kPackedInt8_4W4C},
    {"4h4w", utils::kPackedInt8_4H4W},
    {"4c1w", utils::kPackedInt8_4C1W}};

} // namespace

vkapi::ScalarType scalar_type(const std::string& name) {
  return kScalarTypes.at(name);
}

vkapi::BufferBindInfo
meta_ubo(ComputeGraph& graph, ValueRef tensor, const std::string& field) {
  if (field == "meta") {
    return graph.meta_ubo(tensor);
  }
  if (field == "sizes") {
    return graph.sizes_ubo(tensor);
  }
  if (field == "strides") {
    return graph.strides_ubo(tensor);
  }
  if (field == "numel") {
    return graph.numel_ubo(tensor);
  }
  if (field == "logical_limits") {
    return graph.logical_limits_ubo(tensor);
  }
  throw std::invalid_argument("Unknown tensor metadata field: " + field);
}

TensorData prepare_tensor_data(const Json& spec) {
  const Json descriptions = spec.value("tensors", Json::object());
  auto host_data = make_tensor_data(descriptions);
  for (const auto& item : descriptions.items()) {
    const Json& tensor = item.value();
    if (tensor.value("storage", "buffer") != "constant" ||
        tensor.value("dtype", "float32") != "float16") {
      continue;
    }
    auto& data = host_data.at(item.key());
    std::vector<uint8_t> half_data(data.size() / 2);
    for (size_t i = 0; i < data.size() / sizeof(float); ++i) {
      float value;
      std::memcpy(&value, data.data() + i * sizeof(float), sizeof(float));
      const executorch::runtime::etensor::Half half(value);
      std::memcpy(half_data.data() + i * sizeof(half), &half, sizeof(half));
    }
    data = std::move(half_data);
  }
  return host_data;
}

std::map<std::string, ValueRef> create_tensors(
    const Json& spec,
    ComputeGraph& graph,
    const TensorData& host_data,
    bool with_staging) {
  std::map<std::string, ValueRef> tensors;
  const Json descriptions = spec.value("tensors", Json::object());
  for (const auto& item : descriptions.items()) {
    const Json& tensor = item.value();
    const auto sizes = tensor.at("sizes").get<std::vector<int64_t>>();
    const auto dtype = scalar_type(tensor.value("dtype", "float32"));
    const auto& data = host_data.at(item.key());
    const std::string storage = tensor.value("storage", "buffer");
    const bool constant = storage == "constant";
    const vkapi::ScalarType source_dtype = dtype == vkapi::kHalf && !constant
        ? vkapi::kFloat
        : dtype == vkapi::kInt8x4 ? vkapi::kChar
                                  : dtype;
    const ValueRef source =
        graph.add_tensorref(sizes, source_dtype, data.data());
    if (constant) {
      tensors.emplace(item.key(), source);
      continue;
    }
    const auto storage_type = storage == "buffer" ? utils::kBuffer
        : storage == "texture2d"                  ? utils::kTexture2D
                                                  : utils::kTexture3D;
    const ValueRef ref = graph.add_tensor(
        sizes,
        dtype,
        storage_type,
        kMemoryLayouts.at(tensor.value("layout", "width")));
    tensors.emplace(item.key(), ref);
    if (with_staging) {
      graph.set_input_tensor(ref, source_dtype);
    } else if (dtype == vkapi::kInt8x4) {
      add_prepack_int8x4_buffer_node(graph, source, ref);
    } else {
      add_prepack_standard_node(graph, source, ref);
    }
  }
  return tensors;
}

Json actual_tensor_dtypes(
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors) {
  Json result = Json::object();
  for (const auto& tensor : tensors) {
    const auto dtype = graph.dtype_of(tensor.second);
    std::string name = vkapi::to_string(dtype);
    for (const auto& entry : kScalarTypes) {
      if (entry.second == dtype) {
        name = entry.first;
        break;
      }
    }
    result[tensor.first] = name;
  }
  return result;
}

Json dispatch_tensor_bindings(ComputeGraph& graph, const ExecuteNode& node) {
  Json bindings = Json::object();
  size_t index = 0;
  for (const auto& group : node.args()) {
    for (ValueRef ref : group.refs) {
      if (ref < 0 || !graph.val_is_tensor(ref)) {
        ++index;
        continue;
      }
      const std::string name = "value_" + std::to_string(ref);
      const std::string storage = graph.is_buffer_storage(ref) ? "buffer"
          : graph.storage_type_of(ref) == utils::kTexture2D    ? "texture2d"
                                                               : "texture3d";
      const auto access = group.access;
      bindings["binding_" + std::to_string(index)] = {
          {"name", name},
          {"binding_index", index},
          {"type", storage},
          {"sizes", graph.sizes_of(ref)},
          {"dtype", actual_tensor_dtypes(graph, {{name, ref}}).at(name)},
          {"access",
           access & vkapi::kWrite
               ? (access & vkapi::kRead ? "read_write" : "write")
               : "read"}};
      ++index;
    }
  }
  return bindings;
}

} // namespace vkcompute::shader_profiling
