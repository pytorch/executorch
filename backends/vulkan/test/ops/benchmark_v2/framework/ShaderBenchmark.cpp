// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ShaderBenchmark.h>

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Timing.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Validation.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/WorkgroupUtils.h>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkcompute::shader_profiling {

namespace {

std::vector<uint8_t> packed_blocks(const Json& blocks) {
  std::vector<uint8_t> bytes;
  for (const Json& block : blocks) {
    if (block.value("dtype", "float32") == "float16") {
      throw std::invalid_argument("Parameter blocks do not support float16");
    }
    const auto data = make_data(block, block.at("data").size());
    bytes.insert(bytes.end(), data.begin(), data.end());
  }
  if (bytes.size() % 4 != 0) {
    throw std::invalid_argument(
        "Parameter blocks must be padded to a multiple of four bytes");
  }
  return bytes;
}

} // namespace

Report benchmark_shader(
    const Json& spec,
    const TensorData& host_data,
    bool validation_only) {
  GraphConfig config;
  config.force_resize = true;
  ComputeGraph graph(config);
  const auto tensors = create_tensors(spec, graph, host_data, validation_only);

  const std::string shader_name = spec.at("shader");
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(shader_name);
  // JSON describes shader invocations, rather than logical output elements.
  shader.out_tile_size = LocalWorkGroup(1, 1, 1);
  api::Context& context = *graph.context();
  context.check_device_capabilities(shader);
  const Json& bindings = spec.at("bindings");
  if (bindings.size() != shader.kernel_layout.size()) {
    throw std::invalid_argument(
        "Binding count does not match shader signature");
  }
  std::vector<vkapi::VulkanBuffer> uniforms;
  std::map<size_t, vkapi::BufferBindInfo> params;
  std::map<ValueRef, VkDescriptorType> image_bindings;
  for (size_t i = 0; i < bindings.size(); ++i) {
    const Json& binding = bindings.at(i);
    const VkDescriptorType expected = shader.kernel_layout.at(i);
    if (binding.contains("tensor")) {
      const ValueRef tensor =
          tensors.at(binding.at("tensor").get<std::string>());
      const bool buffer = graph.is_buffer_storage(tensor);
      const std::string access = binding.value("access", "read");
      if ((buffer && expected != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
          (!buffer && expected != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
           expected != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
          (expected == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
           access != "read")) {
        throw std::invalid_argument(
            "Tensor storage/access does not match shader binding " +
            std::to_string(i));
      }
      if (!buffer) {
        const auto previous = image_bindings.emplace(tensor, expected);
        if (!previous.second && previous.first->second != expected) {
          throw std::invalid_argument(
              "Sampled and storage bindings must use separate texture tensors");
        }
      }
    } else {
      if (expected != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
        throw std::invalid_argument(
            "Expected uniform buffer at binding " + std::to_string(i));
      }
      if (binding.contains("meta_ubo")) {
        params.emplace(
            i,
            meta_ubo(
                graph,
                tensors.at(binding.at("meta_ubo").get<std::string>()),
                binding.value("field", "meta")));
      } else {
        const auto bytes = packed_blocks(binding.at("uniform"));
        if (bytes.empty()) {
          throw std::invalid_argument("Uniform buffer must not be empty");
        }
        uniforms.push_back(
            context.adapter_ptr()->vma().create_uniform_buffer(bytes.size()));
        vkapi::MemoryMap map(uniforms.back(), vkapi::kWrite);
        std::memcpy(map.data<uint8_t>(), bytes.data(), bytes.size());
        params.emplace(i, vkapi::BufferBindInfo(uniforms.back()));
      }
    }
  }
  vkapi::SpecVarList constants;
  for (const Json& constant :
       spec.value("specialization_constants", Json::array())) {
    if (constant.contains("layout")) {
      constants.append(vkapi::SpecVar(graph.hashed_layout_of(
          tensors.at(constant.at("layout").get<std::string>()))));
    } else {
      const std::string dtype = constant.value("dtype", "int32");
      if (dtype == "bool") {
        constants.append(vkapi::SpecVar(constant.at("value").get<bool>()));
      } else {
        const auto bytes =
            make_data({{"dtype", dtype}, {"data", {constant.at("value")}}}, 1);
        if (bytes.size() != 4) {
          throw std::invalid_argument(
              "Specialization constants must be 32-bit or bool");
        }
        vkapi::SpecVar value;
        if (dtype == "float32") {
          value.type = vkapi::SpecVar::Type::FLOAT;
        } else if (dtype == "int32") {
          value.type = vkapi::SpecVar::Type::INT;
        } else if (dtype == "uint32") {
          value.type = vkapi::SpecVar::Type::UINT;
        } else {
          throw std::invalid_argument(
              "Unsupported specialization constant dtype");
        }
        std::memcpy(&value.value, bytes.data(), 4);
        constants.append(value);
      }
    }
  }
  const auto push_constants =
      packed_blocks(spec.value("push_constants", Json::array()));
  if (push_constants.size() > 128) {
    throw std::invalid_argument(
        "Prototype supports up to 128 bytes of push constants");
  }
  const auto workgroups = pick_workgroup_sizes(spec, graph, shader, tensors);
  const GlobalWorkGrid& gwg = workgroups.gwg;
  const LocalWorkGroup& lwg = workgroups.lwg;

  const size_t input_nodes = graph.execute_nodes().size();
  std::map<std::string, ValueRef> outputs;
  if (validation_only) {
    Json accesses = Json::object();
    for (const auto& binding : bindings) {
      if (binding.contains("tensor") &&
          binding.value("access", "read") != "read") {
        accesses[binding.at("tensor").get<std::string>()] =
            binding.at("access");
      }
    }
    outputs = add_validation_outputs(spec, graph, tensors, accesses);
  }

  graph.update_descriptor_counts(shader, true);
  graph.prepare();
  graph.prepack();
  graph.propagate_resize();
  if (validation_only) {
    copy_validation_inputs(spec, graph, tensors, host_data);
  }

  const VkDescriptorSetLayout shader_layout =
      context.shader_layout_cache().retrieve(shader.kernel_layout);
  const VkPipelineLayout pipeline_layout =
      context.pipeline_layout_cache().retrieve(
          shader_layout, static_cast<uint32_t>(push_constants.size()));
  vkapi::SpecVarList pipeline_constants = {SV(lwg[0]), SV(lwg[1]), SV(lwg[2])};
  pipeline_constants.append(constants);
  const VkPipeline pipeline = context.pipeline_cache().retrieve(
      {pipeline_layout,
       context.shader_cache().retrieve(shader),
       pipeline_constants,
       vkapi::resolve_required_subgroup_size(shader, context.adapter_ptr())});

  vkapi::VulkanFence fence(context.device());
  auto bind_resources = [&](vkapi::DescriptorSet& descriptors,
                            vkapi::PipelineBarrier& barrier) {
    for (size_t i = 0; i < bindings.size(); ++i) {
      const Json& binding = bindings.at(i);
      if (binding.contains("tensor")) {
        const std::string access = binding.value("access", "read");
        const vkapi::MemoryAccessFlags flags = access == "read" ? vkapi::kRead
            : access == "write"                                 ? vkapi::kWrite
                                : vkapi::kReadWrite;
        graph.bind_value_to_descriptor_set(
            tensors.at(binding.at("tensor").get<std::string>()),
            barrier,
            flags,
            descriptors,
            static_cast<uint32_t>(i));
      } else {
        descriptors.bind(static_cast<uint32_t>(i), params.at(i));
      }
    }
  };
  if (!validation_only) {
    // Initial image layout transitions must not be replayed.
    context.set_cmd();
    auto initial_descriptors = context.descriptor_pool().get_descriptor_set(
        shader_layout, shader.kernel_layout);
    vkapi::PipelineBarrier transitions{};
    bind_resources(initial_descriptors, transitions);
    context.register_barrier(transitions);
    context.submit_cmd_to_gpu(fence.get_submit_handle(), true);
    fence.wait();
  }

  auto encode = [&](uint32_t count) {
    auto descriptors = context.descriptor_pool().get_descriptor_set(
        shader_layout, shader.kernel_layout);
    vkapi::PipelineBarrier initial{};
    bind_resources(descriptors, initial);
    context.register_barrier(initial);
    for (uint32_t i = 0; i < count; ++i) {
      context.extract_cmd().bind_pipeline(pipeline, pipeline_layout, lwg);
      // Serialize repeats, including across submissions of the cached buffer.
      vkapi::PipelineBarrier barrier{};
      barrier.stage = {
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
      bind_resources(descriptors, barrier);
      context.register_shader_dispatch(
          descriptors,
          barrier,
          shader,
          gwg,
          lwg,
          push_constants.data(),
          static_cast<uint32_t>(push_constants.size()));
    }
  };
  if (validation_only) {
    context.set_cmd();
    for (size_t i = 0; i < input_nodes; ++i) {
      graph.execute_nodes()[i]->encode(&graph);
    }
    encode(1);
    for (size_t i = input_nodes; i < graph.execute_nodes().size(); ++i) {
      graph.execute_nodes()[i]->encode(&graph);
    }
    context.submit_cmd_to_gpu(fence.get_submit_handle(), true);
    fence.wait();
  }
  const Json definitions =
      spec.value("specialization_constants", Json::array());
  Report specialization = Report::object();
  for (uint32_t i = 0; i < constants.size(); ++i) {
    const vkapi::SpecVar& value = constants.at(i);
    Report entry;
    switch (value.type) {
      case vkapi::SpecVar::Type::FLOAT:
        entry = {{"dtype", "float32"}, {"value", value.value.as_float}};
        break;
      case vkapi::SpecVar::Type::INT:
        entry = {{"dtype", "int32"}, {"value", value.value.as_int32}};
        break;
      case vkapi::SpecVar::Type::UINT:
        entry = {{"dtype", "uint32"}, {"value", value.value.as_uint32}};
        break;
      case vkapi::SpecVar::Type::BOOL:
        entry = {{"dtype", "bool"}, {"value", value.value.as_bool}};
        break;
    }
    if (definitions.at(i).contains("layout")) {
      entry["layout"] = definitions.at(i).at("layout");
    }
    specialization[std::to_string(i + 3)] = std::move(entry);
  }
  Report result = shader_report(
      spec,
      {gwg[0], gwg[1], gwg[2]},
      {lwg[0], lwg[1], lwg[2]},
      actual_tensor_dtypes(graph, tensors),
      specialization);
  if (validation_only) {
    result["validation"] =
        validate_outputs(spec, graph, tensors, outputs, host_data);
  } else {
    result.update(measure(spec, graph, encode, Json(result)));
  }
  return result;
}

} // namespace vkcompute::shader_profiling
