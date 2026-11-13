// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/wmma/WmmaLinear.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkcompute::shader_profiling {

namespace {

// The kernels the graph dispatches in its current state, from the shader
// profiling records; empty without timestamp queries (e.g. SwiftShader).
std::vector<std::string> dispatched_kernels(ComputeGraph& graph) {
  api::Context& context = *graph.context();
  context.flush();
  context.initialize_querypool();
  if (!context.querypool()) {
    return {};
  }
  context.set_shader_profiling_enabled(true);
  context.set_cmd(true);
  context.cmd_reset_querypool();
  for (const auto& node : graph.execute_nodes()) {
    node->encode(&graph);
  }
  vkapi::VulkanFence fence(context.device());
  context.submit_cmd_to_gpu(fence.get_submit_handle(), true);
  fence.wait();
  std::vector<std::string> kernels;
  for (const auto& record : context.querypool().get_shader_timestamp_data()) {
    kernels.push_back(record.kernel_name);
  }
  context.set_shader_profiling_enabled(false);
  context.flush();
  return kernels;
}

// What the run expects of the device, from ETVK_WMMA_EXPECTED_FAMILY rather
// than from the selector under test:
//   xe2, nvidia, none: the device must match that family, every step must have
//     profiling records to check, and a case that does not run (unsupported
//     shader or route) fails instead of reporting SKIP.
//   software: a device without timestamp queries (e.g. SwiftShader); it must
//     match no family, a case that does not run fails as above, and kernels
//     are checked only when records exist.
//   unset: the selector's family is taken as is; records are still required,
//     and a case that does not run reports SKIP.
struct RouteExpectation {
  std::string family; // empty: take the selector's
  bool strict = false;
  bool records_optional = false;
};

RouteExpectation route_expectation() {
  const char* env = std::getenv("ETVK_WMMA_EXPECTED_FAMILY");
  const std::string value = env == nullptr ? "" : env;
  if (value.empty()) {
    return {};
  }
  if (value == "software") {
    return {"none", true, true};
  }
  reference::require(
      value == "xe2" || value == "nvidia" || value == "none",
      "ETVK_WMMA_EXPECTED_FAMILY must be xe2, nvidia, none or software, not " +
          value);
  return {value, true, false};
}

bool is_family_kernel(const std::string& kernel) {
  return kernel.find("_wmma") != std::string::npos;
}

// Bytes of the Workgroup-storage variables of a SPIR-V module.
uint64_t workgroup_bytes(const vkapi::ShaderInfo& shader) {
  const uint32_t* words = shader.src_code.bin;
  const size_t count = shader.src_code.size / sizeof(uint32_t);
  // Result id -> operands (after the result id) of each type and constant.
  std::unordered_map<uint32_t, std::pair<uint32_t, const uint32_t*>> defs;
  std::vector<uint32_t> workgroup_pointer_types;
  for (size_t i = 5; i < count;) {
    const uint32_t length = words[i] >> 16;
    const uint32_t opcode = words[i] & 0xffffu;
    reference::require(length > 0 && i + length <= count, "Malformed SPIR-V");
    const uint32_t* op = words + i + 1;
    if (opcode >= 19 && opcode <= 39) { // OpType*
      defs[op[0]] = {opcode, op + 1};
    } else if (opcode == 43) { // OpConstant: result type, id, value
      defs[op[1]] = {opcode, op + 2};
    } else if (opcode == 59 && op[2] == 4) { // OpVariable, Workgroup
      workgroup_pointer_types.push_back(op[0]);
    }
    i += length;
  }
  std::function<uint64_t(uint32_t)> bytes = [&](const uint32_t type) {
    const auto& [opcode, op] = defs.at(type);
    switch (opcode) {
      case 21: // OpTypeInt
      case 22: // OpTypeFloat
        return uint64_t(op[0] / 8);
      case 23: // OpTypeVector
        return bytes(op[0]) * op[1];
      case 28: // OpTypeArray
        reference::require(
            defs.at(op[1]).first == 43, "Shared array length is not constant");
        return bytes(op[0]) * defs.at(op[1]).second[0];
      default:
        throw std::invalid_argument(
            "Unsupported shared type in " + shader.kernel_name);
    }
  };
  uint64_t total = 0;
  for (const uint32_t pointer : workgroup_pointer_types) {
    total += bytes(defs.at(pointer).second[1]); // OpTypePointer: class, type
  }
  return total;
}

} // namespace

// operator_dynamic_resize, plus the kernel each step dispatches on this
// device. A step's "expected_kernels" maps a family name (CoopmatFamily.h) to
// the prefix of the kernel it must dispatch; a family without an entry, and
// every device that matches no family, must dispatch no family kernel. A
// step's "family_validation_method" maps a family to the validation method of
// its kernel there (e.g. a tighter atol than the existing kernels of other
// devices need), and validates the output there even if the step sets
// "validate_output" to false (e.g. for a shape the existing kernels get
// wrong); without an entry the case's method applies.
Json check_wmma_family_route(const Json& spec, ComputeGraph& graph) {
  const RouteExpectation expectation = route_expectation();
  const std::string family =
      coopmat_family_name(coopmat_family(coopmat_device_key(&graph)));
  reference::require(
      expectation.family.empty() || family == expectation.family,
      "Device matches family " + family +
          "; ETVK_WMMA_EXPECTED_FAMILY expects " + expectation.family);
  Json routed = spec;
  for (Json& step : routed.at("validation_steps")) {
    if (step.contains("family_validation_method") &&
        step.at("family_validation_method").contains(family)) {
      step["validation_method"] =
          step.at("family_validation_method").at(family);
      step["validate_output"] = true;
    }
  }
  bool asserted = false;
  const auto check_step = [&](const Json& step, ComputeGraph& step_graph) {
    const std::vector<std::string> kernels = dispatched_kernels(step_graph);
    Json expected = nullptr;
    if (step.contains("expected_kernels") &&
        step.at("expected_kernels").contains(family)) {
      expected = step.at("expected_kernels").at(family);
    }
    if (kernels.empty()) {
      reference::require(
          expectation.records_optional,
          "No shader profiling records to check the dispatched kernels; on a "
          "device without timestamp queries set "
          "ETVK_WMMA_EXPECTED_FAMILY=software");
      return Json{
          {"kernels", kernels},
          {"expected", expected},
          {"validation_method",
           step.value("validation_method", Json(nullptr))}};
    }
    asserted = true;
    bool found = false;
    for (const std::string& kernel : kernels) {
      if (expected.is_string()) {
        found |= kernel.rfind(expected.get<std::string>(), 0) == 0;
      } else {
        reference::require(
            !is_family_kernel(kernel),
            "Family kernel " + kernel + " dispatched on family " + family +
                ", which expects none for this step");
      }
    }
    reference::require(
        !expected.is_string() || found,
        "Family " + family + " did not dispatch " + expected.dump() +
            "; dispatched: " + Json(kernels).dump());
    return Json{
        {"kernels", kernels},
        {"expected", expected},
        {"validation_method", step.value("validation_method", Json(nullptr))}};
  };
  Json result;
  try {
    result = validate_operator_dynamic_resize(routed, graph, check_step);
  } catch (const vkapi::ShaderNotSupportedError& error) {
    if (!expectation.strict) {
      throw;
    }
    throw std::runtime_error(
        std::string("Required case did not run: ") + error.what());
  } catch (const UnsupportedRoute& error) {
    if (!expectation.strict) {
      throw;
    }
    throw std::runtime_error(
        std::string("Required case did not run: ") + error.what());
  }
  result["family"] = family;
  result["expected_family"] =
      expectation.family.empty() ? Json(nullptr) : Json(expectation.family);
  result["route_asserted"] = asserted;
  return result;
}

// The shared memory each family row declares for selection
// (WmmaRow::shared_bytes) equals the Workgroup variables of every variant of
// the row in the shader registry. Needs no GPU work.
Json check_wmma_family_shared_memory(const Json&, ComputeGraph&) {
  Json variants = Json::array();
  for (const CoopmatFamily family :
       {CoopmatFamily::kXe2, CoopmatFamily::kNvidia}) {
    for (const WmmaRow& row : coopmat_family_rows(family)) {
      const std::pair<uint8_t, const char*> storages[] = {
          {kWmmaTex3dTex2d, "_texture3d_texture2d_half"},
          {kWmmaBufTex2d, "_buffer_texture2d_half"}};
      for (const auto& [bit, suffix] : storages) {
        if ((row.storages & bit) == 0) {
          continue;
        }
        const uint32_t declared = bit == kWmmaTex3dTex2d
            ? row.shared_bytes.texture3d
            : row.shared_bytes.buffer;
        const std::vector<std::string> zp_modes = row.op == WmmaOp::kQ4gsw
            ? std::vector<std::string>{""}
            : std::vector<std::string>{"_zpint8", "_zpinherit"};
        for (const std::string& zp : zp_modes) {
          const std::string name = row.kernel_base + std::string(suffix) + zp;
          reference::require(
              api::shader_registry().has_shader(name),
              "No shader variant " + name);
          const uint64_t reflected = workgroup_bytes(VK_KERNEL_FROM_STR(name));
          reference::require(
              reflected == declared,
              name + " declares " + std::to_string(reflected) +
                  " bytes of shared memory; its family row says " +
                  std::to_string(declared));
          variants.push_back({{"kernel", name}, {"shared_bytes", reflected}});
        }
      }
    }
  }
  return {{"status", "PASS"}, {"variants", variants}};
}

// Family kernels pass the capability gate on a family device: the Vulkan
// memory model is enabled there and accepted for every family variant. On any
// device, the gate's verdict on the upstream shaders that declare
// VulkanMemoryModel (parameters.memory_model_shaders) is reported, not
// asserted; kernel choice is left to the dispatch gates.
Json check_wmma_memory_model_gate(const Json& spec, ComputeGraph& graph) {
  const RouteExpectation expectation = route_expectation();
  const CoopmatFamily family = coopmat_family(coopmat_device_key(&graph));
  const std::string family_name = coopmat_family_name(family);
  reference::require(
      expectation.family.empty() || family_name == expectation.family,
      "Device matches family " + family_name +
          "; ETVK_WMMA_EXPECTED_FAMILY expects " + expectation.family);
  api::Context& context = *graph.context();
  vkapi::Adapter& adapter = *context.adapter_ptr();
  const uint32_t memory_model =
      static_cast<uint32_t>(vkapi::SpirvCapability::VULKAN_MEMORY_MODEL);
  const bool enabled = adapter.supports_vulkan_memory_model();
  const bool accepted = adapter.supports_spirv_capability(memory_model);
  const auto passes_gate = [&](const std::string& name) {
    const vkapi::ShaderInfo& shader = VK_KERNEL_FROM_STR(name);
    const auto& capabilities = shader.required_capabilities;
    reference::require(
        std::find(capabilities.begin(), capabilities.end(), memory_model) !=
            capabilities.end(),
        name + " does not declare VulkanMemoryModel");
    try {
      context.check_device_capabilities(shader);
      return true;
    } catch (const vkapi::ShaderNotSupportedError&) {
      return false;
    }
  };
  Json upstream = Json::object();
  for (const std::string& name :
       spec.at("parameters").at("memory_model_shaders")) {
    upstream[name] = passes_gate(name) ? "accepted" : "rejected";
  }
  Json family_kernels = Json::array();
  if (family != CoopmatFamily::kNone) {
    reference::require(
        enabled && accepted,
        "Vulkan memory model not enabled on family " + family_name);
    for (const WmmaRow& row : coopmat_family_rows(family)) {
      for (const char* io :
           {"_texture3d_texture2d_half", "_buffer_texture2d_half"}) {
        if ((row.storages & (io[1] == 't' ? kWmmaTex3dTex2d : kWmmaBufTex2d)) ==
            0) {
          continue;
        }
        for (const char* zp : row.op == WmmaOp::kQ4gsw
                 ? std::vector<const char*>{""}
                 : std::vector<const char*>{"_zpint8", "_zpinherit"}) {
          const std::string name = row.kernel_base + std::string(io) + zp;
          reference::require(
              passes_gate(name), name + " rejected on family " + family_name);
          family_kernels.push_back(name);
        }
      }
    }
  }
  return {
      {"status", "PASS"},
      {"family", family_name},
      {"memory_model_enabled", enabled},
      {"memory_model_capability_accepted", accepted},
      {"upstream_memory_model_shaders", upstream},
      {"family_kernels_accepted", family_kernels}};
}

REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(wmma_family_route, check_wmma_family_route);
  REGISTER_CASE_CHECK(
      wmma_family_shared_memory, check_wmma_family_shared_memory);
  REGISTER_CASE_CHECK(wmma_memory_model_gate, check_wmma_memory_model_gate);
}

} // namespace vkcompute::shader_profiling
