/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-graph tests for the linear_q4gsw M==3 shared-bicol route: structural
// route selection, the generic fallback, raw-fp32 scale fidelity (no bf16
// rounding), resize re-entry across the M==3 boundary, and fail-closed
// validation. Inputs are generated in-process, so no fixture directory.

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_m3_shared_bicol_wgsl.h>

#include <gtest/gtest.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

// QuantizedLinear.cpp:60-62 -- the M=3 route's workgroup size and the shared
// storage it needs (6 f32 partials per lane).
constexpr uint32_t kM3Invocations = 64u;
constexpr uint32_t kM3StorageBytes = 6u * kM3Invocations * sizeof(float);
constexpr uint32_t kM3PartialArrays = 6u;

// WebGPU spec minimums (https://www.w3.org/TR/webgpu/#limits); every
// conformant device clears the M=3 gate, so a failure here is a real defect.
constexpr uint32_t kSpecMinInvocations = 256u;
constexpr uint32_t kSpecMinWorkgroupSizeX = 256u;
constexpr uint32_t kSpecMinWorkgroupStorage = 16384u;

constexpr const char* kQ4gswOp = "et_vk.linear_q4gsw.default";
constexpr const char* kM3Shader = "q4gsw_linear_m3_shared_bicol";
constexpr const char* kM3Kernel = "linear_q4gsw_m3_shared_bicol";
constexpr const char* kBicolKernel = "linear_q4gsw_coop4_bicol";
constexpr const char* kTiledKernel = "linear_q4gsw_tiled";

// K % 8 == 0 and group_size % 8 == 0 make the shape bicol/M3 eligible
// (QuantizedLinear.cpp:492-495); K % 16 != 0 keeps every non-M3 case on the
// fp32 tiled kernel (steel_workgroup_count, QuantizedLinear.cpp:133-143) and
// K/N stay under the shmem thresholds (QuantizedLinear.cpp:52-53).
constexpr int64_t kK = 72;
constexpr int64_t kGroupSize = 8;
constexpr int64_t kGroups = kK / kGroupSize;
constexpr int64_t kKPacked = kK / 2;
constexpr int64_t kN = 6;
constexpr int64_t kMaxM = 4;

// Project tolerance for cross-kernel fp32 comparisons; the routes reduce K in
// different orders, so bit-exactness is not a contract between them.
constexpr float kAtol = 1e-3f;
constexpr float kRtol = 1e-3f;
// Scale-fidelity gate: >=19x tighter than the >=1.95e-3 relative shift any
// bf16 rounding of the fixture's scales would cause.
constexpr double kScaleGate = 1e-4;
constexpr double kBf16Separation = 1e-3;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
WGPUDevice g_device = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Mirrors m3_shared_supported (QuantizedLinear.cpp:103-109).
bool m3_shared_supported_here(WGPUDevice device) {
  WGPULimits limits = {};
  return wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup >= kM3Invocations &&
      limits.maxComputeWorkgroupSizeX >= kM3Invocations &&
      limits.maxComputeWorkgroupStorageSize >= kM3StorageBytes;
}

struct Q4gswSpec {
  std::vector<uint32_t> input_dims = {3u, static_cast<uint32_t>(kK)};
  uint32_t n = static_cast<uint32_t>(kN);
  uint32_t k_packed = static_cast<uint32_t>(kKPacked);
  uint32_t groups = static_cast<uint32_t>(kGroups);
  uint32_t padded_n = static_cast<uint32_t>(kN);
  int64_t group_size = kGroupSize;
  uint32_t bias_elems = 0u; // 0 -> Null bias arg
  bool dynamic = false; // emit sym_size.int so the input carries dynamic dims
  bool fp16_scales = false;
  bool rank1_scales = false;
  bool null_group_size = false;
  int id_shift = 0; // unused leading values shift every value id
  int mem_obj_base = 0;
};

struct Q4gswIds {
  int input = 0;
  int weight = 0;
  int scales = 0;
  int bias = 0;
  int output = 0;
};

Q4gswIds build_q4gsw_graph(WebGPUGraph& graph, const Q4gswSpec& spec) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  auto add_tensor = [&](vk::VkDataType dtype,
                        const std::vector<uint32_t>& dims,
                        int mem_obj_id) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb, dtype, &dims, /*constant_id=*/-1, mem_obj_id)
            .Union()));
    return id;
  };
  auto add_int = [&](int64_t value) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, value).Union()));
    return id;
  };
  auto add_null = [&]() {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(fbb));
    return id;
  };

  for (int i = 0; i < spec.id_shift; i++) {
    add_int(i);
  }

  std::vector<uint32_t> output_dims = spec.input_dims;
  output_dims.back() = spec.n;

  Q4gswIds ids;
  ids.input =
      add_tensor(vk::VkDataType::FLOAT32, spec.input_dims, spec.mem_obj_base);
  ids.weight = add_tensor(
      vk::VkDataType::UINT8, {spec.n, spec.k_packed}, spec.mem_obj_base + 1);
  const vk::VkDataType scales_dtype =
      spec.fp16_scales ? vk::VkDataType::FLOAT16 : vk::VkDataType::FLOAT32;
  ids.scales = spec.rank1_scales
      ? add_tensor(scales_dtype, {spec.groups}, spec.mem_obj_base + 2)
      : add_tensor(
            scales_dtype, {spec.groups, spec.padded_n}, spec.mem_obj_base + 2);
  const int group_size =
      spec.null_group_size ? add_null() : add_int(spec.group_size);
  ids.bias = spec.bias_elems == 0u
      ? add_null()
      : add_tensor(
            vk::VkDataType::FLOAT32, {spec.bias_elems}, spec.mem_obj_base + 4);
  ids.output =
      add_tensor(vk::VkDataType::FLOAT32, output_dims, spec.mem_obj_base + 3);

  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  if (spec.dynamic) {
    const int dim = add_int(0);
    const int symint = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::SymInt, vk::CreateSymInt(fbb, 0).Union()));
    const std::vector<int32_t> sym_args = {ids.input, dim, symint};
    chain.push_back(
        vk::CreateOperatorCallDirect(fbb, 0, "sym_size.int", &sym_args));
  }
  const std::vector<int32_t> q4_args = {
      ids.input, ids.weight, ids.scales, group_size, ids.bias, ids.output};
  chain.push_back(vk::CreateOperatorCallDirect(
      fbb, static_cast<uint32_t>(chain.size()), kQ4gswOp, &q4_args));

  std::vector<uint32_t> input_ids = {
      static_cast<uint32_t>(ids.input),
      static_cast<uint32_t>(ids.weight),
      static_cast<uint32_t>(ids.scales)};
  if (spec.bias_elems != 0u) {
    input_ids.push_back(static_cast<uint32_t>(ids.bias));
  }
  const std::vector<uint32_t> output_ids = {static_cast<uint32_t>(ids.output)};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  graph.set_device(g_device);
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  return ids;
}

struct HostFixture {
  std::vector<float> input; // rows * K, row-major
  std::vector<uint8_t> weight; // N * K_packed
  std::vector<float> scales; // groups * padded_N
  std::vector<float> bias; // N, empty when the graph has no bias
};

uint32_t next_u32(uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

float unit_float(uint32_t& state) {
  return static_cast<float>(next_u32(state) >> 8u) / 8388608.0f - 1.0f;
}

// A value exactly halfway between two bf16 neighbours (mantissa low bits
// 0x8000), so ANY bf16 rounding of it moves the value >= 2^-9 relative.
float bf16_midpoint(uint32_t index, int exponent) {
  const uint32_t bits = (static_cast<uint32_t>(127 + exponent) << 23u) |
      ((index & 0x7Fu) << 16u) | 0x8000u;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float truncate_to_bf16(float value) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &value, sizeof(bits));
  bits &= 0xFFFF0000u;
  float truncated = 0.0f;
  std::memcpy(&truncated, &bits, sizeof(truncated));
  return truncated;
}

HostFixture make_random_fixture(int64_t rows, bool with_bias) {
  HostFixture fixture;
  uint32_t state = 0x5eed1234u;
  fixture.input.resize(static_cast<size_t>(rows * kK));
  for (float& value : fixture.input) {
    value = unit_float(state);
  }
  fixture.weight.resize(static_cast<size_t>(kN * kKPacked));
  for (uint8_t& byte : fixture.weight) {
    byte = static_cast<uint8_t>(next_u32(state) >> 24u);
  }
  fixture.scales.resize(static_cast<size_t>(kGroups * kN));
  for (size_t i = 0; i < fixture.scales.size(); i++) {
    fixture.scales[i] =
        bf16_midpoint(static_cast<uint32_t>(11u * i + 3u), /*exponent=*/-6);
  }
  if (with_bias) {
    fixture.bias.resize(static_cast<size_t>(kN));
    for (float& value : fixture.bias) {
      value = unit_float(state);
    }
  }
  return fixture;
}

// All nibbles are 0xF (dequant +7, q4gsw_linear.wgsl:69) and row r is the
// constant r+1, so out[r][c] == 7 * (r+1) * group_size * sum_g scales[g][c].
HostFixture make_analytic_fixture(int64_t rows) {
  HostFixture fixture;
  fixture.input.resize(static_cast<size_t>(rows * kK));
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t k = 0; k < kK; k++) {
      fixture.input[static_cast<size_t>(r * kK + k)] =
          static_cast<float>(r + 1);
    }
  }
  fixture.weight.assign(static_cast<size_t>(kN * kKPacked), 0xFFu);
  fixture.scales.resize(static_cast<size_t>(kGroups * kN));
  for (size_t i = 0; i < fixture.scales.size(); i++) {
    fixture.scales[i] =
        bf16_midpoint(static_cast<uint32_t>(17u * i + 5u), /*exponent=*/0);
  }
  return fixture;
}

double
analytic_expected(const std::vector<float>& scales, int64_t row, int64_t col) {
  double scale_sum = 0.0;
  for (int64_t g = 0; g < kGroups; g++) {
    scale_sum += scales[static_cast<size_t>(g * kN + col)];
  }
  return 7.0 * static_cast<double>(row + 1) * static_cast<double>(kGroupSize) *
      scale_sum;
}

std::vector<float>
run_graph(WebGPUGraph& graph, const HostFixture& fixture, int64_t m) {
  std::vector<InputData> inputs;
  inputs.push_back(
      {fixture.input.data(),
       static_cast<size_t>(m * kK) * sizeof(float),
       false,
       true});
  inputs.push_back(
      {fixture.weight.data(), fixture.weight.size(), false, false});
  inputs.push_back(
      {fixture.scales.data(),
       fixture.scales.size() * sizeof(float),
       false,
       true});
  if (!fixture.bias.empty()) {
    inputs.push_back(
        {fixture.bias.data(),
         fixture.bias.size() * sizeof(float),
         false,
         true});
  }
  std::vector<float> out(static_cast<size_t>(m * kN), 0.0f);
  std::vector<OutputData> outputs(1);
  outputs[0] = {out.data(), out.size() * sizeof(float), /*host_is_fp32=*/true};

  graph.copy_inputs(inputs);
  const WebGPUExecutionPlan plan = graph.make_execution_plan({});
  graph.execute(plan);
  graph.copy_outputs(outputs, plan);
  return out;
}

struct ActiveDispatch {
  std::string kernel_name;
  uint32_t workgroup_count_x = 0u;
  size_t count = 0u;
};

// The dispatches a route group left runnable; a selected route has exactly one.
ActiveDispatch active_dispatch(WebGPUGraph& graph) {
  ActiveDispatch active;
  for (size_t i = 0; i < graph.num_dispatches(); i++) {
    const WebGPUDispatch& dispatch = graph.dispatch_at(i);
    if (dispatch.kind != WebGPUDispatch::Kind::Compute ||
        dispatch.workgroup_count_x == 0u) {
      continue;
    }
    active.kernel_name = dispatch.kernel_name;
    active.workgroup_count_x = dispatch.workgroup_count_x;
    active.count++;
  }
  return active;
}

size_t count_kernel(WebGPUGraph& graph, const char* kernel_name) {
  size_t count = 0;
  for (size_t i = 0; i < graph.num_dispatches(); i++) {
    if (graph.dispatch_at(i).kernel_name == kernel_name) {
      count++;
    }
  }
  return count;
}

void expect_close(
    const std::vector<float>& got,
    const std::vector<float>& want,
    size_t count,
    const std::string& label) {
  ASSERT_GE(got.size(), count) << label;
  ASSERT_GE(want.size(), count) << label;
  for (size_t i = 0; i < count; i++) {
    ASSERT_TRUE(std::isfinite(got[i])) << label << " i=" << i;
    const float abs_err = std::fabs(got[i] - want[i]);
    const float rel_err = abs_err / std::fmax(std::fabs(want[i]), 1e-6f);
    EXPECT_TRUE(abs_err <= kAtol || rel_err <= kRtol)
        << label << " i=" << i << " got=" << got[i] << " want=" << want[i];
  }
}

void expect_build_error(const Q4gswSpec& spec, const char* expected) {
  WebGPUGraph graph;
  std::string error;
  try {
    build_q4gsw_graph(graph, spec);
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  EXPECT_EQ(error, expected);
  EXPECT_EQ(graph.memory_stats().num_dispatches, 0);
}

// Strips `//` and `/* */` so a shader's prose cannot satisfy or trip a check.
std::string strip_wgsl_comments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  for (size_t i = 0; i < src.size();) {
    if (src.compare(i, 2, "//") == 0) {
      while (i < src.size() && src[i] != '\n') {
        i++;
      }
    } else if (src.compare(i, 2, "/*") == 0) {
      i += 2;
      while (i + 1 < src.size() && src.compare(i, 2, "*/") != 0) {
        i++;
      }
      i = i + 1 < src.size() ? i + 2 : src.size();
    } else {
      out.push_back(src[i++]);
    }
  }
  return out;
}

// The declared x dim, resolving a `const <ident>: u32 = <n>u;` indirection.
uint32_t declared_workgroup_size_x(const std::string& src) {
  const std::string code = strip_wgsl_comments(src);
  const size_t at = code.find("@workgroup_size");
  if (at == std::string::npos) {
    return 0;
  }
  const size_t open = code.find('(', at);
  const size_t close = code.find(')', open);
  if (open == std::string::npos || close == std::string::npos) {
    return 0;
  }
  std::string token = code.substr(open + 1, close - open - 1);
  token = token.substr(0, token.find(','));
  const size_t begin = token.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) {
    return 0;
  }
  token = token.substr(begin, token.find_last_not_of(" \t\n\r") + 1 - begin);
  if (!token.empty() && std::isdigit(static_cast<unsigned char>(token[0]))) {
    return static_cast<uint32_t>(std::strtoul(token.c_str(), nullptr, 10));
  }
  const size_t decl = code.find("const " + token);
  const size_t eq =
      decl == std::string::npos ? std::string::npos : code.find('=', decl);
  if (eq == std::string::npos) {
    return 0;
  }
  return static_cast<uint32_t>(
      std::strtoul(code.c_str() + eq + 1, nullptr, 10));
}

std::string to_lower(const std::string& text) {
  std::string out = text;
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

size_t count_occurrences(const std::string& haystack, const char* needle) {
  size_t count = 0;
  const std::string pattern(needle);
  for (size_t at = haystack.find(pattern); at != std::string::npos;
       at = haystack.find(pattern, at + pattern.size())) {
    count++;
  }
  return count;
}

} // namespace

// The M=3 route is gated on device limits, not skipped: every WebGPU-conformant
// device clears them, so an unmet gate is a defect, not an unsupported setup.
TEST(Q4gswM3Device, ClearsTheSharedBicolLimitGate) {
  WGPULimits limits = {};
  ASSERT_EQ(wgpuDeviceGetLimits(g_device, &limits), WGPUStatus_Success);
  EXPECT_GE(limits.maxComputeInvocationsPerWorkgroup, kSpecMinInvocations);
  EXPECT_GE(limits.maxComputeWorkgroupSizeX, kSpecMinWorkgroupSizeX);
  EXPECT_GE(limits.maxComputeWorkgroupStorageSize, kSpecMinWorkgroupStorage);
  ASSERT_TRUE(m3_shared_supported_here(g_device))
      << "device is below the M=3 gate (64 invocations, 64 x, "
      << kM3StorageBytes << " shared bytes)";
}

// The op compiles the M=3 pipeline from kQ4gswLinearM3SharedBicolWGSL directly
// (QuantizedLinear.cpp:635, :694), so pin the registry to that same source and
// assert the source itself carries no scale truncation.
TEST(Q4gswM3Shader, ServesRawFp32ScalesWithNoBf16Rounding) {
  const WebGPUShaderInfo& info = get_webgpu_shader_info(kM3Shader);
  ASSERT_NE(info.source, nullptr);
  EXPECT_STREQ(info.source, kQ4gswLinearM3SharedBicolWGSL);
  EXPECT_EQ(declared_workgroup_size_x(info.source), kM3Invocations);
  EXPECT_EQ(info.workgroup_size_x, kM3Invocations)
      << "registry workgroup size (kQ4gswLinearM3SharedBicolWorkgroupSizeX) "
         "disagrees with @workgroup_size in the shader source";

  const std::string src = strip_wgsl_comments(info.source);
  const std::string lowered = to_lower(src);
  EXPECT_NE(
      src.find("var<storage, read> t_scales: array<f32>"), std::string::npos)
      << "t_scales must stay a raw fp32 storage array";
  for (const char* needle : {"f16", "bfloat", "0xffff0000", "bitcast"}) {
    EXPECT_EQ(lowered.find(needle), std::string::npos)
        << "scale-truncation pattern '" << needle << "' reappeared";
  }
  // A 7th partial array would need more shared memory than the CPU gate asks
  // the device for (kQ4gswM3StorageBytes == 6 * 64 * 4).
  EXPECT_EQ(count_occurrences(src, "var<workgroup>"), kM3PartialArrays);
  EXPECT_EQ(count_occurrences(src, "array<f32, WG>"), kM3PartialArrays);
}

TEST(Q4gswM3Route, SelectsSharedBicolOnlyAtMEqualsThree) {
  const struct {
    int64_t m;
    const char* kernel;
  } cases[] = {
      {1, kBicolKernel},
      {2, kTiledKernel},
      {3, kM3Kernel},
      {4, kTiledKernel},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(std::string("M=") + std::to_string(test_case.m));
    Q4gswSpec spec;
    spec.input_dims = {
        static_cast<uint32_t>(test_case.m), static_cast<uint32_t>(kK)};
    WebGPUGraph graph;
    ASSERT_NO_THROW(build_q4gsw_graph(graph, spec));
    ASSERT_EQ(graph.num_dispatches(), 1u);
    EXPECT_EQ(graph.dispatch_at(0).kernel_name, test_case.kernel);
    EXPECT_EQ(count_kernel(graph, kM3Kernel), test_case.m == 3 ? 1u : 0u);
    EXPECT_NE(graph.dispatch_at(0).pipeline, nullptr);
    EXPECT_NE(graph.dispatch_at(0).bind_group, nullptr);
    EXPECT_GT(graph.dispatch_at(0).workgroup_count_x, 0u);
  }
}

// M=3 dispatches ceil(N/2) column-pair workgroups (QuantizedLinear.cpp:189).
TEST(Q4gswM3Route, DispatchesOneWorkgroupPerColumnPair) {
  for (uint32_t n : {6u, 7u}) {
    SCOPED_TRACE(std::string("N=") + std::to_string(n));
    Q4gswSpec spec;
    spec.n = n;
    spec.padded_n = n;
    WebGPUGraph graph;
    ASSERT_NO_THROW(build_q4gsw_graph(graph, spec));
    ASSERT_EQ(graph.num_dispatches(), 1u);
    EXPECT_EQ(graph.dispatch_at(0).kernel_name, kM3Kernel);
    EXPECT_EQ(graph.dispatch_at(0).workgroup_count_x, (n + 1u) / 2u);
    EXPECT_EQ(graph.dispatch_at(0).workgroup_count_y, 1u);
  }
}

// M comes from numel/K and the device limits alone -- not from a value id, a
// dims pattern, a memory-object id, or any artifact-bound role map.
TEST(Q4gswM3Route, SelectionIsStructuralNotArtifactBound) {
  const struct {
    const char* name;
    Q4gswSpec spec;
  } cases[] = {
      {"shifted value ids",
       [] {
         Q4gswSpec s;
         s.id_shift = 9;
         return s;
       }()},
      {"shifted memory-object ids",
       [] {
         Q4gswSpec s;
         s.mem_obj_base = 7;
         return s;
       }()},
      {"rank-3 input [1,3,K]",
       [] {
         Q4gswSpec s;
         s.input_dims = {1u, 3u, static_cast<uint32_t>(kK)};
         return s;
       }()},
      {"wider N",
       [] {
         Q4gswSpec s;
         s.n = 10u;
         s.padded_n = 10u;
         return s;
       }()},
      {"padded scales table",
       [] {
         Q4gswSpec s;
         s.padded_n = 16u;
         return s;
       }()},
      {"biased",
       [] {
         Q4gswSpec s;
         s.bias_elems = static_cast<uint32_t>(kN);
         return s;
       }()},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    WebGPUGraph graph;
    ASSERT_NO_THROW(build_q4gsw_graph(graph, test_case.spec));
    ASSERT_EQ(graph.num_dispatches(), 1u);
    EXPECT_EQ(graph.dispatch_at(0).kernel_name, kM3Kernel);
    EXPECT_EQ(count_kernel(graph, kM3Kernel), 1u);
  }
}

// M3 ineligible (K % 8 or group_size % 8) must still build and dispatch a
// named q4gsw kernel -- never zero dispatches, never a throw.
TEST(Q4gswM3Route, FallsBackToANamedKernelWhenIneligible) {
  const struct {
    const char* name;
    Q4gswSpec spec;
  } cases[] = {
      {"K % 8 != 0",
       [] {
         Q4gswSpec s;
         s.input_dims = {3u, 20u};
         s.k_packed = 10u;
         s.groups = 3u;
         return s;
       }()},
      {"group_size % 8 != 0",
       [] {
         Q4gswSpec s;
         s.input_dims = {3u, 24u};
         s.k_packed = 12u;
         s.groups = 6u;
         s.group_size = 4;
         return s;
       }()},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    WebGPUGraph graph;
    ASSERT_NO_THROW(build_q4gsw_graph(graph, test_case.spec));
    ASSERT_EQ(graph.num_dispatches(), 1u);
    EXPECT_EQ(graph.dispatch_at(0).kernel_name, kTiledKernel);
    EXPECT_EQ(count_kernel(graph, kM3Kernel), 0u);
    EXPECT_NE(graph.dispatch_at(0).pipeline, nullptr);
    EXPECT_GT(graph.dispatch_at(0).workgroup_count_x, 0u);
  }
}

// The M=4 graph runs the generic tiled kernel over the same weights, scales and
// first three input rows, so its rows 0-2 are a same-math oracle for M=3.
TEST(Q4gswM3Numerics, MatchesTheGenericRouteOnTheSameInputs) {
  for (bool with_bias : {false, true}) {
    SCOPED_TRACE(with_bias ? "biased" : "unbiased");
    const HostFixture fixture = make_random_fixture(kMaxM, with_bias);

    Q4gswSpec generic_spec;
    generic_spec.input_dims = {
        static_cast<uint32_t>(kMaxM), static_cast<uint32_t>(kK)};
    generic_spec.bias_elems = with_bias ? static_cast<uint32_t>(kN) : 0u;
    WebGPUGraph generic;
    ASSERT_NO_THROW(build_q4gsw_graph(generic, generic_spec));
    ASSERT_EQ(generic.num_dispatches(), 1u);
    ASSERT_EQ(generic.dispatch_at(0).kernel_name, kTiledKernel);
    const std::vector<float> want = run_graph(generic, fixture, kMaxM);

    Q4gswSpec m3_spec;
    m3_spec.bias_elems = generic_spec.bias_elems;
    WebGPUGraph m3;
    ASSERT_NO_THROW(build_q4gsw_graph(m3, m3_spec));
    ASSERT_EQ(m3.num_dispatches(), 1u);
    ASSERT_EQ(m3.dispatch_at(0).kernel_name, kM3Kernel);
    const std::vector<float> got = run_graph(m3, fixture, 3);

    expect_close(got, want, static_cast<size_t>(3 * kN), "m3 vs generic");
  }
}

// Kills a reintroduced bf16 scale rounding numerically: the fixture's scales
// sit at bf16 midpoints, so rounding them (either direction) shifts every
// output by >= 1.95e-3 relative, 19x the gate this asserts.
TEST(Q4gswM3Numerics, AppliesRawFp32ScalesNotBf16) {
  const HostFixture fixture = make_analytic_fixture(kMaxM);
  for (float scale : fixture.scales) {
    const double relative_shift =
        std::fabs(scale - truncate_to_bf16(scale)) / scale;
    ASSERT_GT(relative_shift, 1.5e-3)
        << "fixture scale " << scale << " is not bf16-sensitive";
  }

  Q4gswSpec m3_spec;
  WebGPUGraph m3;
  ASSERT_NO_THROW(build_q4gsw_graph(m3, m3_spec));
  ASSERT_EQ(m3.num_dispatches(), 1u);
  ASSERT_EQ(m3.dispatch_at(0).kernel_name, kM3Kernel);
  const std::vector<float> got = run_graph(m3, fixture, 3);

  std::vector<float> bf16_scales = fixture.scales;
  for (float& scale : bf16_scales) {
    scale = truncate_to_bf16(scale);
  }
  for (int64_t r = 0; r < 3; r++) {
    for (int64_t c = 0; c < kN; c++) {
      const double observed = got[static_cast<size_t>(r * kN + c)];
      const double expected = analytic_expected(fixture.scales, r, c);
      const double rounded = analytic_expected(bf16_scales, r, c);
      ASSERT_GT(expected, 0.0);
      EXPECT_LT(std::fabs(observed - expected) / expected, kScaleGate)
          << "raw-fp32 scale mismatch at [" << r << "," << c << "]";
      EXPECT_GT(std::fabs(observed - rounded) / expected, kBf16Separation)
          << "output matches the bf16-rounded counterfactual at [" << r << ","
          << c << "]";
    }
  }
}

// The generic route is held to the same raw-fp32 scale contract, so a bf16
// mutant applied to BOTH routes cannot hide behind route parity.
TEST(Q4gswM3Numerics, GenericRouteAlsoAppliesRawFp32Scales) {
  const HostFixture fixture = make_analytic_fixture(kMaxM);
  Q4gswSpec spec;
  spec.input_dims = {static_cast<uint32_t>(kMaxM), static_cast<uint32_t>(kK)};
  WebGPUGraph graph;
  ASSERT_NO_THROW(build_q4gsw_graph(graph, spec));
  ASSERT_EQ(graph.num_dispatches(), 1u);
  ASSERT_EQ(graph.dispatch_at(0).kernel_name, kTiledKernel);
  const std::vector<float> got = run_graph(graph, fixture, kMaxM);

  for (int64_t r = 0; r < kMaxM; r++) {
    for (int64_t c = 0; c < kN; c++) {
      const double observed = got[static_cast<size_t>(r * kN + c)];
      const double expected = analytic_expected(fixture.scales, r, c);
      EXPECT_LT(std::fabs(observed - expected) / expected, kScaleGate)
          << "generic raw-fp32 scale mismatch at [" << r << "," << c << "]";
    }
  }
}

// One dynamic graph resized across the M==3 boundary in both directions: the
// recorded route must be reselected and stay numerically correct each time.
TEST(Q4gswM3Resize, ReselectsAcrossTheM3Boundary) {
  const HostFixture fixture = make_random_fixture(kMaxM, /*with_bias=*/false);

  Q4gswSpec generic_spec;
  generic_spec.input_dims = {
      static_cast<uint32_t>(kMaxM), static_cast<uint32_t>(kK)};
  WebGPUGraph generic;
  ASSERT_NO_THROW(build_q4gsw_graph(generic, generic_spec));
  ASSERT_EQ(generic.num_dispatches(), 1u);
  ASSERT_EQ(generic.dispatch_at(0).kernel_name, kTiledKernel);
  const std::vector<float> want = run_graph(generic, fixture, kMaxM);

  Q4gswSpec spec;
  spec.input_dims = {static_cast<uint32_t>(kMaxM), static_cast<uint32_t>(kK)};
  spec.dynamic = true;
  WebGPUGraph graph;
  Q4gswIds ids;
  ASSERT_NO_THROW(ids = build_q4gsw_graph(graph, spec));
  ASSERT_TRUE(graph.has_dynamic_shapes());
  ASSERT_EQ(graph.num_dispatches(), 3u);
  EXPECT_EQ(graph.dispatch_at(0).kernel_name, kBicolKernel);
  EXPECT_EQ(graph.dispatch_at(1).kernel_name, kM3Kernel);
  EXPECT_EQ(graph.dispatch_at(2).kernel_name, kTiledKernel);
  EXPECT_EQ(count_kernel(graph, kM3Kernel), 1u);
  EXPECT_EQ(active_dispatch(graph).kernel_name, kTiledKernel);

  const struct {
    int64_t m;
    const char* kernel;
  } steps[] = {
      {4, kTiledKernel},
      {3, kM3Kernel},
      {1, kBicolKernel},
      {3, kM3Kernel},
      {2, kTiledKernel},
      {4, kTiledKernel},
  };
  for (const auto& step : steps) {
    SCOPED_TRACE(std::string("live M=") + std::to_string(step.m));
    graph.resize_input(ids.input, {step.m, kK});
    ASSERT_NO_THROW(graph.propagate_resize());
    const ActiveDispatch active = active_dispatch(graph);
    ASSERT_EQ(active.count, 1u);
    EXPECT_EQ(active.kernel_name, step.kernel);
    if (step.m == 1 || step.m == 3) {
      EXPECT_EQ(active.workgroup_count_x, static_cast<uint32_t>((kN + 1) / 2));
    }
    const std::vector<float> got = run_graph(graph, fixture, step.m);
    expect_close(
        got, want, static_cast<size_t>(step.m * kN), "resized vs generic");
  }
}

TEST(Q4gswM3FailsClosed, RejectsMalformedGraphsBeforeAnyDispatch) {
  const struct {
    const char* name;
    Q4gswSpec spec;
    const char* error;
  } cases[] = {
      {"rank-1 scales",
       [] {
         Q4gswSpec s;
         s.rank1_scales = true;
         return s;
       }(),
       "WebGPU linear_q4gsw: malformed input dims"},
      {"K_packed != ceil(K/2)",
       [] {
         Q4gswSpec s;
         s.k_packed = static_cast<uint32_t>(kKPacked) + 1u;
         return s;
       }(),
       "WebGPU linear_q4gsw: K_packed must be ceil(K/2)"},
      {"N*K_packed not u32-aligned",
       [] {
         Q4gswSpec s;
         s.input_dims = {3u, 2u};
         s.n = 1u;
         s.k_packed = 1u;
         s.groups = 1u;
         s.padded_n = 1u;
         return s;
       }(),
       "WebGPU linear_q4gsw: N*K_packed must be a multiple of 4 (u32-packed)"},
      {"fp16 scales",
       [] {
         Q4gswSpec s;
         s.fp16_scales = true;
         return s;
       }(),
       "WebGPU linear_q4gsw: fp32-only (byte-size mismatch)"},
      {"group_size == 0",
       [] {
         Q4gswSpec s;
         s.group_size = 0;
         return s;
       }(),
       "WebGPU linear_q4gsw: group_size <= 0"},
      {"group_size not an Int",
       [] {
         Q4gswSpec s;
         s.null_group_size = true;
         return s;
       }(),
       "WebGPU linear_q4gsw: group_size <= 0"},
      {"too few scale groups",
       [] {
         Q4gswSpec s;
         s.groups = static_cast<uint32_t>(kGroups) - 1u;
         return s;
       }(),
       "WebGPU linear_q4gsw: scales dims too small for K/N"},
      {"padded_N < N",
       [] {
         Q4gswSpec s;
         s.padded_n = static_cast<uint32_t>(kN) - 1u;
         return s;
       }(),
       "WebGPU linear_q4gsw: scales dims too small for K/N"},
      {"undersized bias",
       [] {
         Q4gswSpec s;
         s.bias_elems = static_cast<uint32_t>(kN) - 1u;
         return s;
       }(),
       "WebGPU linear_q4gsw: bias present but null/undersized"},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    expect_build_error(test_case.spec, test_case.error);
  }
}

// A live shape whose element count is not a multiple of K must throw instead of
// dispatching a mis-sized M=3 grid, and the graph must stay retryable.
TEST(Q4gswM3FailsClosed, RejectsALiveShapeThatBreaksTheKContract) {
  Q4gswSpec spec;
  spec.input_dims = {static_cast<uint32_t>(kMaxM), static_cast<uint32_t>(kK)};
  spec.dynamic = true;
  WebGPUGraph graph;
  Q4gswIds ids;
  ASSERT_NO_THROW(ids = build_q4gsw_graph(graph, spec));

  graph.resize_input(ids.input, {3, 8});
  std::string error;
  try {
    graph.propagate_resize();
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  EXPECT_EQ(
      error,
      "WebGPU linear_q4gsw(resize): live input numel not a multiple of K");

  graph.resize_input(ids.input, {3, kK});
  ASSERT_NO_THROW(graph.propagate_resize());
  const ActiveDispatch active = active_dispatch(graph);
  EXPECT_EQ(active.count, 1u);
  EXPECT_EQ(active.kernel_name, kM3Kernel);
}

} // namespace executorch::backends::webgpu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  executorch::backends::webgpu::WebGPUContext ctx;
  try {
    ctx = executorch::backends::webgpu::create_webgpu_context();
  } catch (const std::exception& e) {
    if (std::getenv("WEBGPU_REQUIRE_DEVICE") != nullptr) {
      std::printf(
          "FAIL: WEBGPU_REQUIRE_DEVICE set but no device: %s\n", e.what());
      return 1;
    }
    std::printf("SKIP: %s\n", e.what());
    return 0;
  }
  executorch::backends::webgpu::set_default_webgpu_context(&ctx);
  executorch::backends::webgpu::g_device = ctx.device;
  std::printf("WebGPU device acquired (native)\n");

  const int rc = RUN_ALL_TESTS();
  executorch::backends::webgpu::set_default_webgpu_context(nullptr);
  executorch::backends::webgpu::destroy_webgpu_context(ctx);
  return rc;
}
