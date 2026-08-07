/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>

#include <gtest/gtest.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

constexpr int64_t kInputWidth = 2048;
constexpr int64_t kOutputWidth = 32;
constexpr uint32_t kTopkStagedLanes = 64;

// Must equal the corpus written by test/ops/topk/export_topk_artifacts.py.
const char* const kCases[] = {
    "all_equal",
    // all_negative and straddling_zero are the only rows reaching topk.wgsl:62.
    "all_negative",
    "boundary_ties",
    "descending",
    "infinities",
    "interior_ties",
    "nan_payloads",
    "ordinary",
    "random_seeded",
    "signed_zeros",
    "straddling_zero",
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::string g_dir;
WGPUDevice g_device = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

std::vector<uint32_t> read_words(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || static_cast<size_t>(f.tellg()) != count * sizeof(uint32_t)) {
    return {};
  }
  f.seekg(0);
  std::vector<uint32_t> data(count);
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(count * sizeof(uint32_t)));
  return data;
}

// Strips `//` and `/* */` so a shader's prose cannot shadow its real attribute.
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

struct TopkGraphSpec {
  int64_t input_width = kInputWidth;
  int64_t values_width = kOutputWidth;
  int64_t indices_width = kOutputWidth;
  int64_t k = kOutputWidth;
  int64_t dim = -1;
  bool largest = true;
  bool sorted = true;
  bool values_are_int = false;
  bool indices_are_int = true;
  bool output_list_is_int = false;
  bool drop_last_arg = false;
};

void build_topk_graph(WebGPUGraph& graph, const TopkGraphSpec& spec) {
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
  auto add_bool = [&](bool value) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::Bool, vk::CreateBool(fbb, value).Union()));
    return id;
  };

  const int input = add_tensor(
      vk::VkDataType::FLOAT32,
      {1, 1, static_cast<uint32_t>(spec.input_width)},
      0);
  const int k = add_int(spec.k);
  const int dim = add_int(spec.dim);
  const int largest = add_bool(spec.largest);
  const int sorted = add_bool(spec.sorted);
  const int out_values = add_tensor(
      spec.values_are_int ? vk::VkDataType::INT32 : vk::VkDataType::FLOAT32,
      {1, 1, static_cast<uint32_t>(spec.values_width)},
      1);
  const int out_indices = add_tensor(
      spec.indices_are_int ? vk::VkDataType::INT32 : vk::VkDataType::FLOAT32,
      {1, 1, static_cast<uint32_t>(spec.indices_width)},
      2);

  int output_list = 0;
  if (spec.output_list_is_int) {
    output_list = add_int(0);
  } else {
    const std::vector<int32_t> items = {out_values, out_indices};
    output_list = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::ValueList,
        vk::CreateValueListDirect(fbb, &items).Union()));
  }

  std::vector<int32_t> args = {input, k, dim, largest, sorted, output_list};
  if (spec.drop_last_arg) {
    args.pop_back();
  }
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(
      vk::CreateOperatorCallDirect(fbb, 0, "aten.topk.default", &args));

  const std::vector<uint32_t> input_ids = {static_cast<uint32_t>(input)};
  const std::vector<uint32_t> output_ids = {
      static_cast<uint32_t>(out_values), static_cast<uint32_t>(out_indices)};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  graph.set_device(g_device);
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
}

void run_case(const char* name) {
  const std::string base = g_dir + "/" + name;
  const std::vector<uint32_t> scores =
      read_words(base + ".scores.bin", kInputWidth);
  const std::vector<uint32_t> expected_values =
      read_words(base + ".values.bin", kOutputWidth);
  const std::vector<uint32_t> expected_indices =
      read_words(base + ".indices.bin", kOutputWidth);
  ASSERT_FALSE(scores.empty()) << "missing/short scores fixture for " << name;
  ASSERT_FALSE(expected_values.empty()) << "missing values fixture " << name;
  ASSERT_FALSE(expected_indices.empty()) << "missing indices fixture " << name;

  WebGPUGraph graph;
  build_topk_graph(graph, TopkGraphSpec{});
  ASSERT_EQ(graph.num_dispatches(), 1u);
  EXPECT_EQ(graph.dispatch_at(0).kernel_name, "topk_staged_serial");
  EXPECT_EQ(graph.dispatch_at(0).workgroup_count_x, 1u);

  std::vector<InputData> inputs(1);
  inputs[0] = {scores.data(), scores.size() * sizeof(uint32_t), false, true};
  std::vector<uint32_t> got_values(kOutputWidth, 0u);
  std::vector<int32_t> got_indices(kOutputWidth, -1);
  std::vector<OutputData> outputs(2);
  outputs[0] = {
      got_values.data(), got_values.size() * sizeof(uint32_t), /*fp32=*/true};
  outputs[1] = {
      got_indices.data(), got_indices.size() * sizeof(int32_t), /*fp32=*/false};

  graph.copy_inputs(inputs);
  const WebGPUExecutionPlan plan = graph.make_execution_plan({});
  graph.execute(plan);
  graph.copy_outputs(outputs, plan);

  for (int64_t i = 0; i < kOutputWidth; i++) {
    // Bit-exact: NaN payloads and signed zeros must survive unchanged.
    EXPECT_EQ(got_values[i], expected_values[i]) << name << " value " << i;
    EXPECT_EQ(static_cast<uint32_t>(got_indices[i]), expected_indices[i])
        << name << " index " << i;
  }
}

} // namespace

TEST(TopkFixtureContract, CaseListMatchesTheExportedCorpus) {
  std::ifstream manifest(g_dir + "/cases.txt");
  ASSERT_TRUE(manifest.good()) << "missing " << g_dir << "/cases.txt";
  std::vector<std::string> exported;
  std::string line;
  while (std::getline(manifest, line)) {
    if (!line.empty()) {
      exported.push_back(line);
    }
  }
  std::vector<std::string> expected(std::begin(kCases), std::end(kCases));
  EXPECT_EQ(exported, expected);
}

TEST(TopkShader, RegistryWorkgroupSizeMatchesTheShaderDeclaration) {
  const WebGPUShaderInfo& info = get_webgpu_shader_info("topk");
  ASSERT_NE(info.source, nullptr);
  EXPECT_EQ(declared_workgroup_size_x(info.source), kTopkStagedLanes);
  EXPECT_EQ(info.workgroup_size_x, kTopkStagedLanes)
      << "registry workgroup size disagrees with @workgroup_size; the "
         "generated header parsed a commented-out attribute";
}

TEST(TopkExactness, AllCases) {
  for (const char* name : kCases) {
    SCOPED_TRACE(name);
    run_case(name);
  }
}

TEST(TopkFailsClosed, RejectsMalformedShapesAndScalars) {
  const struct {
    const char* name;
    TopkGraphSpec spec;
  } cases[] = {
      {"k != 32",
       [] {
         TopkGraphSpec s;
         s.k = 16;
         return s;
       }()},
      {"dim != -1",
       [] {
         TopkGraphSpec s;
         s.dim = 2;
         return s;
       }()},
      {"largest = false",
       [] {
         TopkGraphSpec s;
         s.largest = false;
         return s;
       }()},
      {"sorted = false",
       [] {
         TopkGraphSpec s;
         s.sorted = false;
         return s;
       }()},
      {"input width",
       [] {
         TopkGraphSpec s;
         s.input_width = 1024;
         return s;
       }()},
      {"values width",
       [] {
         TopkGraphSpec s;
         s.values_width = 64;
         return s;
       }()},
      {"indices width",
       [] {
         TopkGraphSpec s;
         s.indices_width = 31;
         return s;
       }()},
      {"values dtype",
       [] {
         TopkGraphSpec s;
         s.values_are_int = true;
         return s;
       }()},
      {"indices dtype",
       [] {
         TopkGraphSpec s;
         s.indices_are_int = false;
         return s;
       }()},
      {"output list type",
       [] {
         TopkGraphSpec s;
         s.output_list_is_int = true;
         return s;
       }()},
      {"argument count",
       [] {
         TopkGraphSpec s;
         s.drop_last_arg = true;
         return s;
       }()},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    WebGPUGraph graph;
    EXPECT_THROW(build_topk_graph(graph, test_case.spec), std::runtime_error);
  }
}

TEST(TopkFailsClosed, AcceptsTheExactMtpShape) {
  WebGPUGraph graph;
  EXPECT_NO_THROW(build_topk_graph(graph, TopkGraphSpec{}));
}

} // namespace executorch::backends::webgpu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  executorch::backends::webgpu::g_dir = "/tmp/topk";
  if (argc > 1) {
    executorch::backends::webgpu::g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_TOPK_DIR")) {
    executorch::backends::webgpu::g_dir = env;
  }

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
