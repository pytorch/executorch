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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

constexpr int64_t kVocabSize = 262144;
constexpr int64_t kSelectedCount = 4096;
constexpr uint32_t kUniqueLanes = 64;

constexpr const char* kGenericOp = "aten.scatter.src";
constexpr const char* kUniqueOp = "et_vk.scatter_src_unique.default";

struct FixtureCase {
  std::string name;
  bool parallel_equivalent = false;
  bool official_provenance = false;
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::string g_dir;
WGPUDevice g_device = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || static_cast<size_t>(f.tellg()) != count * sizeof(T)) {
    return {};
  }
  f.seekg(0);
  std::vector<T> data(count);
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(count * sizeof(T)));
  return data;
}

std::vector<FixtureCase> read_cases() {
  std::vector<FixtureCase> cases;
  std::ifstream manifest(g_dir + "/cases.txt");
  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream parts(line);
    FixtureCase entry;
    int equivalent = 0;
    int provenance = 0;
    parts >> entry.name >> equivalent >> provenance;
    entry.parallel_equivalent = equivalent != 0;
    entry.official_provenance = provenance != 0;
    cases.push_back(entry);
  }
  return cases;
}

struct ScatterGraphSpec {
  const char* op = kGenericOp;
  int64_t vocab = kVocabSize;
  int64_t selected = kSelectedCount;
  int64_t output_vocab = kVocabSize;
  int64_t dim = -1;
  bool index_is_int = true;
  bool source_is_int = false;
  bool input_is_int = false;
  bool drop_last_arg = false;
};

void build_scatter_graph(WebGPUGraph& graph, const ScatterGraphSpec& spec) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  auto add_tensor = [&](vk::VkDataType dtype, int64_t width, int mem_obj_id) {
    const int id = static_cast<int>(values.size());
    const std::vector<uint32_t> dims = {1, 1, static_cast<uint32_t>(width)};
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb, dtype, &dims, /*constant_id=*/-1, mem_obj_id)
            .Union()));
    return id;
  };

  const int input = add_tensor(
      spec.input_is_int ? vk::VkDataType::INT32 : vk::VkDataType::FLOAT32,
      spec.vocab,
      0);
  const int dim = static_cast<int>(values.size());
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, spec.dim).Union()));
  const int index = add_tensor(
      spec.index_is_int ? vk::VkDataType::INT32 : vk::VkDataType::FLOAT32,
      spec.selected,
      1);
  const int source = add_tensor(
      spec.source_is_int ? vk::VkDataType::INT32 : vk::VkDataType::FLOAT32,
      spec.selected,
      2);
  const int output = add_tensor(vk::VkDataType::FLOAT32, spec.output_vocab, 3);

  std::vector<int32_t> args = {input, dim, index, source, output};
  if (spec.drop_last_arg) {
    args.pop_back();
  }
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(vk::CreateOperatorCallDirect(fbb, 0, spec.op, &args));

  const std::vector<uint32_t> input_ids = {
      static_cast<uint32_t>(input),
      static_cast<uint32_t>(index),
      static_cast<uint32_t>(source)};
  const std::vector<uint32_t> output_ids = {static_cast<uint32_t>(output)};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  graph.set_device(g_device);
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
}

// The compute dispatch, skipping the flat input->output copy the handler emits.
const WebGPUDispatch& compute_dispatch(WebGPUGraph& graph) {
  for (size_t i = 0; i < graph.num_dispatches(); i++) {
    if (graph.dispatch_at(i).kind == WebGPUDispatch::Kind::Compute) {
      return graph.dispatch_at(i);
    }
  }
  throw std::runtime_error("scatter graph has no compute dispatch");
}

std::vector<float> run_route(
    const char* op,
    const std::vector<float>& base,
    const std::vector<int32_t>& indices,
    const std::vector<float>& source) {
  ScatterGraphSpec spec;
  spec.op = op;
  WebGPUGraph graph;
  build_scatter_graph(graph, spec);

  std::vector<InputData> inputs(3);
  inputs[0] = {base.data(), base.size() * sizeof(float), false, true};
  inputs[1] = {indices.data(), indices.size() * sizeof(int32_t), false, false};
  inputs[2] = {source.data(), source.size() * sizeof(float), false, true};
  std::vector<float> out(static_cast<size_t>(kVocabSize), 0.0f);
  std::vector<OutputData> outputs(1);
  outputs[0] = {out.data(), out.size() * sizeof(float), /*fp32=*/true};

  graph.copy_inputs(inputs);
  const WebGPUExecutionPlan plan = graph.make_execution_plan({});
  graph.execute(plan);
  graph.copy_outputs(outputs, plan);
  return out;
}

} // namespace

TEST(ScatterFixtureContract, ExportedCorpusIsPresentAndLabelled) {
  const std::vector<FixtureCase> cases = read_cases();
  ASSERT_FALSE(cases.empty()) << "missing " << g_dir << "/cases.txt";
  size_t provenance_count = 0;
  size_t equivalent_count = 0;
  for (const FixtureCase& entry : cases) {
    equivalent_count += entry.parallel_equivalent ? 1 : 0;
    provenance_count += entry.official_provenance ? 1 : 0;
    EXPECT_FALSE(entry.official_provenance && !entry.parallel_equivalent)
        << entry.name << ": provenance must imply parallel equivalence";
  }
  EXPECT_GT(provenance_count, 0u);
  EXPECT_GT(equivalent_count, provenance_count)
      << "corpus must contain an equivalent-but-uncertifiable case";
  EXPECT_LT(provenance_count, cases.size());
}

TEST(ScatterRoute, GenericAndUniqueAreDistinguishableWitnesses) {
  WebGPUGraph generic;
  ScatterGraphSpec generic_spec;
  build_scatter_graph(generic, generic_spec);
  const WebGPUDispatch& generic_dispatch = compute_dispatch(generic);
  EXPECT_EQ(generic_dispatch.kernel_name, "scatter");
  EXPECT_EQ(generic_dispatch.workgroup_count_x, 1u);

  WebGPUGraph unique;
  ScatterGraphSpec unique_spec;
  unique_spec.op = kUniqueOp;
  build_scatter_graph(unique, unique_spec);
  const WebGPUDispatch& unique_dispatch = compute_dispatch(unique);
  EXPECT_EQ(unique_dispatch.kernel_name, "scatter_unique_indices");
  EXPECT_EQ(
      unique_dispatch.workgroup_count_x,
      static_cast<uint32_t>(kSelectedCount) / kUniqueLanes);

  // The forced-WG1 control: a witness that reports the parallel route cannot be
  // satisfied by the serial one.
  EXPECT_NE(generic_dispatch.kernel_name, unique_dispatch.kernel_name);
  EXPECT_NE(
      generic_dispatch.workgroup_count_x, unique_dispatch.workgroup_count_x);
  EXPECT_EQ(
      get_webgpu_shader_info("scatter_unique_indices").workgroup_size_x,
      kUniqueLanes);
  EXPECT_EQ(get_webgpu_shader_info("scatter").workgroup_size_x, 1u);
}

TEST(ScatterExactness, MatchesTheCpuAuthorityOnEveryCase) {
  const std::vector<float> base =
      read_bin<float>(g_dir + "/base.bin", kVocabSize);
  ASSERT_FALSE(base.empty()) << "missing " << g_dir << "/base.bin";

  for (const FixtureCase& entry : read_cases()) {
    SCOPED_TRACE(entry.name);
    const std::string prefix = g_dir + "/" + entry.name;
    const std::vector<int32_t> indices =
        read_bin<int32_t>(prefix + ".index.bin", kSelectedCount);
    const std::vector<float> source =
        read_bin<float>(prefix + ".source.bin", kSelectedCount);
    const std::vector<float> expected =
        read_bin<float>(prefix + ".expected.bin", kVocabSize);
    ASSERT_FALSE(indices.empty());
    ASSERT_FALSE(source.empty());
    ASSERT_FALSE(expected.empty());

    const std::vector<float> generic =
        run_route(kGenericOp, base, indices, source);
    ASSERT_EQ(generic.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      ASSERT_EQ(generic[i], expected[i]) << "generic mismatch at " << i;
    }

    if (!entry.parallel_equivalent) {
      continue;
    }
    const std::vector<float> unique =
        run_route(kUniqueOp, base, indices, source);
    ASSERT_EQ(unique.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      ASSERT_EQ(unique[i], expected[i]) << "unique mismatch at " << i;
    }
  }
}

TEST(ScatterExactness, DuplicateDestinationsStayOnTheDuplicateSafeRoute) {
  const std::vector<float> base =
      read_bin<float>(g_dir + "/base.bin", kVocabSize);
  ASSERT_FALSE(base.empty());
  const std::string prefix = g_dir + "/duplicate_destinations";
  const std::vector<int32_t> indices =
      read_bin<int32_t>(prefix + ".index.bin", kSelectedCount);
  const std::vector<float> source =
      read_bin<float>(prefix + ".source.bin", kSelectedCount);
  const std::vector<float> expected =
      read_bin<float>(prefix + ".expected.bin", kVocabSize);
  ASSERT_FALSE(indices.empty() || source.empty() || expected.empty());

  const std::vector<float> generic =
      run_route(kGenericOp, base, indices, source);
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_EQ(generic[i], expected[i]) << "last-write-wins broken at " << i;
  }
}

TEST(ScatterFailsClosed, RejectsMalformedShapesAndScalars) {
  for (const char* op : {kGenericOp, kUniqueOp}) {
    const struct {
      const char* name;
      ScatterGraphSpec spec;
    } cases[] = {
        {"dim != -1",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.dim = 0;
           return s;
         }()},
        {"input width",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.vocab = 1024;
           return s;
         }()},
        {"selected width",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.selected = 2048;
           return s;
         }()},
        {"output width",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.output_vocab = kVocabSize / 2;
           return s;
         }()},
        {"index dtype",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.index_is_int = false;
           return s;
         }()},
        {"source dtype",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.source_is_int = true;
           return s;
         }()},
        {"input dtype",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.input_is_int = true;
           return s;
         }()},
        {"argument count",
         [op] {
           ScatterGraphSpec s;
           s.op = op;
           s.drop_last_arg = true;
           return s;
         }()},
    };
    for (const auto& test_case : cases) {
      SCOPED_TRACE(std::string(op) + " / " + test_case.name);
      WebGPUGraph graph;
      EXPECT_THROW(
          build_scatter_graph(graph, test_case.spec), std::runtime_error);
    }
  }
}

TEST(ScatterFailsClosed, AcceptsTheExactMtpShapeOnBothRoutes) {
  for (const char* op : {kGenericOp, kUniqueOp}) {
    SCOPED_TRACE(op);
    ScatterGraphSpec spec;
    spec.op = op;
    WebGPUGraph graph;
    EXPECT_NO_THROW(build_scatter_graph(graph, spec));
  }
}

} // namespace executorch::backends::webgpu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  executorch::backends::webgpu::g_dir = "/tmp/scatter";
  if (argc > 1) {
    executorch::backends::webgpu::g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_SCATTER_DIR")) {
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
