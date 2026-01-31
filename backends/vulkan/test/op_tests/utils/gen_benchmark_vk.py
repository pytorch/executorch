# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

from executorch.backends.vulkan.test.op_tests.utils.gen_computegraph import (
    ComputeGraphGen,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_correctness_base import (
    CorrectnessTestGen,
)
from executorch.backends.vulkan.test.op_tests.utils.test_suite import VkTestSuite

from torchgen.model import NativeFunction

##########################
## Test Suite Generation ##
##########################

benchmark_fixture_template = """
class GeneratedOpBenchmark_{op_name} : public ::benchmark::Fixture {{
 protected:
  ComputeGraph* graph;
  at::ScalarType test_dtype = at::kFloat;
  float rtol = {rtol};
  float atol = {atol};

  {arg_valuerefs}

  void SetUp(::benchmark::State& state) override {{
    torch::manual_seed(42);
    GraphConfig config;
    config.descriptor_pool_safety_factor = 2.0;
    test_dtype = at::ScalarType(state.range(0));
    const utils::StorageType storage_type = utils::StorageType(state.range(1));
    const utils::GPUMemoryLayout memory_layout = utils::GPUMemoryLayout(state.range(2));
    config.set_storage_type_override(storage_type);
    config.set_memory_layout_override(memory_layout);
    config.enable_querypool = true;
    graph = new ComputeGraph(config);
  }}

  void TearDown(::benchmark::State& state) override {{
    delete graph;
    graph = nullptr;
  }}

  {build_graph_fn}
  {benchmark_fn}
}};
"""

benchmark_template = """
BENCHMARK_DEFINE_F(GeneratedOpBenchmark_{op_name}, {case_name})(benchmark::State& state) {{
    {skips}
    {create_ref_data}
    {call_build_graph}
    ShaderTimes shader_times;
    for (auto _ : state) {{
        {call_benchmark}
        graph->context()->querypool().extract_results();
        QueryPoolResults results = graph->context()->querypool().get_shader_timestamp_data();
        process_querypool_results(results, shader_times);
    }}
    register_shader_time_counters(state, shader_times);
}}

BENCHMARK_REGISTER_F(GeneratedOpBenchmark_{op_name}, {case_name})->Threads(1)->ArgsProduct({combos});
"""


class VkBenchmarkGen(CorrectnessTestGen):
    def __init__(self, op_reg_name: str, f: NativeFunction, inputs: VkTestSuite):
        super().__init__(f, inputs)
        self.op_reg_name = op_reg_name
        self.generator = ComputeGraphGen(
            self.op_reg_name, self.f, self.suite_def, inputs.force_io
        )

    def gen_call_benchmark(self, prepack=False) -> str:
        test_str = f"benchmark_{self.op_name}("
        if prepack:
            test_str = f"prepacked_benchmark_{self.op_name}("
        for binding in self.f_sig.arguments():
            arg = binding.argument
            test_str += f"{arg.name}, "
        test_str = test_str[:-2] + ");"
        test_str = re.sub(r"^", "  ", test_str, flags=re.M)
        return test_str

    def gen_call_build_graph(self, prepack=False) -> str:
        test_str = f"build_graph_{self.op_name}("
        if prepack:
            test_str = f"prepacked_build_graph_{self.op_name}("
        for binding in self.f_sig.arguments():
            arg = binding.argument
            test_str += f"{arg.name}, "
        test_str = test_str[:-2] + ");"
        test_str = re.sub(r"^", "  ", test_str, flags=re.M)
        return test_str

    def gen_combos(self, inputs) -> str:
        dtypes_list = ", ".join(f"int({dtype})" for dtype in self.suite_def.dtypes)
        storage_types_list = ", ".join(
            f"int({storage_type})" for storage_type in self.suite_def.storage_types
        )
        layouts_list = ", ".join(f"int({layout})" for layout in self.suite_def.layouts)
        return f"{{ {{ {dtypes_list} }}, {{ {storage_types_list} }}, {{ {layouts_list} }} }}"

    def generate_benchmark_case(self, inputs, prepack=False) -> str:
        return benchmark_template.format(
            op_name=f"{self.op_name}",
            case_name=self.gen_case_name(inputs, prepack),
            skips=self.generator.gen_conditional_skips(
                'state.SkipWithError("unsupported type"); return;'
            ),
            create_ref_data=self.gen_create_ref_data(inputs),
            call_build_graph=self.gen_call_build_graph(prepack),
            call_benchmark=self.gen_call_benchmark(prepack),
            combos=self.gen_combos(inputs),
        )

    def generate_benchmark(self) -> str:
        benchmarks_cpp = ""
        for inputs in self.suite_def.input_cases:
            if not self.suite_def.requires_prepack:
                benchmarks_cpp += self.generate_benchmark_case(inputs)
            if self.suite_def.supports_prepack():
                benchmarks_cpp += self.generate_benchmark_case(inputs, prepack=True)
        return benchmarks_cpp

    def generate_benchmark_fixture(self) -> str:
        build_graph_fn = ""
        benchmark_fn = ""
        if not self.suite_def.requires_prepack:
            build_graph_fn = self.generator.gen_build_graph_fn()
            benchmark_fn = self.generator.gen_op_exec_graph_fn()

        prepacked_build_graph_fn = ""
        prepacked_benchmark_fn = ""
        if self.suite_def.supports_prepack():
            self.generator.should_prepack = True
            prepacked_build_graph_fn = self.generator.gen_build_graph_fn()
            build_graph_fn += "\n\n  "
            build_graph_fn += prepacked_build_graph_fn
            prepacked_benchmark_fn = self.generator.gen_op_exec_graph_fn()
            benchmark_fn += "\n\n  "
            benchmark_fn += prepacked_benchmark_fn

        return benchmark_fixture_template.format(
            op_name=self.op_name,
            build_graph_fn=build_graph_fn,
            benchmark_fn=benchmark_fn,
            rtol=self.suite_def.rtol,
            arg_valuerefs=self.generator.gen_arg_valueref_decls(),
            atol=self.suite_def.atol,
        )


##########################
## Test File Generation ##
##########################

cpp_test_template = """
#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <benchmark/benchmark.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

using namespace vkcompute;
using TensorOptions = at::TensorOptions;

vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {{
  switch (at_scalartype) {{
    case c10::kDouble:
      return vkapi::kDouble;
    case c10::kFloat:
      return vkapi::kFloat;
    case c10::kHalf:
      return vkapi::kHalf;
    case c10::kInt:
      return vkapi::kInt;
    case c10::kLong:
      return vkapi::kInt;
    case c10::kChar:
      return vkapi::kChar;
    case c10::kBool:
      return vkapi::kBool;
    default:
      VK_THROW("Unsupported at::ScalarType!");
  }}
}}

at::Tensor make_casted_randint_tensor(
    std::vector<int64_t> sizes,
    at::ScalarType dtype = at::kFloat,
    int64_t low = 1,
    int64_t high = 20) {{

  // For some reason range needs to be passed in as explicit variables
  // otherwise 0s will be generated.
  return at::randint(1, 20, sizes, at::device(at::kCPU).dtype(dtype));
}}

at::Tensor make_rand_tensor(
    std::vector<int64_t> sizes,
    at::ScalarType dtype = at::kFloat,
    float low = 0.0,
    float high = 1.0) {{
  if (high == 1.0 && low == 0.0)
    return at::rand(sizes, at::device(at::kCPU).dtype(dtype));

  if (dtype == at::kChar)
    return at::randint(high, sizes, at::device(at::kCPU).dtype(dtype));

  return at::rand(sizes, at::device(at::kCPU).dtype(dtype)) * (high - low) + low;
}}

at::Tensor make_seq_tensor(
    std::vector<int64_t> sizes,
    at::ScalarType dtype = at::kFloat,
    float low = 0.0,
    float high = 1.0) {{
  (void)low;
  (void)high;

  int64_t n = 1;
  for (auto size: sizes) {{
    n *= size;
  }}

  std::vector<float> values(n);
  for (int i=0;i<n;i++) {{
    values[i] = (float) (i + 1);
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(values.data(), sizes, at::kFloat).toType(dtype).detach().clone();
}}

at::Tensor make_index_tensor_1d(std::vector<int32_t> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{static_cast<int64_t>(indices.size())}};

  // Clone as original data will be deallocated upon return.
  return at::from_blob(indices.data(), sizes, dtype).detach().clone();
}}

at::Tensor make_index_tensor_2d(std::vector<std::vector<int32_t>> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{
    static_cast<int64_t>(indices.size()),
    static_cast<int64_t>(indices[0].size())}};

  // Flatten indices as from_blob reads garbage otherwise.
  std::vector<int64_t> acc;
  for (auto& vec: indices) {{
    acc.insert(acc.end(), vec.begin(), vec.end());
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(acc.data(), sizes, dtype).detach().clone();
}}

at::Tensor make_index_tensor_3d(std::vector<std::vector<std::vector<int32_t>>> indices) {{
  at::ScalarType dtype = at::kInt;
  std::vector<int64_t> sizes = {{
    static_cast<int64_t>(indices.size()),
    static_cast<int64_t>(indices[0].size()),
    static_cast<int64_t>(indices[0][0].size())}};

  // Flatten indices as from_blob reads garbage otherwise.
  std::vector<int64_t> acc;
  for (auto& v: indices) {{
    for (auto& vv: v) {{
      acc.insert(acc.end(), vv.begin(), vv.end());
    }}
  }}

  // Clone as original data will be deallocated upon return.
  return at::from_blob(acc.data(), sizes, dtype).detach().clone();
}}

using QueryPoolResults = std::vector<vkcompute::vkapi::ShaderResult>;
using ShaderTimes = std::unordered_map<std::string, std::vector<uint64_t>>;

void process_querypool_results(
    QueryPoolResults& results,
    ShaderTimes& shader_times) {{
  for (const vkcompute::vkapi::ShaderResult& r : results) {{
    uint64_t duration_ns = r.end_time_ns - r.start_time_ns;
    if (shader_times.find(r.kernel_name) == shader_times.end()) {{
      shader_times[r.kernel_name] = std::vector<uint64_t>();
    }}
    shader_times[r.kernel_name].emplace_back(duration_ns);
  }}
}}

void register_shader_time_counters(
    benchmark::State& state,
    ShaderTimes& shader_times) {{
  for (auto& times_list : shader_times) {{
    // Filter to_nchw and nchw_to shaders
    if (times_list.first.find("to_nchw") != std::string::npos) {{
        continue;
    }}
    if (times_list.first.find("nchw_to") != std::string::npos) {{
        continue;
    }}

    std::sort(times_list.second.begin(), times_list.second.end());
    uint64_t median_time;
    median_time = times_list.second[times_list.second.size() / 2];
    state.counters[times_list.first + " median ns"] = median_time;
  }}
}}

{benchmark_fixtures}

{def_benchmarks}
"""


class VkBenchmarkFileGen:
    def __init__(self, out_path):
        self.out_path = out_path
        self.suites_gens = []

    def add_suite(self, op_reg_name: str, f: NativeFunction, all_input_cases) -> None:
        suites_gen = VkBenchmarkGen(op_reg_name, f, all_input_cases)
        self.suites_gens.append(suites_gen)

    def generate_benchmarks_cpp(self) -> str:
        return "\n".join([h.generate_benchmark() for h in self.suites_gens])

    def generate_benchmark_fixtures(self) -> str:
        return "\n".join([h.generate_benchmark_fixture() for h in self.suites_gens])

    def generate_cpp(self) -> str:
        return cpp_test_template.format(
            benchmark_fixtures=self.generate_benchmark_fixtures(),
            def_benchmarks=self.generate_benchmarks_cpp(),
        )
