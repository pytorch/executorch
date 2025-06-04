# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.vulkan.test.op_tests.utils.gen_computegraph import (
    ComputeGraphGen,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_correctness_base import (
    CorrectnessTestFileGen,
    CorrectnessTestGen,
)
from executorch.backends.vulkan.test.op_tests.utils.test_suite import VkTestSuite

from torchgen.model import NativeFunction

##################################
## Test Fixture Code Generation ##
##################################

test_fixture_template = """
class GeneratedOpsTest_{op_name} : public ::testing::TestWithParam< ::std::tuple<at::ScalarType, utils::StorageType, utils::GPUMemoryLayout>> {{
 protected:
  ComputeGraph* graph;
  at::ScalarType test_dtype = at::kFloat;
  float rtol = {rtol};
  float atol = {atol};

  void SetUp() override {{
    GraphConfig config;
    utils::StorageType default_storage_type;
    utils::GPUMemoryLayout default_memory_layout;
    std::tie(test_dtype, default_storage_type, default_memory_layout) = GetParam();
    config.set_storage_type_override(default_storage_type);
    config.set_memory_layout_override(default_memory_layout);
    graph = new ComputeGraph(config);

    if (test_dtype == at::kHalf) {{
      rtol = 1e-2;
      atol = 1e-2;
    }}
  }}

  void TearDown() override {{
    delete graph;
    graph = nullptr;
  }}

  {check_fn}
}};
"""


class VkCorrectnessTestGen(CorrectnessTestGen):
    def __init__(self, op_reg_name: str, f: NativeFunction, inputs: VkTestSuite):
        super().__init__(f, inputs)
        self.op_reg_name = op_reg_name
        self.generator = ComputeGraphGen(self.op_reg_name, self.f, self.suite_def)

    def generate_fixture_cpp(self) -> str:
        check_fn = ""
        if not self.suite_def.requires_prepack:
            check_fn = self.generator.gen_op_check_fn()

        prepacked_check_fn = ""
        if self.suite_def.supports_prepack():
            self.generator.should_prepack = True
            prepacked_check_fn = self.generator.gen_op_check_fn()
            check_fn += "\n\n  "
            check_fn += prepacked_check_fn

        return test_fixture_template.format(
            op_name=self.op_name,
            check_fn=check_fn,
            rtol=self.suite_def.rtol,
            atol=self.suite_def.atol,
        )

    def gen_parameterization(self) -> str:
        dtypes = self.suite_def.dtypes
        storage_types = self.suite_def.storage_types
        layouts = self.suite_def.layouts

        return f"""
INSTANTIATE_TEST_SUITE_P(
  Combos_{self.op_name},
  GeneratedOpsTest_{self.op_name},
    ::testing::Combine(
      ::testing::Values({', '.join(dtypes)}),
      ::testing::Values({', '.join(storage_types)}),
      ::testing::Values({', '.join(layouts)})));
        """


##############################
## Test File Code Generation ##
###############################

preamble_str = """
#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <tuple>

using namespace vkcompute;
using TensorOptions = at::TensorOptions;

vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
  switch (at_scalartype) {
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
  }
}

#ifdef USE_VULKAN_FP16_INFERENCE
bool check_close(at::Tensor& t1, at::Tensor& t2, float rtol=1e-2, float atol=1e-2) {
#else
bool check_close(at::Tensor& t1, at::Tensor& t2, float rtol=1e-5, float atol=1e-5) {
#endif
  // Skip checking index tensors
  if (t1.scalar_type() == at::kLong || t2.scalar_type() == at::kLong) {
    return true;
  }
  bool is_close = at::allclose(t1, t2, rtol, atol);
  if (!is_close && t1.numel() < 500) {
    std::cout << "reference: " << std::endl;
    print(t1, 150);
    std::cout << std::endl;
    std::cout << "vulkan: " << std::endl;
    print(t2, 150);
    std::cout << std::endl;
  }
  return is_close;
}
"""


class VkCorrectnessTestFileGen(CorrectnessTestFileGen):
    def __init__(self, out_path: str):
        super().__init__(out_path)

    def generate_preamble(self) -> str:
        return preamble_str

    def add_suite(self, op_reg_name: str, f: NativeFunction, all_input_cases) -> None:
        suites_gen = VkCorrectnessTestGen(op_reg_name, f, all_input_cases)
        self.suites_gens.append(suites_gen)
