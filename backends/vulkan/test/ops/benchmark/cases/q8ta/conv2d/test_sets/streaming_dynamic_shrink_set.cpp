// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

void execute_streaming_dynamic_shrink_test() {
  Conv2dConfig config = {
      OutInChannels(4, 64),
      InputSize2D(30, 99),
      KernelSize(3, 3),
      Stride(1, 1),
      Padding(1, 1),
      Dilation(1, 1),
      1,
      10};
  config.op_name = "conv2d_q8ta_q8csw_q8to";
  config.test_case_name =
      make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
  TestCase test_case = create_test_case_from_config(
      config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_auto",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT);
  for (ValueSpec& input : test_case.inputs()) {
    input.ensure_data_generated(/*seed=*/0);
  }

  BenchmarkGraph benchmark_graph = setup_compute_graph(
      test_case, test_case.operator_name(), /*op_invocations_per_execute=*/1);
  ComputeGraph& graph = *benchmark_graph.graph;
  graph.prepare();
  graph.prepack();

  const std::vector<int64_t> upper_input_sizes = test_case.inputs().at(0).sizes;
  const std::vector<int64_t> shrunk_input_sizes = {1, 64, 20, 51};
  const std::vector<int64_t> shrunk_output_sizes = {1, 4, 20, 51};
  TestCase shrunk_case = test_case;
  shrunk_case.inputs().at(0).sizes = shrunk_input_sizes;
  shrunk_case.inputs().at(0).resize_data(1 * 64 * 20 * 51);
  shrunk_case.outputs().at(0).sizes = shrunk_output_sizes;
  shrunk_case.outputs().at(0).resize_data(1 * 4 * 20 * 51);
  reference_impl(shrunk_case);

  constexpr int kRepetitions = 4;
  for (int repetition = 0; repetition < kRepetitions; ++repetition) {
    graph.resize_input(0, upper_input_sizes);
    graph.propagate_resize();
    graph.maybe_cast_and_copy_into_staging(
        graph.inputs().at(0).staging,
        test_case.inputs().at(0).get_data_ptr(),
        test_case.inputs().at(0).numel(),
        vkapi::kFloat);
    graph.execute();

    graph.resize_input(0, shrunk_input_sizes);
    graph.propagate_resize();
    if (graph.sizes_of(graph.outputs().at(0).value) != shrunk_output_sizes) {
      throw std::runtime_error("streaming im2col output did not shrink");
    }
    graph.maybe_cast_and_copy_into_staging(
        graph.inputs().at(0).staging,
        shrunk_case.inputs().at(0).get_data_ptr(),
        shrunk_case.inputs().at(0).numel(),
        vkapi::kFloat);
    graph.execute();
    graph.maybe_cast_and_copy_from_staging(
        graph.outputs().at(0).staging,
        shrunk_case.outputs().at(0).get_mutable_data_ptr(),
        shrunk_case.outputs().at(0).numel(),
        vkapi::kFloat);
    if (!shrunk_case.outputs().at(0).validate_against_reference(
            shrunk_case.get_abs_tolerance(), shrunk_case.get_rel_tolerance())) {
      throw std::runtime_error("streaming im2col shrink output was stale");
    }
  }

  const Q8taConv2dStreamPlan upper_plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/10,
      /*flattened_kernel_size=*/576,
      /*out_height=*/30,
      /*out_width=*/99,
      kQ8taConv2dIm2ColScratchBudgetBytes);
  if (!upper_plan.feasible || upper_plan.num_tiles != 2 ||
      upper_plan.rows_per_tile <= 20) {
    throw std::runtime_error("dynamic shrink test did not create dead tiles");
  }
  const int64_t resized_scratch_bytes =
      576 * upper_plan.rows_per_tile * utils::align_up_4(51);
  if (resized_scratch_bytes > kQ8taConv2dIm2ColScratchBudgetBytes) {
    throw std::runtime_error("streaming im2col scratch exceeded its cap");
  }
  std::cout << "Streaming im2col dynamic shrink PASSED" << std::endl;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "streaming_dynamic_shrink") {
  TestCaseSet set;
  set.custom_test = execute_streaming_dynamic_shrink_test;
  return set;
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
