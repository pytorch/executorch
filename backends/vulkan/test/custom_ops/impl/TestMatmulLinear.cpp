/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCommon.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Linear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

// impl_selector values:
//   "default"  use the registered aten.{mm,bmm,addmm,linear}.default op
//              (the same routing production code goes through)
//   "coopmat"  force the cooperative-matrix path
//   "tiled"    force the tiled path (linear_vec / matmul_vec)

void test_mm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef mat1 = args.at(0);
  const ValueRef mat2 = args.at(1);
  const ValueRef impl_selector_str = args.at(2);
  const ValueRef out = args.at(3);

  std::string impl_selector = graph.extract_string(impl_selector_str);

  if (impl_selector == "coopmat") {
    if (graph.val_is_tref(mat2)) {
      auto mat2_sizes = graph.sizes_of(mat2);
      int64_t B = mat2_sizes.size() >= 3 ? mat2_sizes.at(0) : 1;
      ValueRef packed = prepack_fp_linear_weight(
          graph, mat2, /*is_transposed=*/false, B, /*force_buffer=*/true);
      add_linear_coopmat_node(
          graph,
          mat1,
          packed,
          kDummyValueRef,
          false,
          out,
          utils::safe_downcast<int32_t>(B));
    } else {
      add_matmul_coopmat_node(graph, mat1, mat2, out);
    }
    return;
  }

  if (impl_selector == "tiled") {
    if (graph.val_is_tref(mat2)) {
      auto mat2_sizes = graph.sizes_of(mat2);
      int64_t B = mat2_sizes.size() >= 3 ? mat2_sizes.at(0) : 1;
      ValueRef packed =
          prepack_fp_linear_weight(graph, mat2, /*is_transposed=*/false, B);
      add_linear_tiled_node(
          graph,
          mat1,
          packed,
          kDummyValueRef,
          false,
          out,
          utils::safe_downcast<int32_t>(B));
    } else {
      add_matmul_tiled_node(graph, mat1, mat2, out);
    }
    return;
  }

  std::string op_name = "aten.mm." + impl_selector;
  VK_GET_OP_FN(op_name.c_str())(graph, {mat1, mat2, out});
}

void test_bmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef mat1 = args.at(0);
  const ValueRef mat2 = args.at(1);
  const ValueRef impl_selector_str = args.at(2);
  const ValueRef out = args.at(3);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.bmm." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {mat1, mat2, out});
}

void test_addmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef self = args.at(0);
  const ValueRef mat1 = args.at(1);
  const ValueRef mat2 = args.at(2);
  const ValueRef beta = args.at(3);
  const ValueRef alpha = args.at(4);
  const ValueRef impl_selector_str = args.at(5);
  const ValueRef out = args.at(6);

  std::string impl_selector = graph.extract_string(impl_selector_str);
  std::string op_name = "aten.addmm." + impl_selector;

  VK_GET_OP_FN(op_name.c_str())(graph, {self, mat1, mat2, beta, alpha, out});
}

void test_linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef impl_selector_str = args.at(3);
  const ValueRef out = args.at(4);

  std::string impl_selector = graph.extract_string(impl_selector_str);

  if (impl_selector == "coopmat" || impl_selector == "tiled") {
    bool has_bias = graph.val_is_not_none(bias);
    bool force_buffer = (impl_selector == "coopmat");
    ValueRef packed_weight = prepack_fp_linear_weight(
        graph, weight, /*is_transposed=*/true, /*B=*/1, force_buffer);
    ValueRef packed_bias = kDummyValueRef;
    if (has_bias) {
      packed_bias = prepack_standard(
          graph,
          bias,
          graph.storage_type_of(out),
          utils::kWidthPacked,
          /*passthrough=*/force_buffer);
    }
    if (impl_selector == "coopmat") {
      add_linear_coopmat_node(
          graph, input, packed_weight, packed_bias, has_bias, out);
    } else {
      add_linear_tiled_node(
          graph, input, packed_weight, packed_bias, has_bias, out);
    }
    return;
  }

  std::string op_name = "aten.linear." + impl_selector;
  VK_GET_OP_FN(op_name.c_str())(graph, {input, weight, bias, out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_mm.default, test_mm);
  VK_REGISTER_OP(test_etvk.test_bmm.default, test_bmm);
  VK_REGISTER_OP(test_etvk.test_addmm.default, test_addmm);
  VK_REGISTER_OP(test_etvk.test_linear.default, test_linear);
}

} // namespace vkcompute
