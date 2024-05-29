/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/custom_ops/op_llama_cpp_linear.h>
#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

#include <torch/library.h>

/**
 * Define the following operators in PyTorch:
 * 
 * llama_cpp::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor
 * 
 * This one has the same semantics of aten::_weight_int8pack_mm
 * 
 * llama_cpp::_weight_int8pack_mm.out(Tensor self, Tensor mat2, Tensor scales, *, Tensor(a!) out) -> Tensor(a!)
 * 
 * This one is the out variant.
*/
namespace torch {
namespace executor {
namespace native {

Tensor& _llama_cpp_mm_int8_out_no_context(
  const Tensor& A, 
  const Tensor& B, 
  const Tensor& scales, 
  Tensor& C) {
    exec_aten::RuntimeContext context{};
    return torch::executor::native::_llama_cpp_mm_int8_out(
        context, A, B, scales, C
    );
}

at::Tensor& _llama_cpp_mm_int8_out_aten(
  const at::Tensor& A, 
  const at::Tensor& B, 
  const at::Tensor& scales, 
  at::Tensor& C) {
    at::Tensor out = at::_weight_int8pack_mm(A, B, scales);
    C.copy_(out);
    return C;
}

at::Tensor _llama_cpp_mm_int8_aten_metal(
  const at::Tensor& A, 
  const at::Tensor& B, 
  const at::Tensor& scales) {
    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);

    TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
                __func__,
                " : expect A to be either 32-bit or 16-bit float tensor.");
    TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
    TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

    TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
    TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
    TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

    auto C = at::empty({M, N}, A.options());
    WRAP_TO_ATEN(_llama_cpp_mm_int8_out_no_context, 3)(A, B, scales, C);
    return C;
}

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY(llama_cpp, m) {
    m.def("_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor");
    m.def("_weight_int8pack_mm.out(Tensor self, Tensor mat2, Tensor scales, *, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(llama_cpp, CompositeExplicitAutograd, m) {
    m.impl("_weight_int8pack_mm", at::_weight_int8pack_mm);
    m.impl("_weight_int8pack_mm.out", torch::executor::native::_llama_cpp_mm_int8_out_aten);
}

TORCH_LIBRARY_IMPL(llama_cpp, Metal, m) {
    m.impl("_weight_int8pack_mm", torch::executor::native::_llama_cpp_mm_int8_aten_metal);
    m.impl("_weight_int8pack_mm.out", WRAP_TO_ATEN(torch::executor::native::_llama_cpp_mm_int8_out_no_context, 3));
}