/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/prim_ops/et_copy_index.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

using KernelArrayRef = ::torch::executor::ArrayRef<::torch::executor::Kernel>;
using torch::executor::function::et_copy_index;

namespace torch {
namespace executor {
namespace function {

namespace {

static Kernel prim_ops[] = {
    // aten::sym_size.int(Tensor self, int dim) -> SymInt
    Kernel(
        "aten::sym_size.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& self = *stack[0];
          EValue& dim = *stack[1];
          EValue& out = *stack[2];
          exec_aten::Tensor self_tensor = self.to<exec_aten::Tensor>();
          int64_t dim_val = dim.to<int64_t>();
          int64_t size = self_tensor.size(dim_val);
          out = EValue(size);
        }),
    // aten::sym_numel(Tensor self) -> SymInt
    Kernel(
        "aten::sym_numel",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& self = *stack[0];
          EValue& out = *stack[1];
          exec_aten::Tensor self_tensor = self.to<exec_aten::Tensor>();
          int64_t numel = self_tensor.numel();
          out = EValue(numel);
        }),
    // executorch_prim::add.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::add.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() + b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() + b.toDouble());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::sub.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::sub.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() - b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() - b.toDouble());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::mul.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::mul.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() * b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() * b.toDouble());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::floordiv.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::floordiv.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() / b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() / b.toDouble());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::eq.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::eq.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() == b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() == b.toDouble());
          } else if (a.isBool() && b.isBool()) {
            out = EValue(a.toBool() == b.toBool());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::gt.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::gt.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() > b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() > b.toDouble());
          } else if (a.isBool() && b.isBool()) {
            out = EValue(a.toBool() > b.toBool());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::lt.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::lt.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() < b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() < b.toDouble());
          } else if (a.isBool() && b.isBool()) {
            out = EValue(a.toBool() < b.toBool());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::ge.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::ge.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() >= b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() >= b.toDouble());
          } else if (a.isBool() && b.isBool()) {
            out = EValue(a.toBool() >= b.toBool());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::le.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::le.Scalar",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() <= b.toInt());
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() <= b.toDouble());
          } else if (a.isBool() && b.isBool()) {
            out = EValue(a.toBool() <= b.toBool());
          } else {
            // TODO Fail using runtime context
            ET_CHECK(false);
          }
        }),

    // executorch_prim::floordiv.int(int, int) -> int
    Kernel(
        "executorch_prim::floordiv.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() / b.toInt());
        }),

    // executorch_prim::et_copy_index.tensor(tensor, tensor) -> tensor
    Kernel("executorch_prim::et_copy_index.tensor", &et_copy_index),

};

static KernelArrayRef kernel_array_ref(
    prim_ops,
    prim_ops + sizeof(prim_ops) / sizeof(Kernel));

// Return value not used. Keep the static variable assignment to register
// operators in static initialization time.
static auto success_with_kernel_reg = register_kernels(kernel_array_ref);

} // namespace
} // namespace function
} // namespace executor
} // namespace torch
