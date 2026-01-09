/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/prim_ops/et_copy_index.h>
#include <executorch/kernels/prim_ops/et_view.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

/*
For internal builds using buck rules, the target that depends on
selective prim ops, will manage its own artifacts. It is in the
artifacts directory where the geneated selected_prim_ops.h resides
and thus compilation sources must be copied there including
selective_build_prim_ops.h. Hence it does not have fully qualified
name unlike the header files above.
*/
#ifdef ET_PRIM_OPS_SELECTIVE_BUILD
#include "selective_build_prim_ops.h"
#endif

#include <algorithm>
#include <cmath>

using torch::executor::function::et_copy_index;

namespace torch {
namespace executor {
namespace function {

namespace {

#define __ET_PRIM_OP_ERROR_IMPL(a, b, context) \
  else {                                       \
    ET_KERNEL_CHECK_MSG(                       \
        context,                               \
        false,                                 \
        InvalidType,                           \
        /* void */,                            \
        "%zu, %zu",                            \
        (size_t)a.tag,                         \
        (size_t)b.tag);                        \
  }

#define __NUMBER_ET_PRIM_OP_IMPL(operator, stack, context) \
  EValue& a = *stack[0];                                   \
  EValue& b = *stack[1];                                   \
  EValue& out = *stack[2];                                 \
  if (a.isInt() && b.isInt()) {                            \
    out = EValue(a.toInt() operator b.toInt());            \
  } else if (a.isDouble() && b.isDouble()) {               \
    out = EValue(a.toDouble() operator b.toDouble());      \
  } else if (a.isInt() && b.isDouble()) {                  \
    out = EValue(a.toInt() operator b.toDouble());         \
  } else if (a.isDouble() && b.isInt()) {                  \
    out = EValue(a.toDouble() operator b.toInt());         \
  }

#define __ET_PRIM_OP_NUM_ARGS_CHECK_IMPL(stack, context) \
  ET_KERNEL_CHECK_MSG(                                   \
      context,                                           \
      stack.size() == 3,                                 \
      InvalidProgram,                                    \
      /* void */,                                        \
      "Expected %zu args, got %zu",                      \
      (size_t)3,                                         \
      stack.size());

#define ALGEBRA_ET_PRIM_OP(operator, stack, context) \
  __ET_PRIM_OP_NUM_ARGS_CHECK_IMPL(stack, context)   \
  __NUMBER_ET_PRIM_OP_IMPL(operator, stack, context) \
  __ET_PRIM_OP_ERROR_IMPL(a, b, context)

#define BOOLEAN_ET_PRIM_OP(operator, stack, context) \
  __ET_PRIM_OP_NUM_ARGS_CHECK_IMPL(stack, context)   \
  __NUMBER_ET_PRIM_OP_IMPL(operator, stack, context) \
  else if (a.isBool() && b.isBool()) {               \
    out = EValue(a.toBool() operator b.toBool());    \
  }                                                  \
  __ET_PRIM_OP_ERROR_IMPL(a, b, context)

void floor_div_double(double a, double b, EValue& out) {
  if (b == 0) {
    out = EValue(std::signbit(a) ? -INFINITY : INFINITY);
    return;
  }
  const auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && std::signbit(b) != std::signbit(mod)) {
    out = EValue(div - 1);
    return;
  }
  out = EValue(div);
}

static Kernel prim_ops[] = {
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_ATEN_SYM_SIZE_INT)
    // aten::sym_size.int(Tensor self, int dim) -> SymInt
    Kernel(
        "aten::sym_size.int",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& self = *stack[0];
          EValue& dim = *stack[1];
          EValue& out = *stack[2];
          executorch::aten::Tensor self_tensor =
              self.to<executorch::aten::Tensor>();
          int64_t dim_val = dim.to<int64_t>();
          int64_t size = self_tensor.size(dim_val);
          out = EValue(size);
        }),
#endif
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_ATEN_LOCAL_SCALAR_DENSE)
    // aten::_local_scalar_dense(Tensor self) -> Scalar
    Kernel(
        "aten::_local_scalar_dense",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& self = *stack[0];
          EValue& out = *stack[1];
          executorch::aten::Tensor self_tensor =
              self.to<executorch::aten::Tensor>();
          ET_KERNEL_CHECK_MSG(
              context,
              self_tensor.numel() >= 1,
              InvalidArgument,
              /* void */,
              "Expected tensor with at least 1 element");
          ET_SWITCH_REAL_TYPES_AND(
              Bool,
              self_tensor.scalar_type(),
              context,
              "_local_scalar_dense",
              CTYPE,
              [&]() {
                out = EValue(Scalar(self_tensor.const_data_ptr<CTYPE>()[0]));
              });
        }),
#endif
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_ATEN_SYM_NUMEL)
    // aten::sym_numel(Tensor self) -> SymInt
    Kernel(
        "aten::sym_numel",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& self = *stack[0];
          EValue& out = *stack[1];
          executorch::aten::Tensor self_tensor =
              self.to<executorch::aten::Tensor>();
          int64_t numel = self_tensor.numel();
          out = EValue(numel);
        }),
#endif
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_SYM_MAX_SCALAR)
    // executorch_prim::sym_max.Scalar(SymInt a, SymInt b) -> SymInt
    Kernel(
        "executorch_prim::sym_max.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());

          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(std::max(a.toInt(), b.toInt()));
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "sym_max only supports int inputs, got %zu, %zu",
                (size_t)a.tag,
                (size_t)b.tag);
          }
        }),
#endif
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_SYM_MIN_SCALAR)
    // executorch_prim::sym_min.Scalar(SymInt a, SymInt b) -> SymInt
    Kernel(
        "executorch_prim::sym_min.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(std::min(a.toInt(), b.toInt()));
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "sym_min only supports int inputs, got %zu, %zu",
                (size_t)a.tag,
                (size_t)b.tag);
          }
        }),
#endif
#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_ADD_SCALAR)
    // executorch_prim::add.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::add.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ALGEBRA_ET_PRIM_OP(+, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_SUB_SCALAR)
    // executorch_prim::sub.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::sub.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ALGEBRA_ET_PRIM_OP(-, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_MUL_SCALAR)
    // executorch_prim::mul.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::mul.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ALGEBRA_ET_PRIM_OP(*, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_FLOORDIV_SCALAR)
    /**
     * Python's __floordiv__ operator is more complicated than just floor(a /
     * b). It aims to maintain the property: a == (a // b) * b + remainder(a, b)
     * which can otherwise fail due to rounding errors in the remainder.
     * So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
     * With some additional fix-ups added to the result.
     *
     * executorch_prim::floordiv.Scalar(Scalar, Scalar) -> Scalar
     */
    Kernel(
        "executorch_prim::floordiv.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            const int64_t quot = a.toInt() / b.toInt();
            if ((a.toInt() < 0) == (b.toInt() < 0)) {
              out = EValue(quot);
              return;
            }
            const int64_t rem = a.toInt() % b.toInt();
            out = EValue(rem ? quot - 1 : quot);
            return;
          } else if (a.isDouble() && b.isDouble()) {
            floor_div_double(a.toDouble(), b.toDouble(), out);
          } else if (a.isInt() && b.isDouble()) {
            floor_div_double(static_cast<double>(a.toInt()), b.toDouble(), out);
          } else if (a.isDouble() && b.isInt()) {
            floor_div_double(a.toDouble(), static_cast<double>(b.toInt()), out);
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "%zu, %zu",
                (size_t)a.tag,
                (size_t)b.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_TRUEDIV_SCALAR)
    // executorch_prim::truediv.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::truediv.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          // can't use macro because of custom casting behavior
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(
                static_cast<double>(a.toInt()) /
                static_cast<double>(b.toInt()));
          } else if (a.isDouble() && b.isDouble()) {
            out = EValue(a.toDouble() / b.toDouble());
          } else if (a.isInt() && b.isDouble()) {
            out = EValue(a.toInt() / b.toDouble());
          } else if (a.isDouble() && b.isInt()) {
            out = EValue(a.toDouble() / b.toInt());
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "%zu, %zu",
                (size_t)a.tag,
                (size_t)b.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_SYM_FLOAT_SCALAR)
    // executorch_prim::sym_float.Scalar(Scalar) -> Scalar
    Kernel(
        "executorch_prim::sym_float.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          // can't use macro because of custom casting behavior
          // TODO: Now that we are reliably generating conversion operators,
          // we can remove the mixed type handling for other operators
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& a = *stack[0];
          EValue& out = *stack[1];
          if (a.isInt()) {
            out = EValue(static_cast<double>(a.toInt()));
          } else if (a.isDouble()) {
            // TODO: This should be impossible
            out = EValue(a.toDouble());
          } else {
            ET_KERNEL_CHECK_MSG(
                context, false, InvalidType, /* void */, "%zu", (size_t)a.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_EQ_SCALAR)
    // executorch_prim::eq.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::eq.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          BOOLEAN_ET_PRIM_OP(==, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_GT_SCALAR)
    // executorch_prim::gt.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::gt.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          BOOLEAN_ET_PRIM_OP(>, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_LT_SCALAR)
    // executorch_prim::lt.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::lt.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          BOOLEAN_ET_PRIM_OP(<, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_GE_SCALAR)
    // executorch_prim::ge.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::ge.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          BOOLEAN_ET_PRIM_OP(>=, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_LE_SCALAR)
    // executorch_prim::le.Scalar(Scalar, Scalar) -> bool
    Kernel(
        "executorch_prim::le.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          BOOLEAN_ET_PRIM_OP(<=, stack, context);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_NEG_SCALAR)
    // executorch_prim::neg.Scalar(Scalar) -> Scalar
    Kernel(
        "executorch_prim::neg.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& a = *stack[0];
          EValue& out = *stack[1];
          if (a.isInt()) {
            out = EValue(-a.toInt());
          } else if (a.isDouble()) {
            out = EValue(-a.toDouble());
          } else {
            ET_KERNEL_CHECK_MSG(
                context, false, InvalidType, /* void */, "%zu", (size_t)a.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_FLOORDIV_INT)
    // executorch_prim::floordiv.int(int, int) -> int
    Kernel(
        "executorch_prim::floordiv.int",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() / b.toInt());
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_MOD_INT)
    // executorch_prim::mod.int(int, int) -> int
    Kernel(
        "executorch_prim::mod.int",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() % b.toInt());
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_MOD_SCALAR)
    // executorch_prim::mod.Scalar(Scalar, Scalar) -> Scalar
    Kernel(
        "executorch_prim::mod.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 3,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)3,
              stack.size());
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          if (a.isInt() && b.isInt()) {
            out = EValue(a.toInt() % b.toInt());
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "%zu, %zu",
                (size_t)a.tag,
                (size_t)b.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_CEIL_SCALAR)
    // ceil.Scalar(Scalar a) -> Scalar
    Kernel(
        "executorch_prim::ceil.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& a = *stack[0];
          EValue& out = *stack[1];
          if (a.isDouble()) {
            out = EValue(static_cast<int64_t>(ceil(a.toDouble())));
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "Unsupported DType %zu",
                (size_t)a.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_ROUND_SCALAR)
    // round.Scalar(Scalar a) -> Scalar
    Kernel(
        "executorch_prim::round.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& a = *stack[0];
          EValue& out = *stack[1];
          if (a.isDouble()) {
            // Round half to even to match Python round(). Need an explicit
            // implementation as not all platforms support fenv rounding modes.
            // See
            // https://codeyarns.com/tech/2018-08-17-how-to-round-half-to-even.html
            const auto val = a.toDouble();
            const auto r = round(val);
            const auto d = r - val;
            auto res = 0.0;

            if (std::abs(d) != 0.5) {
              res = r;
            } else if (fmod(r, 2.0) == 0.0) {
              res = r;
            } else {
              res = val - d;
            }

            out = EValue(static_cast<int64_t>(res));
          } else {
            ET_KERNEL_CHECK_MSG(
                context,
                false,
                InvalidType,
                /* void */,
                "Unsupported DType %zu",
                (size_t)a.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_TRUNC_SCALAR)
    // trunc.Scalar(Scalar a) -> Scalar
    Kernel(
        "executorch_prim::trunc.Scalar",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          ET_KERNEL_CHECK_MSG(
              context,
              stack.size() == 2,
              InvalidProgram,
              /* void */,
              "Expected %zu args, got %zu",
              (size_t)2,
              stack.size());
          EValue& a = *stack[0];
          EValue& out = *stack[1];
          if (a.isDouble()) {
            out = EValue(static_cast<int64_t>(trunc(a.toDouble())));
          } else {
            ET_KERNEL_CHECK_MSG(
                context, false, InvalidType, /* void */, "%zu", (size_t)a.tag);
          }
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_ET_COPY_INDEX_TENSOR)
    // executorch_prim::et_copy_index.tensor(tensor, tensor) -> tensor
    Kernel(
        "executorch_prim::et_copy_index.tensor",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          et_copy_index(context, stack);
        }),
#endif

#if !defined(EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD) || \
    defined(INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT)
    // executorch_prim::et_view.default(Tensor, int[]) -> Tensor
    Kernel(
        "executorch_prim::et_view.default",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
          et_view(context, stack);
        }),
#endif

};

executorch::runtime::Span<const executorch::ET_RUNTIME_NAMESPACE::Kernel>
    kernel_span(prim_ops, prim_ops + sizeof(prim_ops) / sizeof(Kernel));

// Return value not used. Keep the static variable assignment to register
// operators in static initialization time.
auto success_with_kernel_reg =
    executorch::ET_RUNTIME_NAMESPACE::register_kernels(kernel_span);

} // namespace
} // namespace function
} // namespace executor
} // namespace torch
