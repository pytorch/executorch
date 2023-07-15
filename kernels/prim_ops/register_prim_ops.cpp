#include <executorch/kernels/prim_ops/et_copy_index.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/operator_registry.h>

using OpArrayRef = ::torch::executor::ArrayRef<::torch::executor::Operator>;
using torch::executor::function::et_copy_index;

namespace torch {
namespace executor {
namespace function {

namespace {

static Operator prim_ops[] = {
    // aten::sym_size.int(Tensor self, int dim) -> SymInt
    Operator(
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
    Operator(
        "aten::sym_numel",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& self = *stack[0];
          EValue& out = *stack[1];
          exec_aten::Tensor self_tensor = self.to<exec_aten::Tensor>();
          int64_t numel = self_tensor.numel();
          out = EValue(numel);
        }),
    // executorch_prim::add.int(int, int) -> int
    Operator(
        "executorch_prim::add.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() + b.toInt());
        }),

    // executorch_prim::sub.int(int, int) -> int
    Operator(
        "executorch_prim::sub.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() - b.toInt());
        }),

    // executorch_prim::mul.int(int, int) -> int
    Operator(
        "executorch_prim::mul.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() * b.toInt());
        }),

    // executorch_prim::floordiv.int(int, int) -> int
    Operator(
        "executorch_prim::floordiv.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() / b.toInt());
        }),

    // executorch_prim::eq.int(int, int) -> bool
    Operator(
        "executorch_prim::eq.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() == b.toInt());
        }),

    // executorch_prim::gt.int(int, int) -> bool
    Operator(
        "executorch_prim::gt.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() > b.toInt());
        }),

    // executorch_prim::lt.int(int, int) -> bool
    Operator(
        "executorch_prim::lt.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() < b.toInt());
        }),

    // executorch_prim::ge.int(int, int) -> bool
    Operator(
        "executorch_prim::ge.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() >= b.toInt());
        }),

    // executorch_prim::le.int(int, int) -> bool
    Operator(
        "executorch_prim::le.int",
        [](RuntimeContext& context, EValue** stack) {
          (void)context;
          EValue& a = *stack[0];
          EValue& b = *stack[1];
          EValue& out = *stack[2];
          out = EValue(a.toInt() <= b.toInt());
        }),

    // executorch_prim::et_copy_index.tensor(tensor, tensor) -> tensor
    Operator("executorch_prim::et_copy_index.tensor", &et_copy_index),

};

static OpArrayRef op_array_ref(
    prim_ops,
    prim_ops + sizeof(prim_ops) / sizeof(Operator));

// Return value not used. Keep the static variable assignment to register
// operators in static initialization time.
static auto success_with_op_reg = register_operators(op_array_ref);

} // namespace
} // namespace function
} // namespace executor
} // namespace torch
