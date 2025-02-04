# Backend Dialect
## Overview

_Backend dialect_ is a special variant of [edge dialect](./ir-exir.md), because it contains backend specific nodes and metadata, after backend specific graph transformations. Backend dialect is an optional stage, only needed if we want to introduce backend-awareness into the graph. More specifically, a graph in backend dialect may contain operators or delegated lowered modules (see [delegate doc](./compiler-delegate-and-partitioner.md)) that are only meaningful to the target backend. One use case is that if we want to fuse operators into a single operator, for example, fusing consecutive addmm + relu to a single operator addmm_relu, we can do that here.

This document describes how to introduce backend specific operators.

Difference between custom ops and backend specific ops: while custom ops are showing up in eager mode, ATen dialect, and edge dialect, backend-specific ops are only being introduced by passes happening after edge dialect.


## When to use

This dialect allows the introduction of operators that do not conform to the schema defined in the canonical ATen operator set, and which do not appear in any of the dialects above (ATen dialect and edge dialect). Consider using backend operators if your use case satisfies one or more of the following criteria:



* Your backend provides a library that optimizes a certain operator that is equivalent to a subgraph. For example, `linear_relu` (equivalent to linear + relu) that can be executed faster on a certain backend.
* There's a need to retrace the graph module after it is already lowered to a backend. When we retrace, backend operators can transform back to the original subgraph (in ATen dialect) where normal custom op doesn't take care of that.
* Your backend-specific operator doesn't have a generic CPU kernel but only a kernel for a certain backend. Using a backend operator can workaround this issue by using the original subgraph as default kernel and keeping the graph module runnable.
* Alternatively, you can use delegate if you are concerned it might be an overkill and just want something more lightweight and only requires Python code at the compiler stage.


## APIs

For an operator/subgraph replacement, the common flow is:



1. Register an operator that has the same input and output as the subgraph. This operator won’t have the target-specific implementations (also, doesn’t need to in the compilation stage), but it needs to give the same result as the subgraph.
2. Create a pattern that allows the compiler to find the subgraph and substitute it with the replacement.
3. Write a pass to replace the subgraph with the new operator.

In order to facilitate the process, we provide an API to help reduce the effort for ExecuTorch users to do these steps.


### Pass Infra Entry Point

To lower edge ops to backend ops, a pass will perform pattern matching to identify the edge ops of interest in the graph, and then replace them with equivalent backend operators. There are two APIs to register such passes:



* `transform()`. An API on ExportProgram that allows users to provide custom passes. Note that this is not guarded by any validator so the soundness of the program is not guaranteed.
* [ExecutorchBackendConfig.passes](https://github.com/pytorch/executorch/blob/main/exir/capture/_config.py#L40). If added here, the pass will be part of the lowering process from backend dialect to ExecutorchProgram.

Example: one such pass is QuantFusion. This pass takes a "canonical quantization pattern", ie. "dequant - some_op - quant" and fuses this pattern into a single operator that is backend specific, i.e. `quantized_decomposed::some_op`. Another simpler example is [here](https://github.com/pytorch/executorch/blob/main/exir/passes/replace_edge_with_backend_pass.py#L20) where we replace `sym_size` operators to the ones that are understood by ExecuTorch


### Pattern Binding Decorator

We provide a decorator `bind_pattern_to_op` to help users easily register their backend operators into EXIR. This decorator takes:



* a `torch.Library` object, it indicates which library or namespace this backend operator belongs to.
* a name or schema. If we already defined the schema of the backend operator in the `torch.Library` object, only a name is needed. Otherwise we can register the schema if a schema string is being passed in.

This decorator should be added to the pattern we are trying to match (and then lower to this backend op) on edge dialect. This way we are registering this pattern as a `CompositeImplicitAutograd` kernel for this backend operator.

Then the operator can be accessed/used from the passes. The `CompositeImplicitAutograd` kernel makes sure:



1. No need for the user to write a (CPU) runnable kernel.
2. Ensures the retrace-ability of `ExportProgram`. Once retraced, the backend operator will be decomposed into the ATen ops used in the pattern.


## Example

Let’s assume a simple program that contains both add and relu operators:
```python
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = x + y
    return torch.ops.aten.relu.default(z)
```
After lowering to edge dialect it becomes:
```
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    %aten_relu_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.relu.default](args = (%aten_add_tensor,), kwargs = {})
    return (aten_relu_default,)
```
Now I want to write a pass to merge `add` and `relu` into `add_relu`, the first step is to write a pattern:
```python
# In the pattern, we can use edge ops and ATen ops interchangably
def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.ops.aten.add.Tensor(x, y)
    out = torch.ops.aten.relu.default(z)
    return out
```
Then we need to create an operator library from the fused operator namespace, then use the decorator on our pattern:

```python
lib = Library("foo_namespace", "DEF")

@bind_pattern_to_op(lib, "add_relu(Tensor self, Tensor other) -> Tensor")
def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.ops.aten.add.Tensor(x, y)
        out = torch.ops.aten.relu.default(z)
        return out
```
This way we are registering the pattern as a kernel to `add_relu` and it is ready to be used in a pass. A simple pass looks like this:
```python
class AddReluFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        # decorator registers this pattern as a CompositeExplicitAutograd kernel, since there's no kernel registered before.
        @bind_pattern_to_op(lib, "add_relu")
        def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = torch.ops.aten.add.Tensor(x, y)
            out = torch.ops.aten.relu.default(z)
            return out

        def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.ops.foo_namespace.add_relu.default(x, y)

        subgraph_rewriter.replace_pattern(
            graph_module,
            _trace_and_lower_to_edge_ops(pattern),
            _trace_and_lower_to_edge_ops(replacement),
        )
        return PassResult(graph_module, True)
```
The result graph looks like this:
```
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %foo_namespace_add_relu_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.foo_namespace.add_relu.default](args = (%arg0_1, %arg1_1), kwargs = {})
    return (foo_namespace_add_relu_default,)
```
### Op Set

There are the backend operators currently using `bind_pattern_to_op` API.

* `executorch_prims::add.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.add
  * backend: executor
* `executorch_prims::mul.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.mul
  * backend: executor
* `executorch_prims::sub.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.sub
  * backend: executor
* `executorch_prims::floordiv.int(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.floordiv
  * backend: executor
* `executorch_prims::truediv.int(Scalar a, Scalar b) -> Scalar`
  * pattern: builtin.div
  * backend: executor
* `executorch_prims::sym_float.Scalar(Scalar a) -> Scalar`
  * pattern: builtin.float
  * backend: executor
* `executorch_prims::gt.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.gt
  * backend: executor
* `executorch_prims::lt.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.lt
  * backend: executor
* `executorch_prims::ge.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.ge
  * backend: executor
* `executorch_prims::le.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.le
  * backend: executor
* `executorch_prims::eq.int(SymInt a, SymInt b) -> bool`
  * pattern: builtin.eq
  * backend: executor
* `executorch_prims::mod.Scalar(SymInt a, SymInt b) -> SymInt`
  * pattern: builtin.divmod
  * backend: executor
* `executorch_prims::neg.Scalar(Scalar a) -> Scalar`
  * pattern: operator.ne
  * backend: executor
* `quantized_decomposed::embedding_byte(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor`
  * pattern: [source](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py)
  * backend: quantization
* `quantized_decomposed::add(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc`
  * pattern: [source](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py)
  * backend: quantization
* `quantized_decomposed::add.scalar(Tensor qa, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, ScalarType a_dtype, Scalar b, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max, ScalarType out_dtype) -> Tensor`
  * pattern: [source](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py)
  * backend: quantization
* `quantized_decomposed::add_relu(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc`
  * pattern: [source](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py)
  * backend: quantization
