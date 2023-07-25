# Backend Dialect


## Properties
Backend dialect is the name we gave to the `ExportedProgram` in Edge dialect, after optional target specific passes. The difference between backend dialect and edge dialect is that backend dialect is target-aware and may contain operators or submodules that are only meaningful to the target backend.

Two possible new components we may see in a backend dialect, comparing with Edge dialect, are:
1. Backend specific operators. Backend specific ops are set of operators the that user specifies, and the target backend will need to implement for the final program to run in that backend
2. Lowered module from delegate. See details in [delegate tutorial](../tutorials/backend_delegate.md), this doc is focusing more on the backend ops.

Another property to notice is that the memory formats of the tensor can be any format (this is subject to change in the near future when we introduce dim order to backend dialect).


## Intent

This dialect allows introduction of operators that do not conform to the schema defined in the canonical ATen operator set, and are not showing up in any of the dialects above (ATen dialect and edge dialect). Consider to use backend operators if your use case satisfies one or more of the following criteria:

1. Your backend provides a library that optimizes a certain operator that is equivalent to a subgraph. E.g., linear_relu (equivalent to linear + relu) that can be executed faster on a certain backend.
2. There's a need to retrace the graph module after it is already lowered to a backend. When we retrace, backend operators can transform back to the original subgraph (in ATen dialect) where normal custom op doesn't take care of that.
3. Your backend specific operator doesn't have a generic CPU kernel but only a kernel for a certain backend. Using backend operator can workaround this issue by using the original subgraph as default kernel and keep the graph module runnable.


## How to use

To lower edge ops to backend ops, a pass will perform pattern matching to identify the edge ops of interest in the graph, and then replace them with equivalent backend operators. There are two APIs to register such passes:

* `transform()`. An API on `ExportProgram` that allows users to provide custom passes. Note that this is not guarded by any validator so the soundness of the program is not guaranteed.
* [`ExecutorchBackendConfig.passes`](https://github.com/pytorch/executorch/blob/main/exir/capture/_config.py#L40). If added here, the pass will be part of the lowering process from backend dialect to `ExecutorchProgram`.

Example: one of such passes is `QuantFusion`. This pass takes a "canonical quantization pattern", ie. "dequant - some_op - quant" and fuse this pattern into a single operator that is backend specific, i.e. `quantized_decomposed::some_op`. You can find more details [here](../tutorials/short_term_quantization_flow.md). Another simpler example is [here](https://github.com/pytorch/executorch/blob/main/exir/passes/replace_edge_with_backend_pass.py#L20) where we replace sym_size operators to the ones that are understood by Executorch.

## API

We provide a decorator `bind_pattern_to_op` to help users to easily register their backend operators into EXIR. This decorator takes:
* a `torch.Library` object, it indicates which library or namespace this backend operator belongs to.
* a name or schema. If we already defined the schema of the backend operator in the `torch.Library` object, only a name is needed. Otherwise we can register the schema if a schema string is being passed in.

This decorator should be added to the pattern we are trying to match (and then lower to this backend op) on edge dialect. This way we are registering this pattern as a `CompositeImplicitAutograd` kernel for this backend operator.

Then the operator can be accessed/used from the passes. The `CompositeImplicitAutograd` kernel makes sure:
1. No need for the user to write a (CPU) runnable kernel
2. Ensures the retracability of `ExportProgram`. Once retraced, the backend operator will be decomposed into the ATen ops used in the pattern.

## Op Set

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
