# Export IR Specification

Export IR is an intermediate representation (IR) for the result of
`torch.export`. To read more on the details of Export IR, please read this
[document](https://pytorch.org/docs/main/export.ir_spec.html).

The Exported IR is a specification that consists of the following parts:

1. A definition of computation graph model.
2. Set of operators allowed in the graph.

A **dialect** is an Exported IR graph composed with the operations defined
below, but with additional properties (such as restrictions on operator set or
metadata) that are meant for a specific purpose.

The EXIR dialects that currently exist are:

* [ATen Dialect](#aten-dialect)
* [Edge Dialect](#edge-dialect)
* [Backend Dialect](#backend-dialect)

These dialects represent stages that a captured program goes through from
program capture to conversion into an executable format. For example, the
Executorch compilation process starts from a Python program capture into ATen
Dialect, then ATen Dialect is converted to Edge Dialect, Edge to Backend, and
finally to a binary format for execution.

## ATen Dialect

ATen dialect will be used as the entry point of the ExecuTorch compilation
pipeline, it is the first time an eager mode Pytorch program becomes an Exported
IR graph. At this stage, functionalization is performed, removing any tensor
aliases and mutations, and allowing for more flexible graph transformations to
be made. Additionally, all tensors are converted to continuous format.

The goal of this dialect is to capture users' programs as faithfully as possible
(while remaining valid Exported IR). Registered custom operators that user has called
in eager mode will preserve as-is in ATen dialect. However, we should refrain
from adding custom ops in the graph via passes.

For now, the function of ATen dialect is to further lower to Edge dialect.
However, in the future we can see this one as the common integration point for
other export use cases.

### ATen Dialect Properties

An ATen dialect graph is a valid Export IR graph with the following additional
properties:

1. All operators in `call_function` nodes are either ATen operators (in the
  `torch.ops.aten` namespace, higher order operators (like control flow
  operators), or a registered custom operator. A registered custom operator is
  an operator registered into the current Pytorch eager mode runtime, usually
  with `TORCH_LIBRARY` call (implies schema). Details for how to register a
  custom operator can be found
  [here](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.3rgxk3v387wl).
2. Every operator must also have a meta kernel. A meta kernel is a
  function that, given the shapes of the input tensors, can return the shape of
  output tensor. Details on how to write a meta kernel can be found
  [here](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0).
3. Input value type must be “Pytree-able”. As a consequence, the output
  types are also Pytree-able because all operators output are pytree-able.
4. Ops of ATen dialect can choose to work Dynamic dtypes, implicit type
  promotions and implicit broadcasting of tensors.
5. All tensors memory formats are in `torch.contiguous_format`.

### ATen Operator Definition

The operator set definition can be found [here](./ir-ops-set-definition.md).

## Edge Dialect

This dialect is meant to introduce specializations that are useful for Edge
devices but not necessarily for general (server) export. However, we still
withhold specializing further to each different hardware. In other words, we
don’t want to introduce any new hardware dependent concepts or data; besides
those already present in users’ original python program.

### Edge Dialect Properties

An Edge dialect graph is a valid Export IR graph with the following additional
properties:

1. All operators in OpCall nodes are either from a predefined operator set,
   called **“Edge Operators”**, or a registered custom operator. An Edge operator is a
   ATen operator with dtype specialization. This allows users to register
   kernels that only work for certain dtypes to reduce binary size.
2. Input and output of the graph, and as well as to every node, cannot be Scalar. I.e.
   All scalar types (such as float, int) are converted to Tensor.

### Using the Edge Dialect

The Edge dialect is represented with `exir.EdgeProgramManager` Python class in
memory. This contains one or multiple `torch.export.ExportedProgram`s which
contain the graph representation of a method.

```python
import torch
from executorch import exir

class MyModule(torch.nn.Module):
    ...

a = MyModule()
tracing_inputs = (torch.rand(2, 2),)
aten_dialect_program = torch.export.export(a, tracing_inputs)
edge_dialect_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)
print(edge_dialect_program.exported_program)
```

At this point, user defined graph transformation can be run through
`edge_dialect_program.transform(pass)`. Order matters. Note: If the custom pass
is touching `node.target`, be aware that all of the `node.target` at this stage
are "Edge ops" (more details below) and not torch ops like in the ATen dialect.
A tutorial on pass writing can be found
[here](./compiler-custom-compiler-passes.md). After all these passes are
executed, `to_edge()` will make sure the graph is still valid.

### Edge Operators

As mentioned before, an edge operator is an ATen core operator with type
specialization. This means an instance of the edge operator contains a set of
dtype constraints, that describe all the tensor dtypes supported by both the
ExecuTorch runtime and their ATen kernels. These dtype constraints are expressed
in a DSL defined in
[edge.yaml](https://github.com/pytorch/executorch/blob/main/exir/dialects/edge/edge.yaml).
Here's an example of the dtype constraints:

```
- func: sigmoid
  namespace: edge
  inherits: aten::sigmoid
  type_alias:
    T0: [Bool, Byte, Char, Int, Long, Short]
    T1: [Double, Float]
    T2: [Float]
  type_constraint:
  - self: T0
    __ret_0: T2
  - self: T1
    __ret_0: T1
```
This is saying if `self` tensor is one of the type `Bool, Byte, Char, Int, Long, Short`, then the return tensor would be `Float`. If `self` is one of `Double, Float`, the return tensor will be the same dtype.

After these dtype constraints are collected and documented in edge.yaml, EXIR
consumes the file, and loads the constraints into EXIR Edge operators. This
makes it convenient for developers to learn the supported dtypes of any argument
in the Edge op schema. For example we can do:


```python
from executorch.exir.dialects._ops import ops as exir_ops # import dialects ops
sigmoid = exir_ops.edge.aten.sigmoid.default
print(sigmoid._schema)
# aten::sigmoid(Tensor self) -> Tensor
self_arg = sigmoid._schema.arguments[0]
_return = sigmoid._schema.returns[0]

print(self_arg.allowed_types)
# {torch.float32, torch.int8, torch.float64, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}

print(_return.allowed_types)
# {torch.float32, torch.float64}
```

These constraints are helpful for someone who wants to write a custom kernel for this operator. Also inside EXIR, we offer a validator to check if the graph is still complying with these dtype constraints, after custom transformations.

### Op Set (WIP)

Check out
[edge.yaml](https://github.com/pytorch/executorch/blob/main/exir/dialects/edge/edge.yaml)
for the complete list of operators having dtype constraints specified. We are
gradually expanding this operator set and targeting to provide dtype constraints
for all core ATen ops.

## Backend Dialect

Backend dialect is the name we gave to the `ExportedProgram` in Edge dialect,
after optional **target specific** passes. The difference between backend
dialect and edge dialect is that backend dialect is target-aware and may contain
operators or submodules that are only meaningful to the target backend. Backend
specific operators are new components we may see in a backend dialect, comparing
with Edge dialect. They are a set of operators for the target backend.

Another property to notice is that the memory formats of the tensor can be any
format (this is subject to change in the near future when we introduce dim order
to backend dialect).

This dialect allows introduction of operators that do not conform to the schema
defined in the canonical ATen operator set, and are not showing up in any of the
dialects above (ATen dialect and edge dialect). Consider to use backend
operators if your use case satisfies one or more of the following criteria:

1. Your backend provides a library that optimizes a certain operator that is
  equivalent to a subgraph. E.g., linear_relu (equivalent to linear + relu) that
  can be executed faster on a certain backend.
2. There's a need to retrace the graph module after it is already lowered to a
  backend. When we retrace, backend operators can transform back to the original
  subgraph (in ATen dialect) where normal custom op doesn't take care of that.
3. Your backend specific operator doesn't have a generic CPU kernel but only a
  kernel for a certain backend. Using backend operator can workaround this issue
  by using the original subgraph as default kernel and keep the graph module
  runnable.

### Running Backend Passes

To lower edge ops to backend ops, a pass will perform pattern matching to
identify the edge ops of interest in the graph, and then replace them with
equivalent backend operators. There are two APIs to register such passes:

* `transform()`. An API on `ExportProgram` that allows users to provide custom
  passes. Note that this is not guarded by any validator so the soundness of the
  program is not guaranteed.
* [`ExecutorchBackendConfig.passes`](https://github.com/pytorch/executorch/blob/main/exir/capture/_config.py#L40).
  If added here, the pass will be part of the lowering process from backend
  dialect to `ExecutorchProgram`.

Example: One such pass is `QuantFusion`. This pass takes a "canonical
quantization pattern", that is, "dequant - some_op - quant", and fusees this
pattern into a single operator that is backend specific, that is,
`quantized_decomposed::some_op`. You can find more details
[here](./quantization-custom-quantization.md). Another simpler example is
[here](https://github.com/pytorch/executorch/blob/main/exir/passes/replace_edge_with_backend_pass.py#L20)
where we replace sym_size operators with ones that are understood by ExecuTorch.

### Backend Dialect Operators

We provide a decorator `bind_pattern_to_op` to help users easily register their
backend operators into Export IR. This decorator takes:
their backend operators into Export IR. This decorator takes:
* a `torch.Library` object, that indicates which library or namespace this backend
  operator belongs to.
* a name or schema. If we already defined the schema of the backend operator in
  the `torch.Library` object, only a name is needed. Otherwise we can register
  the schema if a schema string is being passed in.

This decorator should be added to the pattern we are trying to match (and then
lower to this backend op) on the edge dialect. This way we are registering this
pattern as a `CompositeImplicitAutograd` kernel for this backend operator.

Then the operator can be accessed/used from the passes. The `CompositeImplicitAutograd` kernel makes sure:
1. No need for the user to write a (CPU) runnable kernel
2. Ensures the retracability of `ExportProgram`. Once retraced, the backend
  operator will be decomposed into the ATen ops used in the pattern.

Unlike edge dialect where we have a well defined op set, for backend dialect,
since it is target-aware we will be allowing user to use our API to register
target-aware ops and they will be grouped by namespaces. Here are some examples:
`executorch_prims` are ops that are used by ExecuTorch runtime to perform
operation on `SymInt`s. `quantized_decomposed` are ops that fuses edge operators
for quantization purpose and are meaningful to targets that support
quantization.

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
