# Symbolic TOSA Shape Lowering

This note explains how `resolve_view_copy_inferred_dim_pass.py`,
`symbolic_to_tosa_shape_pass.py`, and `symbolic_materialization_helper.py`
work together.

## Goal

PyTorch export represents dynamic shape values as scalar symbolic values such as
`SymInt`s and FX nodes like `aten.sym_size.int`. TOSA shape ops represent shape
values as one-dimensional shape tensors, modeled in fake execution as
`SymInt[]`.

The pass converts scalar symbolic shape computations and args containg `SymInt[]`
into TOSA shape operands.

For example, this FX-style shape computation:

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s0, 3, 4]"):
        sym_size = torch.ops.aten.sym_size.int(x, 0)
        add = operator.add(sym_size, 1)
        view = torch.ops.aten.view.default(x, [add, 12])
        return (view,)
```

is rewritten conceptually into:

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s0, 3, 4]"):
        dim = executorch_exir_dialects_edge__ops_backend_tosa_DIM_default(
            x,
            axis=0,
        )
        one = executorch_exir_dialects_edge__ops_backend_tosa_CONST_SHAPE_default(
            [1]
        )
        add = executorch_exir_dialects_edge__ops_backend_tosa_ADD_SHAPE_default(
            dim,
            one,
        )
        twelve = executorch_exir_dialects_edge__ops_backend_tosa_CONST_SHAPE_default(
            [12]
        )
        shape = executorch_exir_dialects_edge__ops_backend_tosa_CONCAT_SHAPE_default(
            [add, twelve]
        )
        view = torch.ops.aten.view.default(x, shape)
        return (view,)
```

The important representation change is:

```text
scalar symbolic value  ->  TOSA shape operand
s0                     ->  [s0]
s0 + 1                 ->  [s0 + 1]
[scalar pieces]        ->  CONCAT_SHAPE([...])
```

## `ResolveViewCopyInferredDimPass`

`ResolveViewCopyInferredDimPass` runs before `SymbolicToTosaShapesPass` and
normalizes view shapes that use one inferred dimension. For example:

```text
view_copy(x, [sym_size(x, 0), -1])
```

The `-1` is replaced with the inferred output dimension from fake tensor
metadata. If that dimension is dynamic, the pass uses
`Graph.materialize_symints(...)` to lift the raw `torch.SymInt` expression into
ordinary FX symbolic IR rooted at existing producers, such as
`aten.sym_size.int` and `operator.mul`. After this pass,
`SymbolicToTosaShapesPass` sees an explicit shape expression rather than a
view-specific `-1` convention.

## `SymbolicToTosaShapesPass`

`SymbolicToTosaShapesPass` is an `ArmPass` that intercepts symbolic shape nodes
while the FX graph is being transformed.

It handles three cases.

### 1. `aten.sym_size.int`

`aten.sym_size.int(x, dim)` reads one dynamic dimension from a tensor. The pass
lowers it to TOSA `DIM`:

```text
aten.sym_size.int(x, 0)  ->  tosa.DIM(x, axis=0)
```

The original scalar `SymInt` result becomes a one-element TOSA shape operand.

Example:

```python
# before
sym_size = torch.ops.aten.sym_size.int(x, 0)

# after
dim = tosa.DIM.default(x, axis=0)
```

`DIM` is special because the axis stays as a keyword argument. The helper does
not materialize the axis as a TOSA shape operand.

### 2. Symbolic Arithmetic

The pass lowers supported Python symbolic arithmetic operators to TOSA shape
arithmetic ops:

```text
operator.add       ->  tosa.ADD_SHAPE
operator.sub       ->  tosa.SUB_SHAPE
operator.mul       ->  tosa.MUL_SHAPE
operator.mod       ->  tosa.MOD_SHAPE
operator.floordiv  ->  tosa.DIV_FLOOR_SHAPE
```

For example:

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s0, s1, 4]"):
        s0 = torch.ops.aten.sym_size.int(x, 0)
        s1 = torch.ops.aten.sym_size.int(x, 1)
        product = operator.mul(s0, s1)
        view = torch.ops.aten.view.default(x, [product, 4])
        return (view,)
```

becomes conceptually:

```python
dim_0 = tosa.DIM.default(x, axis=0)
dim_1 = tosa.DIM.default(x, axis=1)
product = tosa.MUL_SHAPE.default(dim_0, dim_1)
four = tosa.CONST_SHAPE.default([4])
shape = tosa.CONCAT_SHAPE.default([product, four])
view = torch.ops.aten.view.default(x, shape)
```

Only symbolic arithmetic with at least one TOSA shape operand is lowered.
Ordinary scalar or Python arithmetic is forwarded to the base pass. Unsupported
arithmetic with a shape operand raises `NotImplementedError`.

Nested expressions are lowered one symbolic operation at a time. For example,
this exported FX form:

```python
s0 = torch.ops.aten.sym_size.int(x, 0)
s1 = torch.ops.aten.sym_size.int(x, 1)
sub = operator.sub(s0, 1)
add = operator.add(sub, s1)
floordiv = operator.floordiv(add, 2)
```

is represented as a chain like:

```text
DIM(x, axis=0)
CONST_SHAPE([1])
SUB_SHAPE(dim_0, one)
DIM(x, axis=1)
ADD_SHAPE(sub, dim_1)
CONST_SHAPE([2])
DIV_FLOOR_SHAPE(add, two)
```

### 3. Shape Lists Used By Operators

Some operators receive shape-like Python lists or tuples. For example `view`
can receive a shape argument like:

```python
sym_size = torch.ops.aten.sym_size.int(x, 0)
view = torch.ops.aten.view.default(x, [sym_size, 12])
```

After `sym_size` has been lowered to a TOSA shape `ProxyValue`, this list is a
mixed Python container:

```text
[ProxyValue(DIM), 12]
```

The pass detects list or tuple arguments containing at least one TOSA shape
`ProxyValue`, including nested shape proxies. It then asks the helper to turn
the whole container into a single TOSA shape operand:

```text
[DIM(x, axis=0), 12]
  -> CONST_SHAPE([12])
  -> CONCAT_SHAPE([DIM, CONST_12])
```

The rewritten operator receives the `CONCAT_SHAPE` result instead of the Python
list.

## Shape-Marked Nodes

The pass uses `meta_has_shape_mark(...)` to distinguish TOSA shape values from
ordinary tensor values.

If an operator result is already marked as a TOSA shape value, the pass forwards
it unchanged:

```python
if meta_has_shape_mark(meta.data):
    return super().call_operator(op, args, kwargs, meta, updated)
```

This avoids recursively trying to lower TOSA shape ops that are already in the
right representation.

## `SymbolMaterializationHelpers`

`SymbolMaterializationHelpers` owns the TOSA-specific materialization logic and
cache. The pass decides *when* something should become a TOSA shape value; the
helper decides *how* to build or reuse the required TOSA shape nodes.

The helper accepts valid shape pieces made from:

- `ProxyValue` objects that already produce TOSA shape values;
- Python `int` constants;
- nested Python `list` or `tuple` containers of those values.

It is not responsible for lowering arbitrary raw `torch.SymInt` values. Those
should already have FX producers by the time this helper is used. For inferred
`view_copy` dimensions, `ResolveViewCopyInferredDimPass` creates those FX
producers before TOSA shape materialization runs.

## `materialize_arglist(...)`

`materialize_arglist(shape_arg, meta)` converts a Python shape container into a
single TOSA shape operand.

For a single existing shape proxy, it reuses the proxy:

```text
[DIM(x, axis=0)] -> DIM(x, axis=0)
```

For an integer, it creates or reuses `CONST_SHAPE`:

```text
[7] -> CONST_SHAPE([7])
```

For multiple pieces, it flattens nested lists/tuples, materializes each piece,
and creates `CONCAT_SHAPE`:

```text
[[DIM(x, axis=0)], [2, 3]]
  -> DIM(x, axis=0)
  -> CONST_SHAPE([2])
  -> CONST_SHAPE([3])
  -> CONCAT_SHAPE([DIM, CONST_2, CONST_3])
```

If the same integer constant is needed again, the cached `CONST_SHAPE` node is
reused:

```text
materialize_arglist([5])        -> CONST_SHAPE([5])
materialize_arglist([dim, 5])   -> CONCAT_SHAPE([dim, cached_CONST_5])
```

## `materialize_shape_op(...)`

`materialize_shape_op(target, args, kwargs, meta)` creates or reuses a TOSA
shape op result.

For non-`DIM` shape ops, each argument is first converted into a TOSA shape
operand with `materialize_arglist(...)`:

```text
ADD_SHAPE(dim, 1)
  -> ADD_SHAPE(dim, CONST_SHAPE([1]))
```

For `DIM`, the input tensor is passed through directly and the axis remains in
`kwargs`:

```text
DIM(x, axis=1)
```

The helper caches shape-op outputs by `str(meta.data["val"])`. If the same
symbolic output shape is requested again, the existing `ProxyValue` is reused.

## Cache Behavior

The helper has one cache:

```python
self._shape_to_proxyval: dict[str, ProxyValue]
```

It stores:

- integer constants under `str(value)`, for example `"1"`;
- shape-op results under `str(meta.data["val"])`, for example `"[s0 + 1]"`.

This cache is local to one helper/pass instance. It is used to avoid duplicating
shape producers while lowering one graph.

## Metadata

Shape nodes are created through `ArmPass.call_shape_operator(...)`. The helper
passes through the `NodeMetadata` it received, so metadata such as `val`, debug
handles, and TOSA shape markers can be attached by the pass infrastructure.

The tests assert that metadata is preserved on emitted `CONST_SHAPE` and
`CONCAT_SHAPE` nodes.

## Canonicalizing Mixed Shape Containers

Arm/TOSA lowering code may temporarily build mixed Python containers of integer
constants, shape-producing `ProxyValue`s, and nested lists or tuples. That is a
normal intermediate form while constructing shape arguments.

`InsertDynamicPaddingPass` is the main current producer of this form. For a
dynamic 2D convolution or pool, it rewrites implicit spatial padding into an
explicit `PAD` op and resets the original op padding to zeros. The new `PAD`
argument is intentionally left as a flattened Python list:

```text
[0, 0, *spatial_padding, 0, 0]
```

where `spatial_padding` may contain shape-producing proxies that came from
`Graph.materialize_symints(...)`. Before this container reaches serialization,
`ResolveViewCopyInferredDimPass` and `SymbolicToTosaShapesPass` run after
`InsertDynamicPaddingPass`. The symbolic pass detects the shape-marked values
in the list and asks the helper to flatten and materialize
the pieces. Conceptually, the 2D padding list becomes a single shape operand such as:

```text
CONCAT_SHAPE([
  CONST_SHAPE([0]),
  CONST_SHAPE([0]),
  spatial_pad_0,
  spatial_pad_1,
  spatial_pad_2,
  spatial_pad_3,
  CONST_SHAPE([0]),
  CONST_SHAPE([0]),
])
```