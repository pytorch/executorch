# Edge dialect

Edge dialect is a dialect of EXIR satifying the following properties:

## Properties

1. All operators in OpCall nodes are either from a predefined operator set,
   called **“Edge Operators”**, or a registered custom operator. An Edge operator is a
   ATen operator with dtype specialization.
2. Input and output of the graph, and as well as to every node, cannot be Scalar. I.e.
   All scalar types (such as float, int) are converted to Tensor.

## Intent

This dialect is meant to introduce specializations that are useful for Edge
devices but not necessarily for general (server) export.
However, we still withhold specializing further to each different hardware.
In other words, we don’t want to introduce any new hardware dependent concepts or data;
besides those already present in users’ original python program.

## How to use

A GraphModule in EXIR edge dialect is represented with `torch.fx.GraphModule` Python class
in memory. To obtain such a class, one start with a `torch.nn.Module`:

```python
import torch
from executorch import exir

class MyModule(torch.nn.Module):
    ...
a = MyModule()
tracing_inputs = (torch.rand(2, 2),)
edge_dialect_module = exir.capture(a, tracing_inputs).to_edge().module
```

As we can see if no input is provided to `to_edge()` API, the lowering process from ATen dialect to edge dialect should be invisible to the user. However we provide some knobs for advanced usage:

* `EdgeCompileConfig.passes`
User defined graph transformation goes in here. Order matters. Note: if the custom pass is touching `node.target`, be aware that all of the `node.target` at this stage are "Edge ops" (more details below) and not torch ops like in ATen dialect. Tutorial on pass writing can be found [here](../tutorials/passes). After all these passes are executed, `to_edge()` will make sure the graph is still valid.

* `EdgeCompileConfig._check_ir_validity`
Default value is true. If set to false, graph validaity check will be turned off. Turn this flag off with caution, since the graph may become invalid after `to_edge()`.

## Edge Operator

As mentioned before, an edge operator is an ATen core operator with type specialization. This means the instance of edge operator contains a set of dtype constraints, to describe all the tensor dtypes supported by both Executorch runtime and their ATen kernels. These dtype constraints are expressed in a DSL defined in [edge.yaml](https://github.com/pytorch/executorch/blob/main/exir/dialects/edge/edge.yaml). Here's an example of the dtype constraints:

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

After these dtype constraints are collected and documented in edge.yaml, EXIR consumes it, load them into EXIR Edge operators. This is convenient for developers to learn the supported dtypes of any argument in Edge op schema. For example we can do:

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

## Op Set (WIP)

Check out [edge.yaml](https://github.com/pytorch/executorch/blob/main/exir/dialects/edge/edge.yaml) for the complete list of operators having dtype constraints specified. We are gradually expanding this operator set and targeting to provide dtype constraints for all core ATen ops.
