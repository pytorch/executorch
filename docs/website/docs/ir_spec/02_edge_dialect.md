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

## Edge Operator

As mentioned before, an edge operator is an ATen core operator with type specialization. This means the instance of edge operator contains a set of dtype constraints, for tensor dtypes supported by both Executorch runtime and their ATen kernels. Inside the compiler we will validate the Edge dialect graph module against these constraints, to make sure the graph module is supported on Executorch. These dtype constraints are expressed in a DSL defined in [edge.yaml](https://www.internalfb.com/code/fbsource/fbcode/executorch/exir/dialects/edge/edge.yaml). Here's an example of the dtype constraints:

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

## Op Set

Check out [edge.yaml](https://www.internalfb.com/code/fbsource/fbcode/executorch/exir/dialects/edge/edge.yaml) for the complete list of operators having dtype constraints specified. We are gradually expanding this operator set and targeting to provide dtype constraints for all ATen core ops.
Then `edge_dialect_module` is a GraphModule of edge dialect.
