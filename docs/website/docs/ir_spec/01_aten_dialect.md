# ATen Dialect


## Properties:

An ATen dialect graph is a valid EXIR graph with the following additional properties:


1. All operators in OpCall nodes are either from a predefined operator set,
  called ["Core ATen Operators”](https://pytorch.org/docs/stable/ir.html), or a
  registered custom operator. A registered custom operator is an operator
  registered into the current Pytorch eager mode runtime, usually with
  TORCH_LIBRARY call (implies schema).
2. Every ATen operator must also have a  meta kernel. A meta kernel is a
  function that, given the shapes of the input tensors, can return the shape of
  output tensor.
3. Input value type must be “Pytree-able[See 2]”. As a consequence, the output
  types are also Pytree-able because all the operators output are pytree-able.
4. Ops of Aten dialect can choose to work Dynamic dtypes, implicit type
  promotions and implicit broadcasting of tensors.
5. All tensors memory formats are in [**Pytorch Default Dims Format:**](./00_exir.md#memory-formats)
  i.e. torch.contiguous_format.

<table>
  <tr>
   <td>
Op Set
   </td>
   <td>Canonical ATen
   </td>
   <td>Custom Op
   </td>
   <td><del>All ATen Ops</del>
   </td>
  </tr>
  <tr>
   <td>ATen
   </td>
   <td>Allowed
   </td>
   <td>Allowed, must have meta kernel
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Edge
   </td>
   <td>Aten + Type specializations
   </td>
   <td>Allowed
   </td>
   <td>
   </td>
  </tr>
</table>



## Intent

This section describes what we envision ATen dialect is used for.

ATen dialect will be used as the entry point of the executorch compilation
pipeline, it is the first time an eager mode Pytorch program becomes an EXIR
graph. At this stage, functionalization is performed, so all the tensor aliases
are made a copy of. Therefore, all tensors are converted to continuous format.

The goal of this dialect is to capture users' programs as faithfully as possible
(while remaining valid EXIR). Registered Custom Operators that user has called
in eager mode will preserve as-is in ATen dialect. However, we should refrain
from adding custom ops in the graph via passes.

For now, the function of ATen dialect is to further lower to edge dialect.
However, in the future we can see this one as the common integration point for
other export use cases.

## ATen Operator Definition

[under construction]
