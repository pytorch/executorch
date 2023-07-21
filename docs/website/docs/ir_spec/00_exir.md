# EXIR Reference

**Last Update:** July 21, 2023

EXIR is an intermediate representation (IR) for compilers, which bears
similarities to MLIR and TorchScript. It is specifically designed to express the
semantics of PyTorch programs that are written in Python. EXIR primarily
represents computation in a streamlined list of operations, with limited support
for dynamism such as control flows.

To create an EXIR, a front end can be used that soundly captures a PyTorch
program via a trace-specializing mechanism. The resulting EXIR can then be
optimized and executed by a backend.

 The key concepts that will be covered in this document include:
 - ExportedProgram: the datastructure containing the EXIR program
 - Graph: which consists of a list of nodes.
 - Nodes represent operations, control flow, and metadata stored on this node.
 - Values are produced and consumed by nodes.
 - Types are associated with values and nodes.
 - The size and memory layout of values are also defined.


## Assumptions:

This doc assumes that the audience is sufficiently familiar with PyTorch
specifically with `torch.fx` and its related toolings. Thus it will stop
describing contents present in torch.fx documentation and paper. [1](#torchfx)


## What is EXIR:

EXIR is a graph-based intermediate representation IR of PyTorch programs. EXIR
is realized on top of `torch.fx` Graph. In other words, **all EXIR graphs are
also valid FX graphs**, and if interpreted using standard FX semantics
[1](#torchfx), EXIR can be interpreted soundly. One implication is that it can
be converted to a valid Python program via standard FX codegen.

This documentation will primarily focus on highlighting areas where EXIR differs
from FX in terms of its strictness, while skipping parts where it shares
similarities with FX.

You can follow the [tutorial](../tutorials/frontend.md) to play around with what is said here.


## ExportedProgram

The top-level EXIR construct is an `ExportedProgram` class. It bundles the
computational graph of a PyTorch model (which is usually a `torch.nn.Module`)
with the parameters or weights that this model consumes.

The `ExportedProgram` has the following attributes:

* `graph_module (torch.fx.GraphModule)`: Data structure containing the flattened
  computational graph of the PyTorch model. The graph can be directly accessed
  through `ExportedProgram.graph`.
* `graph_signature (ExportGraphSignature)`: The graph signature specifies the
  parameters and buffer names used and mutated within the graph. Instead of
  storing parameters and buffers as attributes of the graph, they are lifted as
  inputs to the graph. The graph_signature is utilized to keep track of
  additional information on these parameters and buffers.
* `call_spec (CallSpec)`:  When running the exported program in eager mode, the
  call spec defines the format specification of inputs and outputs. The graph
  itself accepts a flattened list of inputs and returns a flattened list of
  outputs. In cases where inputs/outputs are not in a flattened list format
  (e.g., a list of lists), we use `call_spec.in_spec` to flatten the inputs and
  `call_spec.out_spec` to unflatten the outputs into the format expected by the
  models when running eagerly.
* `state_dict (Dict[str, Union[torch.Tensor, torch.nn.Parameter]])`: Data structure
  containing the parameters and buffers.
* `range constraints (Dict[sympy.Symbol, RangeConstraint])`: For programs that
  are exported with data dependent behavior, the metadata on each node will
  contain symbolic shapes (hich look like `s0`, `i0`). This attribute maps the
  symbolic shapes to their lower/upper ranges.
* `equality_constraints (List[Tuple[InputDim, InputDim]])`: A list of nodes in
  the graph and dimensions that have the same shape.


## Graph

An EXIR Graph is a PyTorch program represented in the form of a DAG (directed acyclic graph).
Each node in this graph represents a particular computation or operation, and
edges of this graph consist of references between nodes.

We can view Graph having this schema:

```python
class Graph:
  nodes: List[Node]
```

In practice, EXIR's graph is realized as `torch.fx.Graph` Python class.

An EXIR graph contains the following nodes (Nodes will be described in more
details in the next section):

* 0 or more nodes of op type `placeholder`
* 0 or more nodes of op type `call_function`
* exactly 1 node of op type `output`

**Collorary:** The smallest valid Graph will be of one Node. i.e. nodes is never empty.

**Definition:**
The set of `placeholder` nodes of a Graph represents the **inputs** of the Graph of GraphModule.
The `output` node of a Graph represents the **outputs** of the Graph of GraphModule.

Example:
```python
from torch import nn

class MyModule(nn.Module):

    def forward(self, x, y):
      return x + y

mod = torch._export.export(MyModule())
print(mod.graph)
```
Output:
```
graph():
    %arg0_1 : [#users=1] = placeholder[target=arg0_1]
    %arg1_1 : [#users=1] = placeholder[target=arg1_1]
    %add : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    return [add]
```

The above is the textual representation of a Graph, with each line being a node.

## Node

A Node represents a particular computation or operation, and is represented in
Python using `torch.fx.Node` class. Edges between nodes are represented as
direct references to other nodes via the args property of the Node class. Using
the same FX machinery, we can represent the following operations that a
computational graph typically needs, such as operator calls, placeholders (aka
inputs), conditionals, and loops.

The Node has the following schema:

```python
class Node:
  name: str # name of node
  op_name: str  # type of operation

  # interpretation of the fields below depends on op_name
  target: [str|Callable]
  args: List[object]
  kwargs: Dict[str, object]
  meta: Dict[str, object]
```

### FX Text Format

As the example above, notice that each line has this format
```
   %<name>:[...] = <op_name>[target=<target>](args = (%arg1, %arg2, arg3, arg4, …)), kwargs = {})
```

This format captures everything present in the Node class, with exception of `meta`, in a compact format.

Concretely:

**&lt;name>** is the name of node as would appear in `node.name`

**&lt;op_type>** is the `node.op_name` field, which must be one of these:
[call_function](#callfunction), call_method, [placeholder](#placeholder),
[get_attr](#getattr), or [output](#output).

**&lt;target>** is the target of the node as `node.target`. The meaning of this
field depends on `op_type`.

**args1, … args 4…** are what listed in `node.args` tuple, if a value in the list
is a `fx.Node`, then it will be especially indicated with a leading **%.**

For example, a call to the add operator would appear as

```
%add1 = call_function[target = torch.op.aten.add.Tensor](args = (%x, %y), kwargs = {})
```
Where `%x`, `%y` are 2 other Nodes that have names x and y.
Worth noting that, the string `torch.op.aten.add.Tensor` represents the
callable object that is actually stored in the target field, not merely its string
name.

The final line of this text format is
```
return [add]
```
is a Node with `op_name = output`, this is used to indicate that we are returning this one element.

### call_function
A `call_function` node represents a call to an operator.

#### Definitions

* **Functional:** We say a callable is “functional” if it satisfy all following requirements:
  * Non-aliasing, ie output tensors do not share data storage with each other or with inputs of the operator
  * Non-mutating, ie the operator does not mutate value of it’s input (for tensors, this includes both metadata and data)
  * No side effects, ie the operator does not mutate states that are visible from outside, like changing values of module parameters.

* **Operator:** is a functional callable with a predefined schema. Examples of
  such operators include functional ATen operators.


#### Representation in FX
```
%name = call_function[target = operator](args = (%x, %y, …), kwargs = {})
```

#### Differences from vanila FX call_function

1. In FX graph, a call_function can refer to any callable, in EXIR, we restrict
this to only Canonical ATen operators (a select subset of PyTorch ATen operator
library), custom operators and control flow operators.
2. In FX graph, calling with both args and kwargs is allowed, in EXIR, only args
will be used, kwargs will be an empty dict.
3. In EXIR, constant arguments will be embedded within the graph.


#### Metadata

`Node.meta` is a dict attached to every FX node. However, FX spec does not
specify what metadata can or will be there. EXIR provides a stronger contract,
specifically all `call_function` nodes will guarantee to have and only have
the following metadata fields:

* `node.meta["stack_trace"]` is string containing the python stack trace
  referencing the original python source code. An example stack trace looks like:
  ```
    File "my_module.py", line 19, in forward
      return x + dummy_helper(y)
    File "helper_utility.py", line 89, in dummy_helper
      return y + 1
  ```
* `node.meta["val"]` describes the output of running the operation. It can be
  of type [`SymInt`](#symint), [`FakeTensor`](#faketensor), a list of
  `TensorMeta` or `None`.

* `node.meta["nn_module_stack"]` describes the "stacktrace" of the `torch.nn.Module`
  from which the node came from, if it was from a `torch.nn.Module` call. For
  example, if a node containing the `addmm` op called from a `torch.nn.Linear`
  module inside of a `torch.nn.Sequential` module, the `nn_module_stack` would
  look something like:
  ```python
  {'self_linear': ('self.linear', <class 'torch.nn.Linear'>), 'self_sequential': ('self.sequential', <class 'torch.nn.Sequential'>)}
  ```

* `node.meta["source_fn"]` contains the torch function or the leaf
  `torch.nn.Module` class this node was called from before decomposition. For
  example, a node containing the `addmm` op from a `torch.nn.Linear` module call
  would contain `torch.nn.Linear` in their `source_fn`, and a node containing
  the `addmm` op from a `torch.nn.functional.Linear` module call would contain
  `torch.nn.functional.Linear` in their `source_fn`.


### placeholder

Placeholder represents input to a graph. Its semantics are exactly the same as in FX.
Placeholder nodes must be the first N nodes in the nodes list of a graph. N can be zero.

#### Representation in FX:

```
%name = placeholder[target = name](args = ())
```

The target field is a string which is the name of input.

`args`, if non empty; should be of size 1 representing the default value of this input.

#### Metadata

Placeholder nodes also have `meta[‘val’]`, like `call_function` nodes. The val field
in this case represents the input shape/dtype that the graph is expected to
receive for this input parameter.

### output

An output call represents a return statement in a function; thus terminates the current graph.
There is one and only one output node, and it will always be the last node of the graph.

#### Representation in FX

```
output[](args = (%something, …))
```

This is the exact semantics as in FX [1](#torchfx). args represents the node to be returned.

#### Metadata

Output node has the same metadata as `call_function` nodes.

### get_attr

`get_attr` nodes represent reading a submodule from the encapsulating
`GraphModule`. Unlike a vanilla FX graph from `torch.fx.symbolic_trace` in which
`get_attr` nodes are used to read attributes such as parameters and buffers from
the top-level `GraphModule`, parameters and buffers will be passed in as inputs
to the graph module, and stored in the toplevel `ExportedProgram`.

#### Representation in FX:

```
%name = get_attr[target = name](args = ())
```

#### Example:
Consider the following model:

```python
class TrueModule(torch.nn.Module):
  def forward(self, x):
    return x.sin()

class FalseModule(torch.nn.Module):
  def forward(self, x):
    return x.cos()

class Module(torch.nn.Module):
  def __init__(self):
    self.true_module = TrueModule()
    self.false_module = FalseModule()

  def forward(self, x):
    return torch.ops.higher_order.cond(x.shape[0] == 1, self.true_module, self.false_module, x)
```

Then, `%name = get_attr[target = true_module](args = ())` appears in the corresponding
graph to read the attribute `self.true_module`.


## EXIR Dialects

EXIR is a specification that consists of the following parts:

1. A definition of computation graph model.
2. Set of operators allowed in the graph.

An EXIR dialect is a EXIR graph composed with the operations defined below, but
with additional properties (such as restrictions on operator set or metadata)
that are meant for a specific purpose.

The EXIR dialects that currently exist are:

* [ATen Dialect](./01_aten_dialect.md)
* [Edge Dialect](./02_edge_dialect.md)
* [Backend Dialect](./03_backend_dialect.md)

These dialects represent stages that a captured program goes through from
program capture to be converted into an executable format. For example,
a compilation pipeline targeting Edge devices may look like this: a Python program
is first captured as ATen dialect, then ATen is converted to Edge Diaelct, Edge
to Backend, and finally converted from EXIR to a binary format for execution.


## References:

### torch.fx

Documentation of torch.fx: [https://pytorch.org/docs/stable/fx.html](https://pytorch.org/docs/stable/fx.html)


### SymInt

A SymInt is an object that can either be a literal integer or a symbol that represents
an Integer (represented in python by `sympy.Symbol` class). When SymInt is a
symbol, it describes a variable of type integer that is unknown to the graph at
compile time, and its value is only know at runtime.

### FakeTensor

A FakeTensor is a object that contains metadata of a tensor. It can be viewed as
having the following metadata.

```python
class FakeTensor:
  size: List[SymInt]
  dtype: dtype
  dim_order: List[int]
```

The size field of FakeTensor is a list of integers or SymInts. If SymInts are
present, this means this tensor has a dynamic shape. If integers are present, it
is assumed that that tensor will have that exact static shape. The rank of the
TensorMeta is never dynamic. The dtype field represents the dtype of the
output of that node. There are no implicit type promotions in Edge IR. There
are no strides in FakeTensor.

In other words:

* If the operator in node.target returns a Tensor, then, node.meta['val'] is a
  FakeTensor describing that tensor.
* If the operator in node.target returns a n-tuple of Tensors, then,
  node.meta['val'] is a n-tuple of FakeTensors describing each tensor.
* If the operator in node.target returns a int/float/scalar that is known at
  compile time, then, node.meta['val'] is None.
* If the operator in node.target returns a int/float/scalar that is not known
  at compile time, then, node.meta['val'] is of type SymInt.

For example:
* `aten::add` returns a Tensor; so its spec will be a FakeTensor with dtype
  and size of the tensor returned by this operators.
* `aten::sym_size` returns an integer; so its val will be a SymInt because its
  value are only available at runtime.
* `max_pool2d_with_indexes` returns a tuple of (Tensor, Tensor); so the spec
  will be also a 2-tuple of FakeTensor object, the first TensorMeta describes
  the first element of the return value etc.

Python code:
```python
def add_one(x):
  return torch.ops.aten(x, 1)
```
Graph:
```
graph():
    %ph_0 : [#users=1] = placeholder[target=ph_0]
    %add_tensor : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%ph_0, 1), kwargs = {})
    return [add_tensor]
```

FakeTensor
```
FakeTensor(dtype=torch.int, size=[2,], device=CPU)
```

### Pytree-able types

The following types are defined as **leaf type:**

**Source: **


<table>
  <tr>
   <td>Type
   </td>
   <td>Definition
   </td>
   <td>Notes
   </td>
  </tr>
  <tr>
   <td>Tensor
   </td>
   <td>Pytorch Tensor type
   </td>
   <td><a href="https://pytorch.org/docs/stable/tensors.html#torch.Tensor">https://pytorch.org/docs/stable/tensors.html#torch.Tensor</a>
   </td>
  </tr>
  <tr>
   <td>Scalar
   </td>
   <td>Any numerical types from Python, including integral types, floating point types, and zero dimensional tensors.
   </td>
   <td>float and int argument types should suffice for most algorithms (you should only use Scalar if the operator truly may accept either type)
   </td>
  </tr>
  <tr>
   <td>int
   </td>
   <td>python int (binded as int64_t in C++)
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>float
   </td>
   <td>python float (binded as double in C++)
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>bool
   </td>
   <td>python bool
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>str
   </td>
   <td>python string
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>ScalarType
   </td>
   <td>Enum type for all permissible dtype
   </td>
   <td><a href="https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype">https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype</a>
   </td>
  </tr>
  <tr>
   <td>Layout
   </td>
   <td>Enum type for all permissible Layout
   </td>
   <td><a href="https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout">https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout</a>
   </td>
  </tr>
  <tr>
   <td>MemoryFormat
   </td>
   <td>Enum type for all permissible MemoryFormat
   </td>
   <td><a href="https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format">https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format</a>
   </td>
  </tr>
  <tr>
   <td>Device
   </td>
   <td>torch.device -> str
   </td>
   <td><a href="https://pytorch.org/docs/stable/tensor_attributes.html#torch.device">https://pytorch.org/docs/stable/tensor_attributes.html#torch.device</a>
   </td>
  </tr>
</table>


The following types are defined as **container type:**


<table>
  <tr>
   <td>Tuple
   </td>
   <td>Python tuple
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>List
   </td>
   <td>Python list
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Dict
   </td>
   <td>Python dict with keys Scalar as defined in the table above.
   </td>
   <td>
   </td>
  </tr>
</table>


**We define a type “Pytree-able”, if it is either a leaf type, or a container type that contains other Pytree-able types.**

NOTE: The concept of pytree is the same as the one documented here for JAX: [https://jax.readthedocs.io/en/latest/pytrees.html](https://jax.readthedocs.io/en/latest/pytrees.html)



### Memory formats

Possible memory formats:

We use the term **Pytorch Default Dims Format **describe the memory format represented by `torch.contiguous_format. `In other words, Let N, C, H, W be number of images, channel, height and weight, then `torch.contiguous_format `will tensor dimensions be in NCHW ordering.

Other memory formats available in torch are: torch.channels_last: = NHWC

Other permutations of NCHW are allowed but we don’t have explicit names for them.


<table>
  <tr>
   <td><strong>Format</strong>
   </td>
   <td><strong>Column Order</strong>
   </td>
  </tr>
  <tr>
   <td>contiguous_format
   </td>
   <td>NCHW
   </td>
  </tr>
  <tr>
   <td>Channels_last
   </td>
   <td>NHWC
   </td>
  </tr>
  <tr>
   <td>(no name)
   </td>
   <td>Other permutations
   </td>
  </tr>
</table>


See more on channel_last mem format: [https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

### Tensor
A Tensor type describes a mathematical tensor.
Let `t` be a Tensor, then, conceptually `t` provides the following interfaces:

1. dtype: t.dtype returns the type of the Scalar associated with this tensor.
   dtype can be one of {int8, int16, int32, int64, float32, float64, bool...},`
   The list of all supported dtypes is listed in this page: https://pytorch.org/docs/stable/tensors.html
2. size: (also known as shape) t.size is a list of integers.
   ** `len(t.size)` is known as the "rank" of tensor
   ** `prod(t.size)` the product of the sizes is the total number of elements in this tensor.

In Python, we use `torch.Tensor` class to represent a Tensor: https://pytorch.org/docs/stable/tensors.html
