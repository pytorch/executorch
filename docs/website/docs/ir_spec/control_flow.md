# Control Flow

EXIR has a couple of special operators used to help specify control flow within
some code, similar to jax's control flow operators. Currently these operators
are only supported for inference.

## torch.ops.cond

The `cond` function represents an “if” statement in other programming languages.
It can logically be seen as implemented as follows:

```python
def cond(
    pred: Union[bool, torch.Tensor],
    true_branch: Callable,
    false_branch: Callable,
    operands: List[torch.Tensor]
):
    if pred:
        return true_branch(*operands)
    else:
        return false_branch(*operands)
```

Parameters
* `pred (Union[bool, torch.Tensor])`: A boolean expression or a tensor with one element,
    indicating which branch function to apply, or a boolean expression
* `true_branch (Callable)`: A callable function (a -> b) that is within the
    scope that is being traced.
* `false_branch (Callable)`: A callable function (a -> b) that is within the
    scope that is being traced. The true branch and false branch must have
    consistent input and outputs, meaning the inputs have to be the same, and
    the outputs have to be the same type and shape.
* `operands (List[torch.Tensor])`: A list of inputs to the true/false
    branches.

Returns:
* Value (b) of either `true_branch(*operands)` or `false_branch(*operands)`,
    depending on the value of `pred`.

### Limitations
* The conditional statement (aka `pred`) must meet one of the following constraints:
  * It's a `torch.Tensor` with only one element, e.g. `torch.tensor(10)` or
      `torch.tensor([[10]])`, etc.
  * It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`
* The operands must be a list of tensors
* The branch function must meet all of the following constraints:
  * The function signature must match with operands
  * The function must return a single tensor with same metadata, e.g. shape,
      dtype, etc.
  * The function can not have closure variables (except `self` variable)
  * The function can not have inplace mutations on inputs or global variables
  * The function can not be static method

### Examples
An basic example of how to use the `cond()` operator:

```python
from functorch.experimental.control_flow import cond

class DynamicShapeCondPredicate(torch.nn.Module):
    """
    A basic usage of control flow based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        return cond(x.shape[0] > 4, true_fn, false_fn, [x])
```

This results in the following GraphModule:

```
# Toplevel graph module
class GraphModule(torch.nn.Module):
    graph():
        %arg0 : f32[s0, s1] = placeholder[target=arg0]
        %sym_size : Sym(s0) = call_function[target=torch.ops.aten.sym_size](args = (%arg0, 0), kwargs = {})
        %gt : Sym(s0 > 4) = call_function[target=operator.gt](args = (%sym_size, 4), kwargs = {})
        %true_graph_0 = get_attr[target=true_graph_0]
        %false_graph_0 = get_attr[target=false_graph_0]
        %cond : f32[s0, s1] = call_function[target=torch.ops.cond](args = (%gt, %true_graph_0, %false_graph_0, [%arg0]), kwargs = {})
        return [cond]

    # true_graph_0
    class <lambda>(torch.nn.Module):
        graph():
            %arg0_1 : f32[s0, s1] = placeholder[target=arg0_1]
            %cos : f32[s0, s1] = call_function[target=torch.ops.aten.cos.default](args = (%arg0_1,), kwargs = {})
            return cos

    # false_graph_0
    class <lambda>(torch.nn.Module):
        graph():
            %arg0_1 : f32[s0, s1] = placeholder[target=arg0_1]
            %sin : f32[s0, s1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
            return sin
```

**See examples of advanced usage of `cond()` operator in ExportDB: [cond tag](https://www.internalfb.com/intern/staticdocs/exportdb/cond.html)**


## torch.ops.map

The `map` function is similar to Python's builtin `map`, where it represents
applying an operation in the first dimension of a tensor.
It can logically be seen as implemented as follows:

```python
def map(f: Callable, xs: Tensor, *args: Any) -> Tensor:
    return torch.stack([f(x, *args) for x in xs])
```

Parameters
* `f (Callable)`: A callable function that is applied element-wise over the
    first dimension of `xs`. It should not consume keyword-only args, and should
    take `1 + len(args)` number of arguments.
* `xs (Tensor)`: The tensor to map over. If `xs` is a TensorList, then it has to
    contain tensors of the same shape.
* `args`: Inputs that are needed for the map function `f` in addition to each
    axis of `xs`

Returns
* A mapped tensor or tensor-list. The return tensor's has shape `xs.size[0]` in
    the first dimension, and the other dimensions depend on the return shape of
    `f`.


An example of how to use the map operator:

```
from functorch.experimental.control_flow import map

def dynamic_shape_map(xs, y):
    """
    functorch map() maps a function over the first tensor dimension.
    """

    def body(x, y):
        return x + y

    return map(body, xs, y)
```

This results in the following GraphModule:

```
# Toplevel graph module
class GraphModule(torch.nn.Module):
    graph():
        %arg0: f32[3, s1] = placeholder[target=arg0]
        %arg1: f32[s1] = placeholder[target=arg1]
        %body_graph_0 = get_attr[target=body_graph_0]
        map_1: f32[3, s1] = call_function[target=torch.ops.map](args = (%body_graph_0, %arg0, %arg1), kwargs = {})
        return [map_1]

    # body_graph_0
    class <lambda>(torch.nn.Module):
        graph():
            %arg0_1: f32[s1] = placeholder[target=arg0_1]
            %arg1_1: f32[s1] = placeholder[target=arg1_1]
            %add: f32[s1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
            return [%add]
```

**See examples of advanced usage of `map()` operator in ExportDB: [map tag](https://www.internalfb.com/intern/staticdocs/exportdb/map.html)**

## torch.ops.while

TODO
<!-- A while loop is another control flow construct representing a repeated action.


#### Representation in FX


```
%name = call_function[target = exir.while_loop](args = (condition, body, init_val), kwargs = {})
```


Above, both condition and body are “functions” represented by GraphModule. The semantics of this node is interpreted as, while `condition` is true, keep executing `body`. The return value of this node is the last val produced by `body` (i.e. the one that failed `cond`).

The implementation of exir.while_loop matches this description:


```
def while_loop(condition, body, init_val):
    val = init_val
    while cond(*val):
       val = body(*val)
    return val
``` -->
