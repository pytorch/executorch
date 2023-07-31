# Soundness

The main mechanism to ensure the correctness of the exported program when called
with other inputs is the specification of constraints on the shapes of inputs.
Concretely, dimensions that are static are specialized, and other dimensions
must be marked dynamic, along with any constraints they must satisfy. This
information is then converted into assertions on each dimension that other
inputs must satisfy.

## Specifying Constraints
To mark a dimension dynamic, you pass the relevant example input tensor and
dimension to torch._export.dynamic_dim. It is possible to specify bounds on and
equalities between such dimensions. In particular, you can use Python relational
operators to provide:
- expected upper and lower bounds of dynamic dimensions;
- expected equalities between dynamic dimensions.

You then pass these specifications as constraints to export.
```python
from torch._export import dynamic_dim, export

def foo(x):  # expect x to be a tensor
  ...

t = torch.rand(4, 8, 32)  # example input tensor
# mark some dimensions of input tensor to be dynamic (assumed static by default)
constraints = [
    dynamic_dim(t, 0),
    dynamic_dim(t, 1) <= 256,
]
exported_foo = export(foo, t, constraints=constraints)

# expect that exported_foo can now be called with other input tensors
# and constraints encode conditions on such input tensors for correctness
```
Note that dynamic dimensions are tracked "symbolically" by the compiler—for
correctness, it cannot use their "concrete" values in example inputs in the
exported program, but only the specified constraints on them. When the compiler
finds any additional necessary conditions on them as it traces through the code,
it reports them back as part of a ConstraintViolationError. Next, let us look at
how to fix such an error.

## Constraint Violations and How to Fix Them
Usually you will have some idea of which dimensions you want to be dynamic, and
what bounds you want on them. But suppose that you want the compiler to guide
you. In that case, just specify what you think is reasonable—the compiler will
emit actionable error messages where needed. In the limit, you can specify all
dimensions to be dynamic, with no bounds, and see where that leads!
```python

from torch._export import dynamic_dim, export

def foo(x):  # expect x to be a tensor
  ...

t = torch.rand(4, 8, 32)  # example input tensor
# I want the compiler to guide me on what ranges to specify
constraints = [dynamic_dim(t, i) for i in range(t.dim())]
exported_foo = export(foo, t, constraints=constraints)
```
Suppose that when tracing the code, the compiler finds that dimension 1 must
have a non-trivial upper bound and dimension 2 must be a constant. The compiler
will emit an error of the following form:

```python
torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated!
  ...

The following dimensions have been specialized. They CANNOT be dynamic.
def specializations(x):
  return x.size()[2] == 32

The following dimensions CAN be dynamic. Here’s how to specify constraints on them:
def specify_constraints(x):
  return [
    dynamic_dim(x, 0),
    dynamic_dim(x, 1) <= 256,
  ]
```
In other words, this error means that:
- Dimension 2 of the input was found to be constrained to be 32. The generated code will assume that `x.size()[2] == 32`, possibly use this value for specialization, and will assert this condition on other inputs.
- Dimension 0 and 1 of the input can range over different values. Moreover, dimension 1 cannot be more than 256. The generated code will assume that `x.size()[1] <= 256`, possibly use this upper bound for memory planning, and will assert this condition on other inputs.

At this point, you are free to use these discovered facts as you choose for the
final specification:* You may use them "as is."
- You may include further knowledge based on the intended use of the exported program, such as:
   - upper-bounding dimension 0, say with 1024;
   - tightening the upper bound on dimension 1, say with 128;
   - deciding that one or both should not be considered dynamic: you do this by taking them out of constraints, effectively asking the compiler to specialize on their concrete value in the input.

Or you may be surprised and want to dig in further, to try to find out why the
compiler discovered these facts. For that, you can re-run the export script with
prefix `TORCH_LOGS=dynamic,dynamo` on the command line.) You will see log messages
such as the following:

```python
[INFO] creating symbol: s0 = 4 with source: x.size()[0]
[INFO] creating symbol: s1 = ... x.size()[1]
[INFO] creating symbol: s2 = ... x.size()[2]

[INFO] adding guard: s1 <= s2 * 8 at:
File "example.py", line 629, in foo
    if x.shape[1] <= x.shape[2] * 8:
[INFO] adding guard: s1 * 4 <= 2048 at: ...
[INFO] adding guard: s2 // 8 >= 2 at: ...
[INFO] adding guard: s2 * s2 + s1 * 8 <= 4096 at: ...
[INFO] adding guard: s2 % 4 == 0 at: ...
[INFO] adding guard: s2 * 2 == 64 at: ...
[INFO] Summary of dimension constraints:
The following dimensions have been specialized and CANNOT be dynamic.

def specializations(x: torch.Tensor):
    assert x.size()[0] == 2
    assert x.size()[1] == 2

```
Under the hood, the compiler creates symbols for each dimension—in this case,
`(s0, s1, s2)`—and generates conditions involving these symbols. For example, the
condition `s1 <= s2 * 8` is generated when tracing the shown line of code in
function `foo`. You can also see a list of specializations we have assumed on
input shapes. This can give you an idea of how your code led to the individual
pieces of information being discovered, which were ultimately simplified to
produce the final error message.
