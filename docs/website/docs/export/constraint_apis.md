<h1> Constraint APIs </h1>

To enable the export of input shape-dependent models, it is necessary for users to impose constraints on the tracing inputs, ensuring the safe traversal of the model during the export process. For a comprehensive understanding of how to utilize this feature, please refer to [this overview](./soundness.md). To express these constraints, we have developed the following set of APIs.

# dynamic_dim
```python
def dynamic_dim(x: torch.Tensor, dim: int):
    """
    Marks the dimension `dim` of input `x` as unbounded dynamic
    """
    pass
```

It is possible to impose specific bounds on the marked dynamic dimension. For example:
```python
constraints = [dynamic_dim(x, 0) >= 3, dynamic_dim(x, 0) <= 6]
```

By passing the above `constraints` to export, we effectively establish the range of acceptable values for the 0th dimension of tensor x, constraining it between 3 and 6. Consequently, the PT2 Export functionality can safely trace through the following program:
```python
def f(x):
    if x.shape[0] > 3:
       return x.cos()
    return x.sin()
```

Moreover, it is possible to impose specific equalities between marked dynamic dimensions. For example:
```python
constraints = [dynamic_dim(x, 0) == dynamic_dim(y, 0)]
```

This means that whatever the 0th dimensions of tensors x and y may be, they must be the same. This is useful to export the following program, which implicitly requires this condition because of the semantics of `torch.add`

```python
def f(x, y):
    return x + y
```

# constrain_as_value
```python
def constrain_as_value(symbol, min: Optional[int], max: Optional[int]):
    """
    Adds a minimum and/or maximum constraint on the intermediate symbol during the tracing phase.
    """
```

The `constrain_as_value` function informs us that the specified symbol is guaranteed to fall within the provided minimum and maximum values. If no minimum or maximum values are provided, the symbol is assumed to be unbounded. Here's a concrete example of its usage within a model:
```python
def f(x, y):
    b = y.item()
    constrain_as_value(b, 3, 5)
    if b > 3:
       return x.cos()
    return x.sin()
```

# constrain_as_size
```python
def constrain_as_size(symbol, min: Optional[int] = 2, max: Optional[int]):
    """
    Adds a minimum and/or maximum constraint on the intermediate symbol during the tracing phase,
    with additional checks to ensure the constrained value can be used to construct a tensor shape.
    """
```

The `constrain_as_size` API is similar to constrain_as_value but includes additional verifications to ensure that the constrained value can be used to construct a tensor shape. For instance, our tracer specializes in handling shape sizes of 0 or 1, so this API explicitly raises an error if the constrain_as_size is used with a minimum value less than 2. Here's an example of its usage:
```python
def f(x, y):
    b = y.item()
    constrain_as_size(b, 3, 5)
    z = torch.ones(b, 4)
    return x.sum() + z.sum()
```
