# Pattern matching

A tool which takes a symbolic definition of a pattern of operators and yields all matching instances of
the pattern in the internal TFLite model.

### Example use

```python
matcher = PatternMatcher(
    builder,
    [
        Op(['Transpose'], ['x', 'perm'], ['y']),
        Op(['Reshape', 'Squeeze'], ['y', ...], ['z']),
        Op(['FullyConnected'], ['z', 'w'], ['fc_out'], [
            HasFusedActivationFunction()
        ]),
        MultipleSameOps(['Add'], ['fc_out'])
    ],
    [
        TensorsHaveOneConsumer(['y', 'z']),
        TensorHasData('perm')
    ])

for [transpose, reshape, fc, add_ops], tensor_map, input_to_ops, output_to_op in matcher.match_patterns():
    x = tensor_map['x']
    ...
```

The `PatternMatcher` has 3 parameters in its constructor:

* A [`ModelBuilder`](../converter/builder/model_builder.py) object which encapsulates the internal
  TFLite model. This is the model that the `PatternMatcher` will search.
* A list of symbolic operators, which describe the pattern the `PatternMatcher` will search for. Its details are
  described in a [later section](#blocks-to-define-a-pattern) of this document.
* The last parameter is an optional list of tensor rules defined in [tensor_rules.py](tensor_rules.py). They allow
  additional restrictions to be placed on the tensors present in the pattern. The yielded pattern will always satisfy
  all of these rules.

The PatternMatcher will perform 1 pass through the TFLite model encapsulated by the given `ModelBuilder`, and gradually
yield all matching patterns of operators. So changes to the TFLite model done in the body of the `for` loop above, can
immediately have an effect on the next matched instance of the searched pattern.

The method `match_patterns()` will gradually yield a tuple containing:

* A list of matched operators. Their number and order will exactly match the operators specified in the pattern.
* A dictionary mapping symbolic tensor names (such as `x` or `perm` in the example above) to the actual TFLite tensors
  matched in the model.
* A dictionary mapping the name of a real tensor from the model, to a list of operators which use this tensor as their
  input.
* A dictionary mapping the name of a real tensor from the model, to an operator which produces this tensor as its
  output.

The first block in the pattern must be an `Op`. The pattern matcher will internally go through the model until it finds
a match for this first `Op`. It then sets its current position to this `Op` and tries to match the rest of the pattern.
If it succeeds, it yields the matched operators and returns to the current position (the first `Op`). It then continues
on its single pass through the model and tries to find another match for the first `Op`.

That also means that all subsequent blocks must somehow be connected to some previous block. So the following is **not**
allowed.

```python
Op(['Sqrt'], ['a'], ['b']),
Op(['Cast'], ['c'], ['d'])
```

---

# Blocks to define a pattern

## Op

Block which represents exactly 1 TFLite operator.

The class is defined as follows:

```python
class Op:
    ops: list[str] | None = None
    inputs: list[str | None] | None = None
    outputs: list[str | None] | None = None
    op_rules: list[OpRule] | None = None
```

* The matched TFLite operator will have 1 of the operator types specified in `ops`. If the `ops` is `None`, the operator
  type will not be considered during the pattern matching.

* The `inputs` and `outputs` contain symbolic names, which will uniquely identify actual matched tensors from the TFLite
  model. The number specified tensors will be exactly the same in the matched TFLite operator (except for the case
  of `...`).
    * Instead of a symbolic name, `None` can be given which represents an anonymous tensor, which still however must be
      present
      in the matched operator.
    * Another alternative to a symbolic name is the ellipsis `...`. It represents any number of tensors (including 0).
      It can only be used at the beginning and/or at the end of the `inputs`/`outputs`. If it is at the beginning, the
      matching will be done in reverse, starting with the last tensor.
    * The `inputs` and/or `outpus` can be omitted altogether, if they are `None`. This means that the `PatternMatcher`
      will not take the `inputs`/`outpus` into consideration while matching the pattern.

* `op_rules` is a list of rules that the operator must satisfy in order to be matched. They are defined
  in [operator_rules.py](operator_rules.py). The yielded pattern will always satisfy all these rules.

## MultipleSameOps

Block which represents multiple (at least 1) occurrences of similar operators. The similar operators must all fit 1
common definition, which is similar to the definition of `Op`.

```python
class MultipleSameOps:
    ops: list[str]
    inputs: list[str | None] | None = None
    outputs: list[str | None] | None = None
    op_rules: list[OpRule] | None = None
```

At least 1 input of the `MultipleSameOps` must be the output of a previous block. Other inputs/outputs which have not
been defined by a previous block, will represent a set of tensors instead of just 1 tensor.

The `MultipleSameOps` block will be matched with a list of operators, which consume the already matched input tensor.
All operators consuming this tensor must match the `MultipleSameOps` block, in order for the match to be successful.

### Example use

```python
Op(['Quantize'], outputs=['x']),
MultipleSameOps(['Dequantize'], ['x'], ['y'])
```

The first `Op` defines a `Quantize` operator with an output tensor `x`, which is a single tensor. The
following `MultipleSameOps` represents a set of `N` `Dequantize` operators, which all consume the `x` tensor. These `N`
operators all produce their own output, so `y` represents `N` tensors, not just 1.

Tensor rules can still be used for `y`, and they have to pass for all output tensors of the `Dequantize` operators.

Operator rules can still be used for the `MultipleSameOps`, and they have to pass for all matched operators.

It is **not** possible to use tensor/operator rules to filter the matched operators of `MultipleSameOps`. The pattern
matcher will find all operators which use the `x` tensor, and if and only if they **all** match the definition, the
whole pattern is yielded.

Sets of tensors (such as `y` in the example above) cannot be used as inputs to following blocks right now.

### Semantics of consuming a set of tensors

Currently, it is not allowed for any block to consume a set of tensors, such as the `y` in the example above.

## OneOf

Block which represents a single operator, that has to match at least one given `Op` in the `one_of_ops` list.

```python
class OneOf():
    one_of_ops: list[Op]
```

### Example use

```python
Op(['FullyConnected'], outputs=['y']),
OneOf([
    Op(['Add'], ['y', 'b']),
    Op(['Add'], ['b', 'y'])
])
```

The example code above represents a situation where we do not care if the `Add` uses the output of the `FullyConnected`
as its first input or its second input.