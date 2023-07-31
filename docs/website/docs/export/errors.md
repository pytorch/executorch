# Errors

In this section we discuss errors that can commonly arise during export.

## Expected Graph Breaks

### Unsupported Features
In PT2 Export, we are primarily reusing the same tracing mechanism—Dynamo—that
we use in eager mode. Recall that in eager mode, graph breaks are expected—we
always have a fallback option. A consequence of this design is that Dynamo has
incomplete coverage of PyTorch and Python features. (That said, the fewer graph
breaks there are, generally speaking, the better performance we can expect—even
in eager mode—because it enables optimizations to apply over larger regions of
code. Thus we are actively working on filling in coverage gaps to avoid graph
breaks where possible.)Unfortunately, this means that you may encounter graph
breaks during export due to Dynamo coverage gaps. In such cases, you should
expect to get an error that includes a link to [ExportDB](./exportdb.md). The
corresponding entry should show a minimal negative example (failure) and a
minimal positive example (success) that should help you understand the
limitation and the workaround, i.e., how to fix the error by rewriting code.

## Constraint Violations
Recall that you can specify constraints on dynamic dimensions, which encode the
soundness conditions for export. It is possible that these constraints are not
valid.

### Various Cases
Specifically, the compiler may find that:
- A dynamic dimension must be equal to a constant.
  - In this case, this dimension must be static: you cannot mark it dynamic.
- A dynamic dimension must be in a range that does not follow the specified range, i.e., is not entirely included between the specified lower and upper bounds.
  - In this case, you need to adjust the specified bounds.
  - Note that when bounds are not specified, they are implicitly assumed to be [2, infinity).
  - For technical reasons that are difficult to explain here, they are assumed to be not 0 or 1. This is not a bug, and does not necessarily mean that your exported program will not work for dimensions 0 or 1. It does mean, though, that you should test for these cases.
- A dynamic dimension must be equal to another dynamic dimension that it is not specified equal to.
   - In this case, you need to add the missing equality.
   - By default, all dynamic dimensions are assumed to be independent.
   - For legacy reasons that are difficult to explain here, you might find spurious implicitly assumed equalities when dimensions in your example inputs happen to be equal. If you ever encounter such a case, please report it as a bug.

## Using the Compiler as a Guide

See [this overview](./soundness.md#constraint-violations-and-how-to-fix-them) of
how to fix such errors. Briefly:
* You should see generated functions specializations and specify_constraints on the console that respectively summarize which dimensions are assumed static and what the necessary constraints on the remaining dynamic dimensions are.
* If you agree with this information, you can copy-paste and call specify_constraints with your example inputs to specify constraints, and you can copy-paste and call specializations on your example inputs to assert their constant values.
* If you do not agree and would like to provide tighter constraints, feel free to modify specify_constraints; the compiler will be happy to accept.
* If you do not agree and would like looser constraints, please use TORCH_LOGs=dynamic to enable INFO-level dynamic-shape logging, which will guide you to where the inferred constraints come from. You can also try TORCH_LOGs=+dynamic to enable (further, verbose) DEBUG-level logging.
   * Note that you might have to change your code or your expectations based on this information. If you are absolutely convinced that the compiler has a bug, please report it! For example, there are tricky cases where the constraints may come from non-user code, like a fast path in the compiler itself. We encourage you to try different example inputs to avoid such constraints.

## Missing META Kernels for Operators

### ATen Operators
In the unfortunate case where your model uses an ATen operator that is not
supported yet, you may get an obscure error of the form:
```python
Unable to find op(FakeTensor, FakeTensor, ...)
```
Please report a bug if you encounter this error.

### Custom Operators
In this case you should follow the instructions at [Custom
Operators](./custom_operators.md). Note that the current mechanism is not ideal,
but will be updated soon to make it easy for you to register custom operators.

## Validation Errors
Note that we do not do any validation of the exported program yet; this is
planned for the near future. In these cases you should report a bug since the
issue is likely in PyTorch.

### Correctness
The export workflow should complain when the exported program behaves
differently than the eager program by running the example inputs through both.
