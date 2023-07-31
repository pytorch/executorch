# Background

## Setup
Let's say you have a function that you want to export. You export it by passing
example inputs (tensors) as arguments to `torch._export.export`. The exported
program can then be called with other inputs.
```python
from torch._export import export

def foo(x):  # expect x to be a tensor
  ...

t = torch.rand(4, 8, 32)  # example input tensor
exported_foo = export(foo, t)

# expect that exported_foo can now be called with other input tensor
```
More generally, the function to be exported can take multiple inputs, and the
function itself could be a torch.nn.Module (with a forward method). See [Export
API Reference](./export_api_reference.md).

## Graph Breaks

The PT2 compiler is a "tracing" compiler, which means that it compiles the
execution path—or "trace"—of your function on your example inputs. The
intermediate representation of such a trace is a graph. In eager mode it is
usual to have graph breaks, where the compiler can fail to trace some parts of
the code; this is fine because it can always fall back to the Python interpreter
to fill these gaps. However in export mode we do not want any graph breaks: we
want the compiler to capture the entire execution in a single graph.

### Rewriting Code
s
Graph breaks can arise either because of missing support for Python features, or
because the compiler cannot decide which control flow path to continue tracing
on. In most cases, it is possible to rewrite code to avoid such graph breaks and
complete the export.When the compiler's tracing mechanism does not support some
Python feature, we strive to provide a workaround as part of the error message.
Over time, we expect to fill in such gaps. On the other hand, not being able to
decide which control flow path to continue tracing on is a necessary limitation
of the compiler. You are required to use special operators to unblock such
cases. See [Control Flow Operators](../ir_spec/control_flow.md).

## Shapes

Recall that while we need example inputs for export, we must generalize the
exported program to be callable with other inputs. The main mechanism for this
generalization is through reuse of shapes (of tensors). Next, let us dive deeper
into shapes.

### Static and Dynamic Dimensions

The shape of a tensor is a tuple of dimensions. Roughly speaking, the exported
program can be called with inputs of the same shape as the example inputs. By
default, we assume that dimensions are static: the compiler assumes they are
going to be the same, and specializes the exported program to those
dimensions.However, some dimensions, such as a batch dimension, are expected to
not be the same—the example inputs passed to export may have a different batch
size as inputs to the exported program. Such dimensions must be marked dynamic.
See [Soundness](./soundness.md) to learn how to specify properties of such
dimensions.
