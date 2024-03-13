# Lowering a Model as a Delegate

Audience: ML Engineers, who are interested in applying delegates to accelerate their program in runtime.

Backend delegation is an entry point for backends to process and execute PyTorch
programs to leverage the performance and efficiency benefits of specialized
backends and hardware, while still providing PyTorch users with an experience
close to that of the PyTorch runtime. The backend delegate is usually either provided by
ExecuTorch or vendors. The way to leverage delegation in your program is via a standard entry point `to_backend`.


## Frontend Interfaces

There are three flows for delegating a program to a backend:

1. Lower the whole module to a backend. This is good for testing backends and
    the preprocessing stage.
1. Lower the whole module to a backend and compose it with another module. This
    is good for reusing lowered modules exported from other flows.
1. Lower parts of a module according to a partitioner. This is good for
    lowering models that include both lowerable and non-lowerable nodes, and is
    the most streamlined procecss.

### Flow 1: Lowering the whole module

This flow starts from a traced graph module with Edge Dialect representation. To
lower it, we call the following function which returns a `LoweredBackendModule` (more documentation on this function can be found in the [Export API reference](export-to-executorch-api-reference.rst))

```python
# defined in backend_api.py
def to_backend(
    backend_id: str,
    edge_program: ExportedProgram,
    compile_spec: List[CompileSpec],
) -> LoweredBackendModule:
```

Within this function, the backend's `preprocess()` function is called which
produces a compiled blob which will be emitted to the flatbuffer binary. The
lowered module can be directly captured, or be put back in a parent module to be
captured. Eventually the captured module is serialized in the flatbuffer's model
that can be loaded by the runtime.

The following is an example of this flow:

```python
from executorch.exir.backend.backend_api import to_backend
import executorch.exir as exir
import torch
from torch.export import export
from executorch.exir import to_edge

# The submodule runs in a specific backend. In this example,  `BackendWithCompilerDemo` backend
class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# Convert the lowerable module to Edge IR Representation
to_be_lowered = LowerableSubModel()
example_input = (torch.ones(1), )
to_be_lowered_exir_submodule = to_edge(export(to_be_lowered, example_input))

# Import the backend implementation
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
lowered_module = to_backend('BackendWithCompilerDemo', to_be_lowered_exir_submodule.exported_program(), [])
```

We can serialize the program to a flatbuffer format by directly running:

```python
# Save the flatbuffer to a local file
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(lowered_module.buffer())
```

### Flow 2: Lowering the whole module and composite

Alternatively, after flow 1, we can compose this lowered module with another
module:

```python
# This submodule runs in executor runtime
class NonLowerableSubModel(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, a, b):
        return torch.add(torch.add(a, b), self.bias)


# The composite module, including lower part and non-lowerpart
class CompositeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.non_lowerable = NonLowerableSubModel(torch.ones(1) * 0.3)
        self.lowerable = lowered_module

    def forward(self, x):
        a = self.lowerable(x)
        b = self.lowerable(a)
        ret = self.non_lowerable(a, b)
        return a, b, ret

composite_model = CompositeModel()
model_inputs = (torch.ones(1), )
exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch()

# Save the flatbuffer to a local file
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(exec_prog.buffer)
```

### Flow 3: Partitioning

The third flow also starts from a traced graph module with Edge Dialect
representation. To lower certain nodes in this graph module, we can use the
overloaded [`to_backend`
function](https://github.com/pytorch/executorch/blob/d9eef24bb720804aa7b400b05241487510ae0dc2/exir/backend/backend_api.py#L39).

```python
def to_backend(
    edge_program: ExportedProgram,
    partitioner: Partitioner,
) -> ExportedProgram:
```

This function takes in a `Partitioner` which adds a tag to all the nodes that
are meant to be lowered. It will return a `partition_tags` dictionary mapping tags to
backend names and module compile specs. The tagged nodes will then be
partitioned and lowered to their mapped backends using Flow 1's process.
Available helper partitioners are documented
[here](./compiler-custom-compiler-passes.md). These lowered modules
will be inserted into the top-level module and serialized.

The following is an example of the flow:
```python
import executorch.exir as exir
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.program import (
    EdgeProgramManager,
    to_edge,
)
from torch.export import export
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x + y
        x = x * y
        x = x - y
        x = x / y
        x = x * y
        x = x + y
        return x

model = Model()
model_inputs = (torch.randn(1, 3), torch.randn(1, 3))

core_aten_ep = export(model, model_inputs)
edge: EdgeProgramManager = to_edge(core_aten_ep)
edge = edge.to_backend(AddMulPartitionerDemo())
exec_prog = edge.to_executorch()

# Save the flatbuffer to a local file
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(exec_prog.buffer)
```

## Runtime

After having the program with delegates, to run the model with the backend, we'd need to register the backend.
Depending on the delegate implementation, the backend can be registered either as part of global variables or
explicitly registered inside the main function.

- If it's registered during global variables initialization, the backend will be registered as long as it's statically linked. Users only need to include the library as part of the dependency.

- If the vendor provides an API to register the backend, users need to include the library as part of the dependency, and call the API provided by vendors to explicitly register the backend as part of the main function.
