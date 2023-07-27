# Delegate a PyTorch module to a different backend in Executorch

This note is to demonstrate the basic end-to-end flow of backend delegation in
the Executorch runtime.

At a high level, here are the steps needed for delegation:

1. Add your backend to Executorch.
2. Frontend: lower the PyTorch module or part of the module to a backend.
3. Deployment: load and run the lowered module through Executorch runtime
interface.


## Frontend

There are three flows for delegation:

1. Lower the whole module. Good for testing a fully lowereable module.
1. Lower the whole module and compose it with another module. Good for reusing lowered module exported from other flow.
2. After getting the module, lowering the subgraph partitioned by the according partitioner, like XNNPACK partitioner. Good for lowering a model including both lowerable and non-lowerable nodes.

### Flow 1: Lowering the whole module

The flow starts from a traced graph module with Edge Dialect representation. To lower
it, we call the following function which returns a `LoweredBackendModule` (more
documentation on this function can be found in the Python API reference):

```python
def to_backend(
    backend_id: str,
    edge_program: ExportedProgram,
    compile_spec: List[CompileSpec],
) -> LoweredBackendModule:
```

Within this function, the backend's `preprocess()` function is called which
produces a compiled blob which will be emitted to the flatbuffer binary. The
lowered module can be directly captured, or be put back in a parent module to be
captured.  Eventually the captured module is serialized in the flatbuffers model
that can be loaded by the runtime.

The following is an example of this flow:

```python
from executorch.backends.backend_api import to_backend, MethodCompileSpec
import executorch.exir as exir
import torch

# The submodule runs in a specific backend. In this example,  `BackendWithCompilerDemo` backend
class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# Convert the lowerable module to Edge IR Representation
to_be_lowered = LowerableSubModel()
example_input = (torch.ones(1), )
to_be_lowered_exir_submodule = exir.capture(to_be_lowered, example_input).to_edge()

# Import the backend implementation
from executorch.backends.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
lowered_module = to_backend('BackendWithCompilerDemo', to_be_lowered_exir_submodule, [])
```

We can emit the program directly by running

```python
# API to be added
program = lowered_module.program
```

### Flow 2: Lowering the whole module and composite

After flow 1, alternatively we can compose this lowered module with another module:

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
exec_prog = exir.capture(composite_model, model_inputs).to_edge().to_executorch()

# Save the flatbuffer to a local file
save_path = "delegate.fft"
with open(save_path, "wb") as f:
    f.write(exec_prog.buffer)
```

### Flow 3: Partitioning

The flow starts from a traced graph module with Edge Dialect representation. To lower
certain nodes in this graph module, we can use the overloaded `to_backend`
function (more documentation on this function can be found in the Python API
reference):

```python
def to_backend(
    edge_program: ExportedProgram,
    partitioner: Type[TPartitioner],
) -> ExportedProgram:
```

This function takes in a `Partitioner` which adds a tag to all the nodes that
are meant to be lowered. It will also contain a `partition_tags` mapping tags to
backend names and module compile specs. The tagged nodes will then be
partitioned and lowered to their mapped backends using Flow 1's process.
Available helper partitioner are documented [here](./passes.md#partitioner). These
lowered modules will be inserted into the toplevel module and serialized.

The following is an example of the flow:
```python
from executorch.backends.backend_api import to_backend
import executorch.exir as exir
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
gm = exir.capture(model, model_inputs).to_edge()

from executorch.backends.test.op_partitioner_demo import AddMulPartitionerDemo
exec_prog = to_backend(gm, AddMulPartitionerDemo).to_executorch(
    exir.ExecutorchBackendConfig(passes=SpecPropPass())
)

# Save the flatbuffer to a local file
save_path = "delegate.fft"
with open(save_path, "wb") as f:
    f.write(exec_prog.buffer)
```

## Runtime

The serialized flatbuffer model is loaded by the Executorch runtime. The
preprocessed blob is directly stored in the flatbuffer, which is loaded into a
call to the backend's `init()` function during model initialization stage. At
the model execution stage, the initialized handled can be executed through the
backend's `execute()` function.

To run the real model with executor:

```python
# Load the program with executor runtime
executorch_module = _load_for_executorch_from_buffer(flatbuffer)
print("model_inputs: ", model_inputs)
# Execute the program
model_outputs = executorch_module.forward([*model_inputs])
```

## Error Messages

If there is an error in the backend, for example, if there is any operator that
is not supported by the backend, a debug handler can be thrown. It can surface
back to the Python frontend with source code information. Below is an example
where operator `tan` is not supported in `BackendWithCompilerDemo` backend.

A problematic program:
```python
class TanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tan(x)

tan_module = TanModule()
model_inputs = (torch.ones(1),)
edgeir_m = exir.capture(tan_module, model_inputs).to_edge()
lowered_tan_module = to_backend(
    "BackendWithCompilerDemo", edgeir_m, []
)

class CompositeModelWithTan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lowered_tan = lowered_tan_module

    def forward(self, x):
        output_from_submodule = self.lowered_tan(x)
        return output_from_submodule

composite_model_with_tan = CompositeModelWithTan()
model_inputs = (torch.ones(1),)

composite_model_with_tan(*model_inputs)

exec_prog = (
    exir.capture(composite_model_with_tan, model_inputs).to_edge().to_executorch()
)

buff = exec_prog.buffer
model_inputs = torch.ones(1)

# Load and init the program in executor
executorch_module = _load_for_executorch_from_buffer(buff)

# Expect to throw with debug handler here.
model_outputs = executorch_module.forward([model_inputs])
```

It's expected to capture debug handler like `instruction demo::tan_default<debug_handle>1 is not supported, debug handler is: 1`
