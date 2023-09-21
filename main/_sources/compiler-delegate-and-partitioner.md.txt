# Backend and Delegate

Backend delegation is an entry point for backends to process and execute PyTorch
programs to leverage performance and efficiency benefits of specialized
backends and hardware, while still providing PyTorch users with an experience
close to that of the PyTorch runtime.

At a high level, the entry point for backends is defined by 2 components:

- An IR to represent the program: **Edge Dialect** (which is produced through
    the `to_edge` API)
- A couple of interfaces for backends to implement:
    - Ahead-of-Time (AOT)
        - Program preprocessing (e.g. ahead of time compilation, transformation, optimization...).
    - Runtime
        - Program initialization (e.g. runtime compilation).
        - Program execution.
        - (optional) Program destroy (e.g. release backend owned resource).

## Backend interfaces

A delegate backend implementation is composed of:

1) An ahead-of-time preprocessing interface
2) A runtime initialization and execution interface


### Ahead-of-Time Preprocessing

For the AOT preprocessing, backends are given an edge dialect program,
a list of compile specs specifying the values needed for compilation, and are
expected to return a compiled blob, or binary contains the desired program to be
run in the backend, and profiling information. During serialization, the
compiled blob will be serialized file, and directly loaded to the device. The
API looks something like:

```python
def preprocess(
    edge_program: ExportedProgram,
    compile_specs: List[CompileSpec],
) -> PreprocessResult:
```

A demo of the preprocess function is implemented
[here](https://github.com/pytorch/executorch/blob/main/exir/backend/test/backend_with_compiler_demo.py).
The demo loops through the nodes in the graph module of the `edge_program` and
serializes the `add`, `mul`, and `sin` instructions into a string, which is later
parsed and executed at runtime.

### Runtime Initialization and Execution

During the runtime, the compiled blob from the `preprocess` function will be
loaded and passed directly to the backend's custom `init` function. This
function is responsible for further processing the compiled unit, as well as
perform any backend initialization. The backend's custom `execute` function will
then be called to execute the handle produced by `init`. And finally, if
destroying is required for some backend, backends can implement a `destroy`
function which will be called when the program is out of its lifespan.

```cpp
// Runtime initialization
__ET_NODISCARD virtual Result<DelegateHandle*> init(
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs,
    MemoryAllocator* memory_allocator);

// Runtime execution
__ET_NODISCARD virtual Error execute(DelegateHandle* handle, EValue** args);

// [optional] Runtime destroy. Destroy the resource held by the backend
virtual void destroy(__ET_UNUSED DelegateHandle* handle);
```

Once the backend is ready, they can then be registered:

To register the backend for AOT lowering, just simply import the backend:

```python
from executorch.exir.backend.test.backend_with_compiler_demo import BackendWithCompilerDemo
```

To register the backend for runtime, register via the `register_backend` API:
```cpp
__ET_NODISCARD Error register_backend(const Backend& backend);
```


## Frontend interfaces

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
lower it, we call the following function which returns a `LoweredBackendModule`
(more documentation on this function can be found in the Python API reference):

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
from executorch.exir.backend.backend_api import to_backend, MethodCompileSpec
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
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
lowered_module = to_backend('BackendWithCompilerDemo', to_be_lowered_exir_submodule, [])
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
exec_prog = exir.capture(composite_model, model_inputs).to_edge().to_executorch()

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
    partitioner: Type[TPartitioner],
) -> ExportedProgram:
```

This function takes in a `Partitioner` which adds a tag to all the nodes that
are meant to be lowered. It will return a `partition_tags` mapping tags to
backend names and module compile specs. The tagged nodes will then be
partitioned and lowered to their mapped backends using Flow 1's process.
Available helper partitioner are documented
[here](./compiler-custom-compiler-passes.md). These lowered modules
will be inserted into the top-level module and serialized.

The following is an example of the flow:
```python
from executorch.exir.backend.backend_api import to_backend
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

from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
exec_prog = to_backend(gm, AddMulPartitionerDemo).to_executorch(
    exir.ExecutorchBackendConfig(passes=SpecPropPass())
)

# Save the flatbuffer to a local file
save_path = "delegate.pte"
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

```
> :warning: **pybind is not ready for partner preview**: please use size_test_all_ops or executor_runner cpp binary for now. pybind to run executor will be ready before MVP
```


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
back to the Python frontend with the source code information. Below is an
example where the `tan` operator is not supported in `BackendWithCompilerDemo`
backend.

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

It's expected to capture debug handler like `instruction
demo::tan_default<debug_handle>1 is not supported, debug handler is: 1`


## Common Questions

**1. How can we get data in backend.preprocess?**

The graph module being preprocessed is a lifted graph, this means that static
data like weights and biases are supplied as inputs to the graph. However, we
can access the weights and biases ahead-of-time through the exported program. To
access these parameters from a given node, we can use the function `get_params`
provided in  `torch/_export/utils.py`

**2. How can we embed the data (like weight/bias) to the backend?**

It's common that backend have some ways optimize the const data. In this case,
we'd need to tag the placeholder node which are also the state in the
partitioner, and during backend.preprocess, we can follow the description in the
first question to get the weight.

**3. How can we run the lowered module in Python with the specific backend?**

We haven't added the support yet but that's the plan!

**4. Should we expect to see `get_attr` nodes in the edge dialect program?**

`get_attr` nodes will only show up for submodules used for control flow or
delegation. It won't hold any data.

**5. Can we delegate to multiple backends?**

Yes! There are two ways to do this:

*Option 1: Run to_backend multiple times for different backends*

If we have two backends, backend_1 and backend_2, and they have their own
parititioners: backend_1_parititioner and backend_2_partitioner, we can run it
like:

```python
# Will first lower nodes to backend_1 depending on the backend_1_parititioner depending on partitioner algorithm
exported_program_backend_1 = to_backend(exported_program, backend_1_parititioner)
# For the rest of nodes, they will be lowered to backend_2 depending on backend_2_parititioner
exported_program_backend_1_and_2 = to_backend(exported_program_backend_1, backend_2_parititioner)
```

A more conrete example be found
[here](https://github.com/pytorch/executorch/blob/main/exir/backend/test/demos/test_xnnpack_qnnpack.py).
In this example,
qnnpack is one backend and xnnpack is another backend. We haven't open-sourced
these two backends delegates yet, and this example won't run out of box. It can
be used as a reference to see how it can be done.

This option is easy to try becuase usually all backends will implement their own
parititioner. However this option may get different results if we change the
order of to_backend call. If we want to have a better control on the nodes, like
which backend they should go, option 2 is better.

*Option 2: Have a partitioner which partitions for different backends*

Another option is to create a customized partitioner, say partitioner
`backend_1_2_partitioner`, and inside the partitioner logic,

```python
class Backend_1_2_Partitioner(Partitioner):
    """
    Partitions all add/mul nodes regardless of order for Backend2
    """

    def __init__(self) -> None:
        self.delegation_spec_1 = DelegationSpec("Backend1", [])
        self.delegation_spec_2 = DelegationSpec("Backend2", [])
        self.partition_tags = {}

    def partition(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:

        # Tag all nodes in the first partiton to backend 1
        node_to_backend_1 = ... # some logic to select the nodes from the graph
        delegation_tag = f"backend2_tag{partitioner_1.id}"
        node.meta["delegation_tag"] = delegation_tag
        self.partition_tags[delegation_tag] = self.delegation_spec_1

        # Tag all nodes in the first partiton to backend 2
        node_to_backend_2 = ... # some logic to select the nodes from the graph
        delegation_tag = f"backend2_tag{partitioner_2.id}"
        node.meta["delegation_tag"] = delegation_tag
        self.partition_tags[delegation_tag] = self.delegation_spec_2
        return edge_graph_module
```

**6. Is there an easy way to write a partitioner?**

We provide some helper partitioners
[here](./compiler-custom-compiler-passes.md) to make it easy to find
nodes from decomposed operators.
