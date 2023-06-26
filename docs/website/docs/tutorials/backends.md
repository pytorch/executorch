# Delegate a PyTorch module to Executorch runtime

This note is to demonstrate the basic end-to-end flow of backend delegation in new Executorch runtime. Please refer to the [design doc]( https://docs.google.com/document/d/1YTn1jxdOsY5EezmdiAJYTJPz6EOGY93DSPIpgCWTowc/edit#heading=h.q866lyxainnb). The new delegate workflow can be found in [N1820138](https://www.internalfb.com/intern/anp/view/?id=1820138), and the previous delegate with lite interpreter workflow can be found in [N509022](https://www.internalfb.com/intern/anp/view/?id=509022)

Please note that both Executorch runtime and the new delegate workflow is under development. This note currently covers basic capabilities of this flow.

If your team has an existing delegate workflow, the new flow will be very similar to the old ones. Please reach out to PyTorch Edge team for further discusion.

### Ahead of Time
The flow starts from a traced graph module with Edge IR representation. Then it goes through the preprocess that produces a program with compiled blobs. The lowered module can be directly captured, or be put back in a parent module to be captured. Eventually the captured module is serialized in the flatbuffers model that can be loaded by the runtime.

### Runtime
The serialized flatbuffer model can be loaded by the Executorch runtime. The preprocessed blob is loaded and the backend's init() function is called at the model initialization stage. At the model execution stage, the initialized handled can be executed through the backend.

Using a demo backend, following codes show the three steps to integrate a backend to the flow:

1. Add your backend to Executorch.

2. Frontend: lower the PyTorch module to a backend.

3. Deployment: load and run the lowered module through Executorch runtime interface.

## Python API
The Python API for this backend will be automatically ported when this shared lib is loaded into PyTorch. The following sections show how to use this backend from Python directly.

#### Frontend  end
In this section, a Pytorch Model is created, captured and lowered to `backend_with_compiler_demo`. First we define the module in eager mode. The module can be lowered via the api `to_backend`.

# Example user flow

Following is to show how user can author a model and deploy the model in executor:

```
import executorch.exir as exir

import torch
from executorch.backends.backend_details import to_backend

from executorch.exir import ExecutorchBackendConfig, EdgeCompileConfig
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from torch.fx import symbolic_trace
from executorch.pybindings.portable import _load_for_executorch, _load_for_executorch_from_buffer   # @manual
```

If the whole module in eager mode looks like:

```python
# the submodule runs in executor runtime
class NonLowerableSubModel(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, a, b):
        return torch.add(torch.add(a, b), self.bias)


# the submodule runs in a specific backend. In this example,  `BackendWithCompilerDemo` backend
class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# the composite modules, including lower part and non-lowerpart
class CompositeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.non_lowerable = NonLowerableSubModel(torch.ones(1) * 0.3)
        self.lowerable = LowerableSubModel()

    def forward(self, x):
        a = self.lowerable(x)
        b = self.lowerable(a)
        ret = self.non_lowerable(a, b)
        return a, b, ret
```
And we want to lower `LowerableSubModel` to a specific backend

### The submodule to be lowered.
```python
class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
# sin_module is an nn.Module
to_be_lowered = LowerableSubModel()
example_input = (torch.ones(1), )
to_be_lowered_exir_submodule = exir.capture(to_be_lowered, example_input).to_edge()
```
### Lower the module to the according backend

```python
# import the backend implementation
from executorch.backends.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
method_compile_spec = bytes('{"forward": ""}', encoding="utf8")
lowered_module = to_backend('BackendWithCompilerDemo', to_be_lowered_exir_submodule, method_compile_spec)

# Prepare the model inputs to get exir
model_inputs = (torch.ones(1), )
lowered_sin_graph_module = exir.capture(lowered_module, model_inputs).to_edge()
```
### Composite the module with the lowered module

```python
class NonLowerableSubModel(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, a, b):
        return torch.add(torch.add(a, b), self.bias)

class CompositeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.non_lowerable = NonLowerableSubModel(torch.ones(1) * 0.3)
        self.lowerable = lowered_sin_graph_module

    def forward(self, x):
        a = self.lowerable(x)
        b = self.lowerable(a)
        ret = self.non_lowerable(a, b)
        return a, b, ret

composite_model = CompositeModel()
```

### Run passes, capture and emit the program
```

# Run somes passes. The passes depends on the targets.
# Customized passes can be used in EdgeCompileConfig and ExecutorchBackendConfig
exec_prog = exir.capture(composite_model, model_inputs).to_edge().to_executorch()
graph_module = exec_prog
program = exec_prog.program
flatbuffer = exec_prog.buffer

print("Output trace through graph result")
print(graph_module.graph.print_tabular())

# Uncomment the following lines to save the program to a local file
# save_path = "delegate.fft"
# with open(save_path, "wb") as f:
#     f.write(flatbuffer)
```

# Runtime
### Run the real model with executor
```python
# Load the program with executor runtime
executorch_module = _load_for_executorch_from_buffer(flatbuffer)
print("model_inputs: ", model_inputs)
# Execute the program
model_outputs = executorch_module.forward([*model_inputs])
```

## Error Messages

If there is an error in the backend, for example, if there is any operator that is not supported by the backend, debug handler can be thrown. It can surface back to the Python frontend with source code information. Below is an example where operator `tan` is not supported in `BackendWithCompilerDemo` backend.

A problematic program:
```python
class TanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # TODO(chenlai): add a test with a diffrent method name when
    # it's resolved in compiler side.
    def forward(self, x):
        return torch.tan(x)

tan_module = TanModule()
model_inputs = (torch.ones(1),)
edgeir_m = exir.capture(tan_module, model_inputs).to_edge()
method_compile_spec = bytes('{"forward": ""}', encoding="utf8")
lowered_tan_module = to_backend(
    "BackendWithCompilerDemo", edgeir_m, method_compile_spec
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

It's expected to capture debug handler like `instruction demo::tan_default<debug_handle>1 is not supported, debug handler is: 1`
