# ExecuTorch CoreML Delegate


This subtree contains the CoreML Delegate implementation for ExecuTorch.
CoreML is an optimized framework for running machine learning models on Apple devices. The delegate is the mechanism for leveraging the CoreML framework to accelerate operators when running on Apple devices.

## Layout
- `compiler/` : Lowers a module to CoreML backend.
- `scripts/` : Scripts for installing dependencies and running tests.
- `runtime/`: CoreML delegate runtime implementation.
    - `inmemoryfs`: InMemory filesystem implementation used to serialize/de-serialize AOT blob.
    - `kvstore`: Persistent Key-Value store implementation.
    - `delegate`: Runtime implementation.
    - `include` : Public headers.
    - `tests` :  Tests for CoreML delegate.
    - `workspace` : Xcode workspace for tests.
- `third-party/`: External dependencies.

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please create an issue on [github](https://www.github.com/pytorch/executorch/issues).

## Delegation

For delegating the Program to the **CoreML** backend, the client must be responsible for calling `to_backend` with the **CoreMLBackend** tag.

```python
import executorch.exir as exir
import torch

from executorch.exir.backend.backend_api import to_backend

from executorch.backends.coreml.compiler import CoreMLBackend

class LowerableSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# Convert the lowerable module to Edge IR Representation
to_be_lowered = LowerableSubModel()
example_input = (torch.ones(1), )
to_be_lowered_exir_submodule = exir.capture(to_be_lowered, example_input).to_edge()

# Lower to CoreML backend
lowered_module = to_backend('CoreMLBackend', to_be_lowered_exir_submodule, [])
```

Currently, the **CoreML** backend delegates the whole module to **CoreML**. If a specific op is not supported by the **CoreML** backend then the `to_backend` call would throw an exception. We will be adding a **CoreML Partitioner** to resolve the issue.

The `to_backend` implementation is a thin wrapper over `coremltools`, `coremltools` is responsible for converting an **ExportedProgram** to a **MLModel**. The converted **MLModel** data is saved, flattened, and returned as bytes to **ExecuTorch**.

## Runtime

To execute a **CoreML** delegated **Program**, the client must link to the `coremldelegate` library. Once linked there are no additional steps required, **ExecuTorch** when running the **Program** would call the **CoreML** runtime to execute the **CoreML** delegated part of the **Program**.

Please follow the instructions described in the [CoreML setup](/backends/apple/coreml/setup.md) to link the `coremldelegate` library.
