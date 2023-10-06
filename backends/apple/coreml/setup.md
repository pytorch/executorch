## Setting up CoreML backend

This is a tutorial for setting up the CoreML backend.

## AOT Setup

1. Follow the instructions described in [Setting Up ExecuTorch](/docs/source/getting-started-setup.md) to set up ExecuTorch environment.

2. Run `install_requirements.sh` to install dependencies required by the **CoreML** backend.

```
cd executorch

sh backends/apple/coreml/scripts/install_requirements.sh   

``` 

3. Run the example script to validate that the **CoreML** backend is set up correctly. 

```
cd executorch

# Saves add_all.pte in the current directory.

python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name add 

```

4. You can now integrate the **CoreML** backend in code. The following is an example of this flow:

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


## Integrating CoreML delegate into runtime.

1. Follow the instructions described in [Building with CMake](/docs/source/runtime-build-and-cross-compilation.md#building-with-cmake) to set up CMake build system.

2. Install [Xcode](https://developer.apple.com/xcode/).

3. Install Xcode Command Line Tools.

```bash
xcode-select --install
```

2. Build **CoreML** delegate. The following will create a `executorch.xcodeproj` in `cmake-out` directory.

```bash
cd executorch

# Remove build artifacts.
rm -rf cmake-out

# CoreML delegate supports iOS >= 16.0 and macOS >= 13.0.
cmake . -B./cmake-out -G Xcode \
-DCMAKE_TOOLCHAIN_FILE=third-party/pytorch/cmake/iOS.cmake \
-DIOS_PLATFORM=OS \
-DIOS_DEPLOYMENT_TARGET=16.0 \
-DEXECUTORCH_BUILD_COREML_DELEGATE=ON \
-DPYTHON_EXECUTABLE=$(which python3) \
-DFLATC_EXECUTABLE=$(realpath)/third-party/flatbuffers/flatc

```

3. Open the project in Xcode, and drag the `executorch.xcodeproj` generated from Step 2 to Frameworks.

4. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following libraries.

```
libcoremldelegate.a
libexecutorch.a
```

5. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks.
```
- Accelerate.framework
- CoreML.framework
- libsqlite3.tbd
``` 

6. Navigate to the project Build Settings:
- Set the value Header Search Paths to the parent directory of `executorch` directory.
- Set Library Search Paths to `cmake-out/build`.
- In other linker flags, add a custom linker flag -all_load.

7. The target could now run a **CoreML** delegated **Program**. 
