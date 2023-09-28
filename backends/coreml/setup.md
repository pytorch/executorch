## Setting up CoreML backend

This is a tutorial for setting up the CoreML backend.

## AOT Setup

1. Please follow the instructions described in `setting_up_executorch.md` to set up ExecuTorch AOT.

2. Run `install_requirements.sh` to install dependencies required by the **CoreML** backend.


```
cd executorch

sh backends/coreml/scripts/install_requirements.sh   

``` 

3. There is an example script that delegates a model to the **CoreML** backend. You can run it to quickly validate that the **CoreML** backend is set up correctly.

```
cd executorch

# Saves add_all.pte in the current directory.

python3 -m examples.export.coreml_export_and_delegate -m "add" 

```

4. You can now integrate the **CoreML** backend in code.

```
from executorch.backends.coreml.compiler import CoreMLBackend

# This will delegate the whole program to the CoreML backend. 
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [])

```


## Integrating CoreML delegate.

1. Install buck2

- If you don't have the `zstd` commandline tool, install it with `pip install zstd`.
- Download a prebuilt buck2 archive for your system from [here](https://github.com/facebook/buck2/releases/tag/2023-07-18).
- Decompress with the following command (filename depends on your system)

```bash
# For example, buck2-x86_64-unknown-linux-musl.zst
zstd -cdq buck2-DOWNLOADED_FILENAME.zst > /tmp/buck2 && chmod +x /tmp/buck2
```

You may want to copy the `buck2` binary into your `$PATH` so you can run it as `buck2`.

2. Include the **CoreML** delegate `EXECUTORCH_BUILD_COREML_DELGATE=ON` when building **ExecuTorch**.

```
cd executorch

rm -rf cmake-ios-out

# The delegate supports iOS >= 16.0 and macOS >= 13.0
cmake . -B./cmake-ios-out -G Xcode -DCMAKE_TOOLCHAIN_FILE=backends/coreml/third-party/ios-cmake/ios.toolchain.cmake -DPLATFORM=OS -DDEPLOYMENT_TARGET=16.0 -DEXECUTORCH_BUILD_COREML_DELGATE=ON

```

3. Make sure to add the `coremldelegate` library as a target dependency in the **Build Phases** tab.

 
4. Make sure the target has the following libraries linked in the **Build Phases** tab.
```
- Accelerate.framework
- CoreML.framework
- libcoremldelegate.a
- libexecutorch.a
- libsqlite3.tbd
``` 

4. The target could now run a **CoreML** delegated **Program**. 
  
## CoreML delegate tests set up

1. Build the delegate tests. The build directory is set to `backends/coreml/xcode-test-build`

```
cd executorch

# Builds macOS universal test bundle. 

sh backends/coreml/srcipts/build_coreml_delegate_tests.sh

```

2. Run the tests.


```
cd executorch

# Runs macOS universal test bundle.

sh backends/coreml/srcipts/run_coreml_delegate_tests.sh

```

## CoreML delegate runner set up

1. Build the **CoreML** delegate runner. The build directory is set to `backends/coreml/xcode-runner-build`

```
cd executorch

# Builds the runner app. 

sh backends/coreml/srcipts/build_coreml_delegate_runner.sh

```

2. The executable would be located at `backends/coreml/executorchcoreml_runner`. `executorchcoreml_runner` only runs `Program`s that are delegated to the **CoreML** backend.


```
cd executorch

# Saves add_all.pte in the current directory.

python3 -m examples.export.coreml_export_and_delegate -m "add" 

# Execute the delegated CoreML program.
backends/coreml/executorchcoreml_runner --model_path add_all.pte

```
