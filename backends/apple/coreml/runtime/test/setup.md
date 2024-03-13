## Core ML delegate tests set up

This is a tutorial for setting up tests for the **Core ML** backend.

## Running tests

1. Follow the instructions described in [Setting Up ExecuTorch](/docs/source/getting-started-setup.md) to set up ExecuTorch environment.

2. Run `install_requirements.sh` to install dependencies required by the **Core ML** backend.

```bash
cd executorch

sh backends/apple/coreml/scripts/install_requirements.sh   

``` 

3. Follow the instructions described in [Building with CMake](/docs/source/runtime-build-and-cross-compilation.md#building-with-cmake) to set up CMake build system.

4. Install [Xcode](https://developer.apple.com/xcode/).

5. Install Xcode Command Line Tools.

6. Run `build_tests.sh` to build tests.

```bash
cd executorch

# Builds macOS universal test bundle. 

sh backends/apple/coreml/srcipts/build_tests.sh

```

7. Run `run_tests.sh` to execute the tests.

```
cd executorch

sh backends/apple/coreml/srcipts/run_tests.sh

```
 
## Updating tests

1. Open the Xcode workspace.

```bash
cd executorch

# Builds macOS universal test bundle. 

open backends/apple/coreml/runtime/workspace/executorchcoreml.xcworkspace

```

2. The runtime tests are in the `test` folder, after updating the tests you can use the Xcode tests navigator to run the tests or use the command line.

```bash
cd executorch

# There is no need to build the tests.
sh backends/apple/coreml/srcipts/run_tests.sh

```