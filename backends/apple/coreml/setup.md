# Setting up Core ML backend

This is a tutorial for setting up the Core ML backend.

## AOT Setup

1. Follow the instructions described in [Setting Up ExecuTorch](/docs/source/getting-started-setup.md) to set up ExecuTorch environment.

2. Run `install_requirements.sh` to install dependencies required by the **Core ML** backend.

```
cd executorch

./backends/apple/coreml/scripts/install_requirements.sh

```

3. Run the example script to validate that the **Core ML** backend is set up correctly.

```
cd executorch

# Saves add_coreml_all.pte in the current directory if successful.

python3 -m examples.apple.coreml.scripts.export --model_name add

```

4. You can now integrate the **Core ML** backend in code.

```python
# Delegate to Core ML backend
delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner())
```


## Integrating Core ML delegate into runtime.

1. Follow the instructions described in [Building with CMake](/docs/source/runtime-build-and-cross-compilation.md#building-with-cmake) to set up CMake build system.

2. Install [Xcode](https://developer.apple.com/xcode/).

3. Install Xcode Command Line Tools.

```bash
xcode-select --install
```

4. Build **Core ML** delegate. The following will create `executorch.xcframework` and `coreml_backend.xcframework` in the `cmake-out` directory.

```bash
cd executorch
./build/build_apple_frameworks.sh --coreml
```
5. Open the project in Xcode, and drag `executorch.xcframework` and `coreml_backend.xcframework` frameworks generated from Step 2 to Frameworks.

6. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks:

```
executorch.xcframework
coreml_backend.xcframework
```

5. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks.
```
Accelerate.framework
CoreML.framework
libsqlite3.tbd
```

6. The target could now run a **Core ML** delegated **Program**.
