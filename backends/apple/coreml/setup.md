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
# Lower to Core ML backend
lowered_module = to_backend('CoreMLBackend', to_be_lowered_exir_submodule, [])
```


## Integrating Core ML delegate into runtime.

1. Follow the instructions described in [Building with CMake](/docs/source/runtime-build-and-cross-compilation.md#building-with-cmake) to set up CMake build system.

2. Install [Xcode](https://developer.apple.com/xcode/).

3. Install Xcode Command Line Tools.

```bash
xcode-select --install
```

2. Build **Core ML** delegate. The following will create a `executorch.xcframework` in `cmake-out` directory.

```bash
cd executorch
./build/build_apple_frameworks.sh --Release --coreml
```
3. Open the project in Xcode, and drag the `executorch.xcframework` generated from Step 2 to Frameworks.

4. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks:

```
executorch.xcframework
coreml_backend.xcframework
```

5. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks.
```
- Accelerate.framework
- CoreML.framework
- libsqlite3.tbd
```

6. The target could now run a **Core ML** delegated **Program**.
