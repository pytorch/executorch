# ExecuTorch Wasm Build

This guide describes how to build ExecuTorch for WebAssembly (Wasm).

## Quick Start

To quickly test the build, you can run the following commands

```bash
cd executorch # To the top level dir

source .ci/scripts/setup-emscripten.sh # Install Emscripten and set up the environment variables

bash examples/wasm/test_build_wasm.sh # Run the test build script
```

## Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/Tutorial.html)

## Generate Models

JavaScript does not have direct access to the host file system. To load a model, it needs to be preloaded or embedded into the virtual file system. In this example, models in the `./models/` directory are embedded by default. We will then build `executorch_runner` in Wasm.

1. Following the setup guide in [Setting up ExecuTorch](https://pytorch.org/executorch/main/getting-started-setup)
you should be able to get the basic development environment for ExecuTorch working.

2. Using the script `portable/scripts/export.py` generate a model binary file by selecting a
model name from the list of available models in the `examples/models` dir.

```bash
cd executorch # To the top level dir

mkdir models

# To get a list of example models
python3 -m examples.portable.script.export -h

# To generate a specific pte model into the models/ directory
python3 -m examples.portable.scripts.export --model_name="mv2" --output_dir="models/" # for MobileNetv2

# This should generate ./models/mv2.pte file, if successful.
```

Use -h (or --help) to see all the supported models. For the browser example, make sure you have a model with the file name `model.pte` in the `./models/` directory.

3. Once we have the model binaries (.pte) in `./models/`, we can build `executor_runner` in Wasm with Emscripten. When calling `emcmake cmake`, you can pass the `-DWASM_MODEL_DIR=<path>` option to specify the directory containing the model files instead of `./models/`.

```bash
./install_executorch.sh --clean
(mkdir cmake-out-wasm \
    && cd cmake-out-wasm \
    && emcmake cmake -DEXECUTORCH_PAL_DEFAULT=posix ..) \
  && cmake --build cmake-out-wasm -j32 --target executor_runner
```

If you need to rebuild `executor_runner` after modifying the contents of `./models/`, you can run the following command

```bash
cmake --build cmake-out-wasm -j32 --target executor_runner --clean-first
```

4. Run the model with Node.js. Emscripten should come preinstalled with a compatible version of Node.js. If you have an incompatible version of Node.js installed, you can use the Emscripten-provided version by running `$EMSDK_NODE` instead of `node`.

```bash
# Run the tool on the generated model.
node cmake-out-wasm/executor_runner.js --model_path mv2.pte
```

5. You can also run the model in the browser. Note that you cannot pass command line arguments to the browser version of the tool. By default, the program will load the model `model.pte` and run it. Several browsers do not support `file://` XHR requests to load the Wasm file. To get around this, you can use a local web server. For example, with Python:

```bash
python3 -m http.server --directory cmake-out-wasm
```

The page will be available at http://localhost:8000/executor_runner.html.

## Common Issues

### CompileError: WebAssembly.instantiate() [...] failed: expected table index 0...

This seems to be an issue with Node.js v16. Emscripten should come preinstalled with a compatible version of Node.js. You can use the Emscripten-provided version by running `$EMSDK_NODE` instead of `node`.

```bash
echo $EMSDK_NODE
.../emsdk/node/22.16.0_64bit/bin/node # example output
```

### Failed to open [...]: No such file or directory (44)

The file may not have been present while building the Wasm binary. You can rebuild with the following command

```bash
cmake --build cmake-out-wasm -j32 --target executor_runner --clean-first
```

The path may also be incorrect. The files in the `WASM_MODEL_DIR` are placed into the root directory of the virtual file system, so you would use `--model_path mv2.pte` instead of `--model_path models/mv2.pte`, for example.
