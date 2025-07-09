# ExecuTorch Wasm Build

This guide describes how to build ExecuTorch for WebAssembly (Wasm).

## Directory Structure

```
examples/wasm
└── README.md                         # This file
```

## Prerequisites

- [emscripten](https://emscripten.org/docs/getting_started/Tutorial.html)

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

4. Run the model with Node.js (automatically installed with Emscripten).

```bash
# Run the tool on the generated model.
node cmake-out-wasm/executor_runner.js --model_path mv2.pte
```

5. You can also run the model in the browser. Note that you cannot pass command line arguments to the browser version of the tool. By default, the program will load the model `model.pte` and run it. Several browsers do not support `file://` XHR requests to load the Wasm file. To get around this, you can use a local web server. For example, with Python:

```bash
python3 -m http.server --directory cmake-out-wasm
```

The page will be available at http://localhost:8000/executor_runner.html.
