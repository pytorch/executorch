# Example export script for the ExecuTorch Vulkan backend

This directory contains `export.py`, a utility script that can be used to export
models registered in [`executorch/examples/models/__init__.py`](https://github.com/pytorch/executorch/blob/main/examples/models/__init__.py)
to the Vulkan backend.

## Usage

Note that all example commands are assumed to be executed from executorch root.

```shell
cd ~/executorch
```

### Basic Export

For example, to export MobileNet V2:

```shell
MODEL_NAME=mv2 && \
OUTPUT_DIR=. && \
python -m examples.vulkan.export -m ${MODEL_NAME} -o ${OUTPUT_DIR}
```

This will create a file name `mv2_vulkan.pte` in the specified output directory.

### With dynamic shape support

To enable exporting with dynamic shapes, simply add the `-d` flag.

```shell
MODEL_NAME=mv2 && \
OUTPUT_DIR=. && \
python -m examples.vulkan.export -m ${MODEL_NAME} -o ${OUTPUT_DIR} -d
```

### Export a bundled pte

Use the `-b` flag to export a bundled PTE file (i.e. `.bpte`). This is a `.pte`
file with bundled test cases that can be used for correctness checking.

```shell
MODEL_NAME=mv2 && \
OUTPUT_DIR=. && \
python -m examples.vulkan.export -m ${MODEL_NAME} -o ${OUTPUT_DIR} -d -b
```

This will create a file called `mv2_vulkan.bpte` in the specified output directory.

### With correctness testing

The script can also execute the exported and lowered model via pybindings to
check output correctness before writing the output file.

To enable this, ensure that your machine:

1. Has the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#android) installed
2. Has Vulkan drivers

Additionally, you will need to install the executorch python package from
source, since the Vulkan backend is not included by default in the pip package.

```shell
CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON " ./install_executorch.sh -e
```

Once these conditions are fulfilled, the `--test` flag can be passed to the
script.

```shell
MODEL_NAME=mv2 && \
OUTPUT_DIR=. && \
python -m examples.vulkan.export -m ${MODEL_NAME} -o ${OUTPUT_DIR} -d --test
```

You should see some output like

```shell
INFO:root:✓ Model test PASSED - outputs match reference within tolerance
```

### Quantization support

Support for quantization is under active development and will be added soon!
