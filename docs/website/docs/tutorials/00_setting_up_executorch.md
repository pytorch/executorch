<!--- DEPRECATED This document is moved to executorch/docs/source/getting-started-setup.md---->

# Setting up ExecuTorch

This is a tutorial for building and installing ExecuTorch from the GitHub
repository.

## Ahead-of-Time Setup [(Open on Google Colab)](https://colab.research.google.com/drive/1m8iU4y7CRVelnnolK3ThS2l2gBo7QnAP#scrollTo=1o2t3LlYJQY5)

This will install an `executorch` pip package to your conda environment and
allow you to export your PyTorch model to a flatbuffer file using ExecuTorch.

### Step 1: Clone the ExecuTorch repo

```bash
# Do one of these, depending on how your auth is set up
git clone https://github.com/pytorch/executorch.git
git clone git@github.com:pytorch/executorch.git
```

Ensure that git has fetched and updated the submodules. This is necessary
anytime commit hash of any of the submodules changes. Thus it is safe and
necessary at times to apply this step after you pull changes from upstream.

```bash
cd executorch
git submodule sync
git submodule update --init
```

### Step 2: Set up a dev environment

Install
[Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
and activate a new virtual environment:

```bash
conda create -yn executorch python=3.10.0
conda activate executorch
```

Or alternatively, activate a new Python virtual environment:

```bash
python3 -m venv .executorch
source .executorch/bin/activate
```

### Step 3: Install [Cmake](https://cmake.org/download)

```bash
conda install cmake
```

Alternatively:

```bash
pip install cmake
```

### Step 4: Install `executorch` pip package and dependencies

Install all required python dependencies and PyTorch dependencies.

```bash
# Note: if you are behind a firewall an appropriate proxy server must be setup
# for all subsequent steps.
./install_requirements.sh
```

When getting a new version of the executorch repo (via clone, fetch, or pull),
you may need to re-install a new version the PyTorch nightly pip package. The
`TORCH_VERSION` value in this document will be the correct version for the
corresponsing version of the repo.

### Step 5: Expose FlatBuffers compiler

ExecuTorch uses `flatc` to export models and builds it from sources at
`third-party/flatbuffers`. Make it's available by referring in `$PATH`,
as prompted by the previous step, or exporting as `$FLATC_EXECUTABLE`
enironment variable.
Run `./build/install_flatc.sh` to make sure `flatc` is installed correctly.

## Testing Model Export

Generate a test program file from an `nn.Module` via Python script:

```bash
# Creates the file `add.pte`
python3 -m examples.portable.scripts.export --model_name="add"

# Creates the delegated program `composite_model.pte`, other options are "whole" and "partition"
python3 -m examples.portable.scripts.export_and_delegate --option "composite"
```

Or via Python interpreter:

```python
$ python3
>>> import executorch.exir as exir
>>> from executorch.exir.tests.models import Mul
>>> m = Mul()
>>> print(exir.capture(m, m.get_random_inputs()).to_edge())
>>> open("mul.pte", "wb").write(exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer)
```

Please refer to the [More Examples](./00_setting_up_executorch.md#more-examples)
section for running with more popular models.

## Runtime Setup

Follow
[Ahead-of-Time Setup: Step 1](./00_setting_up_executorch.md#step-1-clone-the-executorch-repo)
above to clone the `executorch` repo if you haven't already.

### Step 1: Install buck2

- If you don't have the `zstd` commandline tool, install it with `pip install zstd`
- Download a prebuilt buck2 archive for your system from the https://github.com/facebook/buck2/releases/tag/2023-07-18 release page. Note that the version is important, and newer or older versions may not work with the version of the buck2 prelude used by the ExecuTorch repo.
- Decompress with the following command (filename depends on your system)

```bash
# For example, buck2-x86_64-unknown-linux-musl.zst or buck2-aarch64-apple-darwin.zst
zstd -cdq buck2-DOWNLOADED_FILENAME.zst > /tmp/buck2 && chmod +x /tmp/buck2
```

You may want to copy the `buck2` binary into your `$PATH` so you can run it as `buck2`.

### Step 2: Build a binary

`executor_runner` is an example wrapper around executorch runtime which includes all the operators and backends

```bash
/tmp/buck2 build //examples/portable/executor_runner:executor_runner --show-output
```

The `--show-output` flag will print the path to the executable if you want to run it directly.

If you run into `Stderr: clang-14: error: invalid linker name in argument '-fuse-ld=lld'`, do
```bash
conda install -c conda-forge lld
```

### Step 3: Run a binary

```bash
# add.pte is the program generated from portable/scripts/export.py during AOT Setup Step 3
/tmp/buck2 run //examples/portable/executor_runner:executor_runner -- --model_path add.pte

# To run a delegated model
/tmp/buck2 run //examples/portable/executor_runner:executor_runner -- --model_path composite_model.pte
```

or execute the binary directly from the `--show-output` path shown when building.

```bash
./buck-out/.../executor_runner --model_path add.pte
```

## More Examples

The [`executorch/examples`](https://github.com/pytorch/executorch/blob/main/examples) directory contains useful examples with a guide to lower and run
popular models like MobileNet V3, Torchvision ViT, Wav2Letter, etc. on ExecuTorch.
