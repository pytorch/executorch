# Setting up Executorch

This is a tutorial for building and installing Executorch from the GitHub repository.

## AOT Setup

This will install an `executorch` pip package to your conda environment and allow you to export your PyTorch model to a flatbuffer file using ExecuTorch.

**Step 1: Set up a conda environment**
```bash
conda create -yn executorch python=3.10.0
conda activate executorch

conda install -c conda-forge flatbuffers

# Install the nightly builds
# Note that if you are behind a firewall an appropriate proxy server must
# be setup for all subsequent steps
pip install --pre torch -i https://download.pytorch.org/whl/nightly/cpu
```

**Step 2: Set up Executorch**. This will install an  `executorch` pip package to your conda environment.
```bash

# Do one of these, depending on how your auth is set up
git clone https://github.com/pytorch/executorch.git
git clone git@github.com:pytorch/executorch.git

# [Runtime requirement] Run the following to get all submodules, only need for runtime setup
git submodule update --init --recursive

pip install .

# cd into a directory that doesn't contain a `./executorch/exir` directory, since
# otherwise python will try using it for `import executorch.exir...` instead of using the
# installed pip package.
cd executorch
```

**Step 3: Try it out**

Via python script:
```bash
# Creates the file `add.ff`
python ~/src/executorch/examples/export/export_example.py -m "add"
```

Or via python interpreter:
```python
(executorch) ~/  $ python
>>> import executorch.exir as exir
>>> from executorch.exir.tests.models import Mul
>>> m = Mul()
>>> print(exir.capture(m, m.get_random_inputs()).to_edge())
>>> exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer
>>> open("add.ff", "wb").write(exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer)
```

## Runtime Setup

**Step 1: Install buck2**

- If you don't have the `zstd` commandline tool, install it with `pip install zstd`
- Download a prebuilt buck2 archive for your system from https://github.com/facebook/buck2/releases/tag/2023-07-18
- Decompress with the following command (filename will need to change for non-Linux systems).

```
zstd -cdq buck2-x86_64-unknown-linux-musl.zst > /tmp/buck2 && chmod +x /tmp/buck2
```

You may want to copy the `buck2` binary into your `$PATH` so you can run it as `buck2`.

**Step 2: Build a binary**

`size_test_all_ops` is a binary including all the operators and backends

```bash
/tmp/buck2 build //test:size_test_all_ops --show-output
```

The `--show-output` flag will print the path to the executable if you want to run it directly.

**Step 3: Run a binary**

```bash
# add.ff is the program generated from export_example.py during AOT Setup Step 3
/tmp/buck2 run //test:size_test_all_ops  -- add.ff
```

or execute the binary directly from the `--show-output` path shown when building.

```bash
./buck-out/.../size_test_all_ops add.ff
```
