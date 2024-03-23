<!---- DO NOT MODIFY Progress Bar Start --->

<div class="progress-bar-wrapper">
   <div class="progress-bar-item">
     <div class="step-number" id="step-1">1</div>
     <span class="step-caption" id="caption-1"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-2">2</div>
     <span class="step-caption" id="caption-2"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-3">3</div>
     <span class="step-caption" id="caption-3"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-4">4</div>
     <span class="step-caption" id="caption-4"></span>
   </div>
</div>

<!---- DO NOT MODIFY Progress Bar End--->

# Setting Up ExecuTorch

This tutorial walks you through an end-to-end example of configuring your
environment for ExecuTorch, installing ExecuTorch,
exporting your model, and finally building and running a runtime.

::::{grid} 2
:::{grid-item-card}  What you will learn
:class-card: card-prerequisites
* How to set up your environment to work on ExecuTorch
* How to generate a sample ExecuTorch program
* How to build and run an ExecuTorch runtime
:::
:::{grid-item-card} Prerequisites
:class-card: card-prerequisites
* Understand the concepts defined in [ExecuTorch Overview](intro-overview.md)
:::
::::

## Supported Host Environments

We have tested these instructions on the following systems, although they should
also work on other systems with similar environments.

**Linux (x86_64)**
- CentOS 8 or later
- Ubuntu 20.04.6 LTS or later
- RHEL 8 or later
- Windows Subsystem for Linux running any of the above

**macOS (x86_64/M1/M2)**
- Big Sur or later

The most critical requirements are:
- The ability to install a recent version of `conda`, described below.
- `g++` version 8 or higher, `clang++` version 8 or higher, or another
  C++17-compatible toolchain that supports GNU C-style [statement
  expressions](https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html) (`({ ...
  })` syntax).

Note that the cross-compilable core runtime code supports a wider range of
toolchains, down to C++11. See the [Runtime Overview](./runtime-overview.md) for
portability details.

## Set up Your Environment

Before you can start working with ExecuTorch, you'll need to set up your
environment. This is an important step to ensure that everything runs
smoothly and efficiently. We recommend using `conda` to create and
manage your virtual environment. Conda is a package management system
and environment manager for Python and other programming languages,
which is built on top of the Python package manager pip, and provides
a more convenient and flexible way to manage packages and environments.
In this section, you will set up your `conda` environment and install
the required dependencies.

Follow these steps:

1. If you do not have it already, install conda on your machine by following the
   steps in the
   [conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Clone the `executorch` repository:

   ```bash
   git clone https://github.com/pytorch/executorch.git
   ```

1. Update the submodules:

   ```bash
   cd executorch
   git submodule sync
   git submodule update --init
   ```

1. Create and activate your conda environment:

   ```bash
   conda create -yn executorch python=3.10.0
   conda activate executorch
   ```

   Or alternatively use a Python virtual environment:

   ```bash
   python3 -m venv .executorch
   source .executorch/bin/activate
   ```

1. Install [Cmake](https://cmake.org/download)

   ```bash
   conda install cmake
   ```

   Alternatively:

   ```bash
   pip install cmake
   ```

1. Install ExecuTorch and dependencies:

   ```bash
   ./install_requirements.sh
   ```

   **Optional:** Install ExecuTorch as an editable installation:
   ```bash
   pip install --editable . --config-settings editable_mode=strict --no-build-isolation
   ```

1. Expose FlatBuffers compiler:

   ExecuTorch uses `flatc` to export models and builds it from sources at
   `third-party/flatbuffers`. Make it available by adding it to the `$PATH` environment variable,
   as prompted by the previous step, or exporting as `$FLATC_EXECUTABLE`
   enironment variable.
   Run `./build/install_flatc.sh` to make sure `flatc` is installed correctly.

You have successfully set up your environment to work with ExecuTorch. The next
step is to generate a sample ExecuTorch program.

## Generate a Sample ExecuTorch program

After you set up your environment, you are ready to convert your programs
into ExecuTorch programs. You will need to use `torch.export` and the
`executorch.exir` to export your program. Then, you can save your program as
a `.pte` file, which is the file extension ExecuTorch expects. To demonstrate
how to do it, we will generate an ExecuTorch program file from an `nn.Module`.

You can generate an ExecuTorch program by using a sample script or by using
the Python interpreter.

We have created the `export.py` script that demonstrates a simple model
export to flatbuffer. This script is available
in the [pytorch/executorch](https://github.com/pytorch/executorch/tree/main/examples/portable)
repository.

To generate a sample program, complete the following steps:

1. Run the `export.py` script:

  ```bash
  python3 -m examples.portable.scripts.export --model_name="add"
  ```

  :::{dropdown} Output
  ```
  Exported graph:
   graph():
     %arg0_1 : [num_users=3] = placeholder[target=arg0_1]
      %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
      %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
      %aten_add_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor, %arg0_1), kwargs = {})
      %aten_add_tensor_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_1, %arg0_1), kwargs = {})
      %aten_add_tensor_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_2, %aten_add_tensor_2), kwargs = {})
      return (aten_add_tensor_3,)
  Saving exported program to add.pte
  ```
  :::

  This command has created a `add.pte` file that contains your sample program,
  which adds its inputs multiple times.

Alternatively, you can use a Python interpreter to perform similar steps, this
time creating a `mul.pte` program file that multiplies its inputs:

```python
import executorch.exir as exir
from executorch.exir.tests.models import Mul
m = Mul()
open("mul.pte", "wb").write(to_edge(export(m, m.get_random_inputs())).to_executorch().buffer)
```

In this step, you learned how you can export your PyTorch program to an ExecuTorch
program. You can apply the same principle to your own PyTorch programs.

The next step is to run your program by setting up Buck2 and building an
`executor_runner`.

## Building a Runtime

After you have exported your program, you are almost ready to run it.
The next step involves using Buck2 to build a runtime.

**Buck2** is an open-source build system that enables developers to manage
project dependencies easily and efficiently. We will use Buck2 to build the
`executor_runner`, a sample wrapper for the ExecuTorch runtime which includes
all the operators and backends.

You will need the following prerequisits for this section:

* The `zstd` command line tool — install by running
   ```bash
   pip3 install zstd
   ```
* Version `${executorch_version:buck2}` of the `buck2` commandline tool — you can download a
  prebuilt archive for your system from [the Buck2
  repo](https://github.com/facebook/buck2/releases/tag/2024-02-15). Note that
  the version is important, and newer or older versions may not work with the
  version of the buck2 prelude used by the ExecuTorch repo.

Complete the following steps:

1. Ensure that Git has fetched and updated the submodules. This is necessary
   whenever the commit hash of a submodule changes. Therefore, you need
   to periodically sync your submodules with upstream:

   ```bash
   cd executorch
   git submodule sync
   git submodule update --init
   ```

1. Install ExecuTorch and dependencies:

   ```bash
   ./install_requirements.sh
   ```

1. Configure Buck2 by decompressing with the following command (filename depends
   on your system):

   ```bash
   # For example, buck2-x86_64-unknown-linux-musl.zst or buck2-aarch64-apple-darwin.zst
   zstd -cdq buck2-DOWNLOADED_FILENAME.zst > /tmp/buck2 && chmod +x /tmp/buck2
   ```

   You may want to copy the `buck2` binary into your `$PATH` so you can run it
   as `buck2`.

1. Build a binary:

   ```bash
   /tmp/buck2 build //examples/portable/executor_runner:executor_runner --show-output
   ```

   :::{dropdown} Output

   ```
   File changed: root//.git/config.lock
   File changed: root//.git/config
   File changed: root//.git/modules
   27036 additional file change events
   Build ID: e725eb0d-f4a1-484e-b0d3-8133d67b6fdd
   Network: Up:   0 B              Down: 670 KiB
   Command: build.                 Remaining: 340/954. Cache hits: 0%. Time elapsed: 13.2s
   …
   Cache hits: 0%. Commands: 376 (cached: 0, remote: 0, local: 376)
   BUILD SUCCEEDED
   ```
   :::

   The `--show-output` flag prints the path to the executable if you want to run it directly.

   If you run into `Stderr: clang-14: error: invalid linker name in argument
   '-fuse-ld=lld'`, `lld` is not available on your system. Try installing it
   with `conda` or with your system's package manager.
   ```bash
   conda install -c conda-forge lld
   ```

Now that you have built our sample programs, you can proceed to
run them.

## Run Your Program

After you build your program, you are ready to run it. We will use
the `buck run` command to run our program.

1. Run the binary:

   * To run the `add.pte` program:

     ```bash
     /tmp/buck2 run //examples/portable/executor_runner:executor_runner -- --model_path add.pte
     ```

     :::{dropdown} Sample Output

     ```
     Build ID: 4a23602b-25ba-4b95-a212-3cd077136062
     Network: Up: 0 B  Down: 0 B
     Jobs completed: 3. Time elapsed: 0.0s.
     I 00:00:00.005837 executorch:executor_runner.cpp:75] Model file add.pte is loaded.
     I 00:00:00.005852 executorch:executor_runner.cpp:85] Running method forward
     I 00:00:00.005860 executorch:executor_runner.cpp:140] Setting up non-const buffer 1, size 48.
     I 00:00:00.005909 executorch:executor_runner.cpp:181] Method loaded.
     I 00:00:00.005913 executorch:util.h:104] input already initialized, refilling.
     I 00:00:00.005915 executorch:util.h:104] input already initialized, refilling.
     I 00:00:00.005917 executorch:executor_runner.cpp:186] Inputs prepared.
     I 00:00:00.005949 executorch:executor_runner.cpp:195] Model executed successfully.
     I 00:00:00.005954 executorch:executor_runner.cpp:210] 8.000000
     ```
     :::

Alternatively, you can execute the binary directly from the `--show-output` path
shown in the build step. For example, you can run the following command for the
`add.pte` program:

```bash
./buck-out/.../executor_runner --model_path add.pte
```

## Next Steps

Congratulations! You have successfully exported, built, and run your first
ExecuTorch program. Now that you have a basic understanding of how ExecuTorch
works, you can start exploring its advanced features and capabilities. Here
is a list of sections you might want to read next:

* [Exporting a model](export-overview.md)
* Using [EXIR](ir-exir.md) for advanced exports
* Review more advanced examples in the [executorch/examples](https://github.com/pytorch/executorch/tree/main/examples) directory
