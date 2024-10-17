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
In this section, we'll learn how to
* Set up an environment to work on ExecuTorch
* Generate a sample ExecuTorch program
* Build and run a program with the ExecuTorch runtime

## System Requirements
### Operating System

We've tested these instructions on the following systems, although they should
also work in similar environments.


Linux (x86_64)
- CentOS 8+
- Ubuntu 20.04.6 LTS+
- RHEL 8+

macOS (x86_64/M1/M2)
- Big Sur (11.0)+

Windows (x86_64)
- Windows Subsystem for Linux (WSL) with any of the Linux options

### Software
* `conda` or another virtual environment manager
  - We recommend `conda` as it provides cross-language
    support and integrates smoothly with `pip` (Python's built-in package manager)
  - Otherwise, Python's built-in virtual environment manager `python venv` is a good alternative.
* `g++` version 7 or higher, `clang++` version 5 or higher, or another
  C++17-compatible toolchain.

Note that the cross-compilable core runtime code supports a wider range of
toolchains, down to C++17. See the [Runtime Overview](./runtime-overview.md) for
portability details.

## Quick Setup: Colab/Jupyter Notebook Prototype

To utilize ExecuTorch to its fullest extent, please follow the setup instructions provided below to install from source.

Alternatively, if you would like to experiment with ExecuTorch quickly and easily, we recommend using the following [colab notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing) for prototyping purposes. You can install directly via `pip` for basic functionality.
  ```bash
  pip install executorch
  ```


## Environment Setup

### Create a Virtual Environment

[Install conda on your machine](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Then, create a virtual environment to manage our dependencies.
   ```bash
   # Create and activate a conda environment named "executorch"
   conda create -yn executorch python=3.10.0
   conda activate executorch
   ```

### Clone and install ExecuTorch requirements

   ```bash
   # Clone the ExecuTorch repo from GitHub
   git clone -b release/0.4 https://github.com/pytorch/executorch.git
   cd executorch

   # Update and pull submodules
   git submodule sync
   git submodule update --init

   # Install ExecuTorch pip package and its dependencies, as well as
   # development tools like CMake.
   # If developing on a Mac, make sure to install the Xcode Command Line Tools first.
   ./install_requirements.sh
   ```

   Use the [`--pybind` flag](https://github.com/pytorch/executorch/blob/main/install_requirements.sh#L26-L29) to install with pybindings and dependencies for other backends.
   ```bash
   ./install_requirements.sh --pybind <coreml | mps | xnnpack>
   ```
After setting up your environment, you are ready to convert your PyTorch programs
to ExecuTorch.

> **_NOTE:_**  Cleaning the build system
>
> When fetching a new version of the upstream repo (via `git fetch` or `git
> pull`) it is a good idea to clean the old build artifacts. The build system
> does not currently adapt well to changes in build dependencies.
>
> You should also update and pull the submodules again, in case their versions
> have changed.
>
> ```bash
> # From the root of the executorch repo:
> rm -rf cmake-out pip-out
> git submodule sync
> git submodule update --init
> ```

## Create an ExecuTorch program

After setting up your environment, you are ready to convert your PyTorch programs
to ExecuTorch.

### Export a Program
ExecuTorch provides APIs to compile a PyTorch [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) to a `.pte` binary consumed by the ExecuTorch runtime.
1. [`torch.export`](https://pytorch.org/docs/stable/export.html)
1. [`exir.to_edge`](https://pytorch.org/executorch/stable/export-to-executorch-api-reference.html#exir.to_edge)
1. [`exir.to_executorch`](ir-exir.md)
1. Save the result as a [`.pte` binary](pte-file-format.md) to be consumed by the ExecuTorch runtime.


Let's try this using with a simple PyTorch model that adds its inputs.

Create `export_add.py` in a new directory outside of the ExecuTorch repo.

**Note: It's important that this file does does not live in the directory that's a parent of the `executorch` directory. We need python to import from site-packages, not from the repo itself.**

```
mkdir -p ../example_files
cd ../example_files
touch export_add.py
```

Add the following code to `export_add.py`:
```python
import torch
from torch.export import export
from executorch.exir import to_edge

# Start with a PyTorch model that adds two input tensors (matrices)
class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()

  def forward(self, x: torch.Tensor, y: torch.Tensor):
      return x + y

# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("add.pte", "wb") as file:
    file.write(executorch_program.buffer)

```

Then, execute it from your terminal.
```bash
python3 export_add.py
```

If it worked you'll see `add.pte` in that directory

See the [ExecuTorch export tutorial](tutorials_source/export-to-executorch-tutorial.py) to learn more about the export process.


## Build & Run

After creating a program go back to the executorch directory to execute it using the ExecuTorch runtime.
```
cd ../executorch
```

For now, let's use [`executor_runner`](https://github.com/pytorch/executorch/blob/main/examples/portable/executor_runner/executor_runner.cpp), an example that runs the `forward` method on your program using the ExecuTorch runtime.

### Build Tooling Setup
The ExecuTorch repo uses CMake to build its C++ code. Here, we'll configure it to build the `executor_runner` tool to run it on our desktop OS.
  ```bash
  # Clean and configure the CMake build system. Compiled programs will
  # appear in the executorch/cmake-out directory we create here.
  (rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)

  # Build the executor_runner target
  cmake --build cmake-out --target executor_runner -j9
  ```

> **_NOTE:_**  Cleaning the build system
>
> When fetching a new version of the upstream repo (via `git fetch` or `git
> pull`) it is a good idea to clean the old build artifacts. The build system
> does not currently adapt well to changes in build dependencies.
>
> You should also update and pull the submodules again, in case their versions
> have changed.
>
> ```bash
> # From the root of the executorch repo:
> rm -rf cmake-out pip-out
> git submodule sync
> git submodule update --init
> ```

### Run Your Program

Now that we've exported a program and built the runtime, let's execute it!

  ```bash
  ./cmake-out/executor_runner --model_path ../example_files/add.pte
  ```
Our output is a `torch.Tensor` with a size of 1. The `executor_runner` sets all input values to a [`torch.ones`](https://pytorch.org/docs/stable/generated/torch.ones.html) tensor, so when `x=[1]` and `y=[1]`, we get `[1]+[1]=[2]`
  :::{dropdown} Sample Output

  ```
Output 0: tensor(sizes=[1], [2.])
  ```
  :::

To learn how to build a similar program, visit the [Runtime APIs Tutorial](extension-module.md).

## Next Steps

Congratulations! You have successfully exported, built, and run your first
ExecuTorch program. Now that you have a basic understanding of ExecuTorch,
explore its advanced features and capabilities below.

* Build an [Android](demo-apps-android.md) or [iOS](demo-apps-ios.md) demo app
* Learn more about the [export process](export-overview.md)
* Dive deeper into the [Export Intermediate Representation (EXIR)](ir-exir.md) for complex export workflows
* Refer to [advanced examples in executorch/examples](https://github.com/pytorch/executorch/tree/main/examples)
