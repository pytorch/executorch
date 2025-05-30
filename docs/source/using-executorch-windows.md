# How to Build ExecuTorch for Windows

This document outlines the current known working build instructions for building and validating ExecuTorch on a Windows machine.

This demo uses the
[MobileNet v2](https://pytorch.org/vision/main/models/mobilenetv2.html) model to
process live camera images leveraging the
[XNNPACK](https://github.com/google/XNNPACK) backend.

Note that all commands should be executed on Windows powershell in administrator mode.

## Pre-requisites

### 1. Install Miniconda for Windows
Install miniconda for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).

### 2. Install Git for Windows
Install Git for Windows from the [official website](https://git-scm.com/download/win).

### 3. Install ClangCL for Windows
Install ClangCL for Windows from the [official website](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170).


## Create the Conda Environment
To check if conda is detected by the powershell prompt, try `conda list` or `conda --version`

If conda is not detected, you could run the powershell script for conda named `conda-hook.ps1`.

```bash
$miniconda_dir\\shell\\condabin\\conda-hook.ps1
```
where `$miniconda_dir` is the directory where you installed miniconda
This is `“C:\Users\<username>\AppData\Local”` by default.

### Create and activate the conda environment:
```bash
conda create -yn et python=3.12
conda activate et
```

## Check Symlinks
Set the following environment variable to enable symlinks:
```bash
git config --global core.symlinks true
```

## Set up ExecuTorch
Clone ExecuTorch from the [official GitHub repository](https://github.com/pytorch/executorch).

```bash
git clone --recurse -submodules https://github.com/pytorch/executorch.git
```

## Run the Setup Script

Currently, there are a lot of components that are not buildable on Windows. The below instructions install a very minimal ExecuTorch which can be used as a sanity check.

#### Move into the `executorch` directory
```bash
cd executorch
```

#### (Optional) Run a --clean script prior to running the .bat file.
```bash
./install_executorch.bat --clean
```

#### Run the setup script.
You could run the .bat file or the python script.
```bash
./install_executorch.bat
# OR
# python install_executorch.py
```

## Export MobileNet V2

Create the following script named export_mv2.py

```bash
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT) # This is torch.nn.Module

import torch
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = mv2.eval() # turn into evaluation mode

example_inputs = (torch.randn((1, 3, 224, 224)),) # Necessary for exporting the model

exported_graph = torch.export.export(model, example_inputs) # Core Aten graph

edge = to_edge(exported_graph) # Edge Dialect

edge_delegated = edge.to_backend(XnnpackPartitioner()) # Parts of the graph are delegated to XNNPACK

executorch_program = edge_delegated.to_executorch() # ExecuTorch program

pte_path = "mv2_xnnpack.pte"

with open(pte_path, "wb") as file:
    executorch_program.write_to_file(file) # Serializing into .pte file
```

### Run the export script to create a `mv2_xnnpack.pte` file.

```bash
python .\\export_mv2.py
```

## Build and Install C++ Libraries + Binaries
```bash
del -Recurse -Force cmake-out; `
cmake . `
  -DCMAKE_INSTALL_PREFIX=cmake-out `
  -DPYTHON_EXECUTABLE=C:\Users\nikhi\miniconda3\envs\et\python.exe `
  -DCMAKE_PREFIX_PATH=C:\Users\nikhi\miniconda3\envs\et\Lib\site-packages `
  -DCMAKE_BUILD_TYPE=Release `
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
  -DEXECUTORCH_BUILD_FLATC=ON `
  -DEXECUTORCH_BUILD_PYBIND=OFF `
  -DEXECUTORCH_BUILD_XNNPACK=ON `
  -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON `
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
  -DEXECUTORCH_ENABLE_LOGGING=ON `
  -T ClangCL `
  -Bcmake-out; `
cmake --build cmake-out -j64 --target install --config Release
```

## Run Mobilenet V2 model with XNNPACK delegation

```bash
.\cmake-out\backends\xnnpack\Release\xnn_executor_runner.exe --model_path=.\mv2_xnnpack.pte
```

The expected output would print a tensor of size 1x1000.

```bash
Output 0: tensor(sizes=[1, 1000], [
  -0.50986, 0.30064, 0.0953904, 0.147726, 0.231205, 0.338555, 0.206892, -0.0575775, … ])
```

Congratulations! You've successfully set up ExecuTorch on your Windows device and ran a MobileNet V2 model.
Now, you can explore and enjoy the power of ExecuTorch on your own Windows device!

