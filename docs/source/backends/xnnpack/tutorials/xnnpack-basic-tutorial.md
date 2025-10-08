# Preparing a Model

This tutorial demonstrates the creation of an ExecuTorch .pte file for the MobileNet V3 Small model using the XNNPACK backend. This .pte file can be run on a variety of devices, including Android, iOS, and desktop.

## Step 1: Environment Setup

This tutorial is intended to be run from a Mac or Linux host and uses Conda for Python environment management. For full setup details and system requirements, see [Getting Started with ExecuTorch](/getting-started).

Create a Conda environment and install the ExecuTorch Python package.
```bash
conda create -y --name executorch python=3.12
conda activate executorch
conda install executorch
```

## Step 2: Model Preparation

Create a python file named `export_mv3.py`. This script will be responsible for loading the MobileNet V3 model from torchvision and create an XNNPACK-targeted .pte file.

```py
# export_mv3.py
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import torch
import torchvision
```

### Model Instantiation and Example Inputs

Instantiate the MobileNet V3 Small model from [torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small). The export process also needs an example model input to trace the model. The model takes a single tensor, so we'll create a single-element tuple with a tensor of size (1,3,224,224), matching the size of the input we'll provide at runtime.
```py
model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
example_inputs = (torch.randn(1,3,224,224),)
```

### Lower the Model

Next, export and lower the model to ExecuTorch. Note that the `XnnpackPartitioner` passed to the `partitioner` parameter tells ExecuTorch to target the XNNPACK backend.
```py
exported_program = torch.export.export(model, example_inputs)

executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

executorch_program.save("mv3_xnnpack.pte")
```

### Run the Script

Save the above script to export_mv3.py and run the script. You should see a file named `mv3_xnnpack.pte` in the current directory.
```bash
python export_mv3.py
```

## Step 3: Running the Model

The .pte file created in the previous step can be run on a variety of devices, including Android, iOS, and desktop. ExecuTorch provides runtime APIs and language bindings for a variety of platforms. This tutorial will demonstrate running the model on a desktop using the Python runtime.

### Smoke Test

First, we'll verify that the model loads and runs correctly by running the model with an all ones input tensor. Create a new script, named `run_mv3.py`, and add the following code.
```py
# run_mv3.py

from executorch.runtime import Runtime
import torch

runtime = Runtime.get()

input_tensor = torch.ones(1, 3, 224, 224)
program = runtime.load_program("mv3_xnnpack.pte")
method = program.load_method("forward")
outputs = method.execute([input_tensor])[0]

print(outputs)
```

When running the script with `python run_mv3.py`, you should see a tensor of size (1, 1000) printed to the console.
```
tensor([[-2.9747e-02, -1.1634e-01,  2.3453e-01, -1.1516e-01,  2.8407e-01,
          1.3327e+00, -1.2022e+00, -4.1820e-01, -8.6148e-01,  9.6264e-01,
          2.0528e+00,  3.2284e-02, -6.7234e-01, -1.3766e-01, -7.8548e-01,
          ...
       ]])
```


# Next Steps

 - See [Edge Platforms](/edge-platforms-section) to deploy the .pte file on Android, iOS, or other platforms.
 - See [Model Export and Lowering](/using-executorch-export) for more information on model preparation.
 - See [XNNPACK Overview](/backends/xnnpack/xnnpack-overview) for more information about the XNNPACK backend.
