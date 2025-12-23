# Preparing a Model for {BACKEND_NAME}

This is a placeholder tutorial.

## Step 1: Environment Setup

This tutorial is intended to be run from a {SUPPORTED_HOST_OS} and uses Conda for Python environment management. For full setup details and system requirements, see [Getting Started with ExecuTorch](/getting-started).

Create a Conda environment and install the ExecuTorch Python package.
```bash
conda create -y --name executorch python=3.12
conda activate executorch
conda install executorch
```

{ADDITIONAL_SETUP_STEPS}

## Step 2: Model Preparation

Create a python file named `export_{model_filename}.py`. This script will be responsible for loading the {EXAMPLE_MODEL} model from {MODEL_SOURCE} and create a {BACKEND_NAME}-targeted .pte file.

```py
# export_{model_filename}.py
from executorch.backends.{backend_name}.partition.{backend_name}_partitioner import {BackendName}Partitioner
from executorch.exir import to_edge_transform_and_lower
import torch
import {MODEL_IMPORT}
```

### Model Instantiation and Example Inputs

Instantiate the {EXAMPLE_MODEL} model from [{MODEL_SOURCE}]({MODEL_SOURCE_URL}). The export process also needs an example model input to trace the model. The model takes {MODEL_INPUT_DESCRIPTION}, so we'll create {INPUT_TUPLE_DESCRIPTION}.
```py
model = {MODEL_INSTANTIATION_CODE}
example_inputs = ({EXAMPLE_INPUTS},)
```

### Lower the Model

Next, export and lower the model to ExecuTorch. Note that the `{BackendName}Partitioner` passed to the `partitioner` parameter tells ExecuTorch to target the {BACKEND_NAME} backend.
```py
exported_program = torch.export.export(model, example_inputs)

executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[{BackendName}Partitioner()],
).to_executorch()

executorch_program.save("{model_filename}_{backend_name}.pte")
```

### Run the Script

Save the above script to export_{model_filename}.py and run the script. You should see a file named `{model_filename}_{backend_name}.pte` in the current directory.
```bash
python export_{model_filename}.py
```

## Step 3: Running the Model

The .pte file created in the previous step can be run on a variety of devices, including {SUPPORTED_PLATFORMS}. ExecuTorch provides runtime APIs and language bindings for a variety of platforms. This tutorial will demonstrate running the model on a desktop using the Python runtime.

### Smoke Test

First, we'll verify that the model loads and runs correctly by running the model with {TEST_INPUT_DESCRIPTION}. Create a new script, named `run_{model_filename}.py`, and add the following code.
```py
# run_{model_filename}.py

from executorch.runtime import Runtime
import torch

runtime = Runtime.get()

input_tensor = {TEST_INPUT_TENSOR}
program = runtime.load_program("{model_filename}_{backend_name}.pte")
method = program.load_method("forward")
outputs = method.execute([input_tensor])[0]

print(outputs)
```

When running the script with `python run_{model_filename}.py`, you should see {EXPECTED_OUTPUT_DESCRIPTION} printed to the console.
```
{EXPECTED_OUTPUT_EXAMPLE}
```

# Next Steps

 - See [Edge Platforms](/edge-platforms-section) to deploy the .pte file on {SUPPORTED_PLATFORMS}.
 - See [Model Export and Lowering](/using-executorch-export) for more information on model preparation.
 - See [{BACKEND_NAME} Overview](/backends/{backend_name}/{backend_name}-overview) for more information about the {BACKEND_NAME} backend. <!-- @lint-ignore placeholder link -->
