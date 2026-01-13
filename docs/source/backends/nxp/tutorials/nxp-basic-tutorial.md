# Preparing a Model for NXP eIQ Neutron Backend

This guide demonstrating the use of ExecuTorch AoT flow to convert a PyTorch model to ExecuTorch
format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

## Step 1: Environment Setup

This tutorial is intended to be run from a Linux and uses Conda or Virtual Env for Python environment management. For full setup details and system requirements, see [Getting Started with ExecuTorch](/getting-started).

Create a Conda environment and install the ExecuTorch Python package.
```bash
conda create -y --name executorch python=3.12
conda activate executorch
conda install executorch
```

Run the setup.sh script to install the neutron-converter:
```commandline
$ ./examples/nxp/setup.sh
```

## Step 2: Model Preparation and Running the Model on Target

See the example `aot_neutron_compile.py` and its [README](https://github.com/pytorch/executorch/blob/release/1.0/examples/nxp/README.md) file. 
