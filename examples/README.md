# Examples

This dir contains scripts and other helper utilities to illustrate an end-to-end workflow to run a torch.nn.module on the Executorch runtime.
It also includes a list of modules, from a simple `Add` to a full model like `MobileNetv2` and `MobileNetv3`, with more to come.


## Directory structure
```bash
examples
├── executor_runner # This is an example C++ wrapper around the ET runtime
├── export          # Python helper scripts to illustrate export workflow
├── models          # Contains a set of simple to PyTorch models
└── README.md       # This file
```

## Using the examples

We will walk through an example model to generate a binary file from a python torch.nn.module
from the `models` dir using scripts from the `export` dir. Then we will run on these binary
model files on the Executorch (ET) runtime. For that we will use `executor_runner`. It is a simple
wrapper for the Executorch runtime to serve as an example. Although simple, it is capable of loading
and executing previously exported binary file(s).


1. Following the setup guide in [Setting up ExecuTorch from GitHub](/docs/website/docs/tutorials/00_setting_up_executorch.md)
you should be able to get the basic development environment for Executorch working.

2. Using the script `export/export_example.py` generate a model binary file by selecting a
model name from the list of available models in the `models` dir.


```bash
cd executorch # To the top level dir

bash examples/install_requirements.sh

# To get a list of example models
python3 -m examples.export.export_example -h

# To generate a specific pte model
python3 -m examples.export.export_example --model_name="mv2" # for MobileNetv2

# This should generate ./mv2.pte file, if successful.
```

Use `-h` (or `--help`) to see all the supported models.

3. Once we have the model binary (pte) file, then let's run it with Executorch runtime using the `executor_runner`.

```bash
buck2 run examples/executor_runner:executor_runner -- --model_path mv2.pte
```

## Dependencies

Various models listed in this directory have dependencies on some other packages, e.g. torchvision, torchaudio.
In order to make sure model's listed in examples are importable, e.g. via

```python
from executorch.examples.models.mobilenet_v3d import MV3Model
m = MV3Model.get_model()
```
we need to have appropriate packages installed. You should install these deps via install_requirements.sh.
