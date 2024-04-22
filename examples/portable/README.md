# ExecuTorch in Portable Mode

This dir contains demos to illustrate an end-to-end workflow of using ExecuTorch in [portable mode](../../docs/source/concepts.md#portable-mode-lean-mode).


## Directory structure
```bash
examples/portable
├── scripts                           # Python scripts to illustrate export workflow
│   ├── export.py
│   └── export_and_delegate.py
├── custom_ops                        # Contains examples to register custom operators into PyTorch as well as register its kernels into ExecuTorch runtime
├── executor_runner                   # Contains an example C++ wrapper around the ExecuTorch runtime
└── README.md                         # This file
```

## Using portable mode

We will walk through an example model to generate a `.pte` file in [portable mode](../../docs/source/concepts.md#portable-mode-lean-mode) from a python `torch.nn.module`
from the [`models/`](../models) directory using scripts in the `portable/scripts` directory. Then we will run on the `.pte` model on the ExecuTorch runtime. For that we will use `executor_runner`.


1. Following the setup guide in [Setting up ExecuTorch](https://pytorch.org/executorch/0.2/getting-started-setup)
you should be able to get the basic development environment for ExecuTorch working.

2. Using the script `portable/scripts/export.py` generate a model binary file by selecting a
model name from the list of available models in the `models` dir.


```bash
cd executorch # To the top level dir

# To get a list of example models
python3 -m examples.portable.scripts.export -h

# To generate a specific pte model
python3 -m examples.portable.scripts.export --model_name="mv2" # for MobileNetv2

# This should generate ./mv2.pte file, if successful.
```

Use `-h` (or `--help`) to see all the supported models.

3. Once we have the model binary (`.pte`) file, then let's run it with ExecuTorch runtime using the `executor_runner`.

```bash
buck2 run examples/portable/executor_runner:executor_runner -- --model_path ./mv2.pte
```


## Custom Operator Registration

Explore the demos in the [`custom_ops/`](./custom_ops) directory to learn how to register custom operators into ExecuTorch as well as register its kernels into ExecuTorch runtime.
