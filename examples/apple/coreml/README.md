# Examples

This directory contains scripts and other helper utilities to illustrate an end-to-end workflow to run a **CoreML** delegated `torch.nn.module` with the **ExecuTorch** runtime.


## Directory structure
```bash
coreml
├── scripts             # Scripts to build the runner.
├── executor_runner     # The runner implementation.
└── README.md           # This file.
```

## Using the examples

We will walk through an example model to generate a **CoreML** delegated binary file from a python `torch.nn.module` then we will use the `coreml/executor_runner` to run the exported binary file.

1. Following the setup guide in [Setting Up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup)
you should be able to get the basic development environment for ExecuTorch working.

2. Run `install_requirements.sh` to install dependencies required by the **CoreML** backend.

```bash
cd executorch

sh backends/apple/coreml/scripts/install_requirements.sh

```

3. Run the export script to generate a **CoreML** delegated binary file.

```bash
cd executorch

# To get a list of example models
python3 -m examples.portable.scripts.export -h

# Generates ./add_coreml_all.pte file if successful.
python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name add
```

4. Once we have the **CoreML** delegated model binary (pte) file, then let's run it with the **ExecuTorch** runtime using the `coreml_executor_runner`.

```bash
cd executorch

# Builds the CoreML executor runner. Generates ./coreml_executor_runner if successful.
sh examples/apple/coreml/scripts/build_executor_runner.sh

# Run the CoreML delegate model.
./coreml_executor_runner --model_path add_coreml_all.pte
```

## Frequently encountered errors and resolution.
- The `examples.apple.coreml.scripts.export_and_delegate` could fail if the model is not supported by the **CoreML** backend. The following models from the examples models list (` python3 -m examples.portable.scripts.export -h`)are currently supported by the **CoreML** backend.

```
add
add_mul
ic4
linear
mul
mv2
mv3
resnet18
resnet50
softmax
vit
w2l
```

- If you encountered any bugs or issues following this tutorial please file a bug/issue [here](https://github.com/pytorch/executorch/issues) with tag #coreml.
