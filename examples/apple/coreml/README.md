# Examples

This directory contains scripts and other helper utilities to illustrate an end-to-end workflow to run a Core ML delegated `torch.nn.module` with the ExecuTorch runtime.


## Directory structure
```bash
coreml
├── scripts             # Scripts to build the runner.
├── executor_runner     # The runner implementation.
└── README.md           # This file.
```

## Using the examples

We will walk through an example model to generate a Core ML delegated binary file from a python `torch.nn.module` then we will use the `coreml_executor_runner` to run the exported binary file.

1. Following the setup guide in [Setting Up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup)
you should be able to get the basic development environment for ExecuTorch working.

2. Run `install_requirements.sh` to install dependencies required by the **Core ML** backend.

```bash
cd executorch

./backends/apple/coreml/scripts/install_requirements.sh

```

3. Run the export script to generate a Core ML delegated binary file.

```bash
cd executorch

# To get a list of example models
python3 -m examples.portable.scripts.export -h

# Generates add_coreml_all.pte file if successful.
python3 -m examples.apple.coreml.scripts.export --model_name add
```

4. Run the binary file using the `coreml_executor_runner`.

```bash
cd executorch

# Builds the Core ML executor runner. Generates ./coreml_executor_runner if successful.
./examples/apple/coreml/scripts/build_executor_runner.sh

# Run the delegated model.
./coreml_executor_runner --model_path add_coreml_all.pte
```

## Frequently encountered errors and resolution.
- The `examples.apple.coreml.scripts.export` could fail if the model is not supported by the Core ML backend. The following models from the examples models list (` python3 -m examples.portable.scripts.export -h`) are currently supported by the Core ML backend.

```text
add
add_mul
dl3
edsr
emformer_join
emformer_predict
emformer_transcribe
ic3
ic4
linear
llama2
llava_encoder
mobilebert
mul
mv2
mv2_untrained
mv3
resnet18
resnet50
softmax
vit
w2l
```

- If you encountered any bugs or issues following this tutorial please file a bug/issue [here](https://github.com/pytorch/executorch/issues) with tag #coreml.
