# Examples

This directory contains scripts and other helper utilities to illustrate an end-to-end workflow to run a **CoreML** delegated `torch.nn.module` with the **ExecuTorch** runtime.


## Directory structure
```bash
coreml
|── scripts             # Scripts to build the runner.
├── executor_runner     # The runner implementation.
└── README.md           # This file.
```

## Using the examples

We will walk through an example model to generate a **CoreML** delegated binary file from a python `torch.nn.module` then we will use the `coreml/executor_runner` to run the exported binary file.

1. Following the setup guide in [Setting Up ExecuTorch](/docs/source/getting-started-setup.md)
you should be able to get the basic development environment for ExecuTorch working.

2. Run `install_requirements.sh` to install dependencies required by the **CoreML** backend.

```bash
cd executorch

sh backends/apple/coreml/scripts/install_requirements.sh   

``` 

3. Run the export script to generate a **CoreML** delegated binary file. 

```
cd executorch

# To get a list of example models
python3 -m examples.export.export_example -h

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
