## ExecuTorch on ARM Cortex-M55 + Ethos-U55

This dir contains scripts to help you prepare setup needed to run a PyTorch
model on an ARM Corstone-300 platform via ExecuTorch. Corstone-300 platform
contains the Cortex-M55 CPU and Ethos-U55 NPU.

We will start from a PyTorch model in python, export it, convert it to a `.pte`
file - A binary format adopted by ExecuTorch. Then we will take the `.pte`
model file and embed that with a baremetal application executor_runner. We will
then take the executor_runner file, which contains not only the `.pte` binary but
also necessary software components to run standalone on a baremetal system.
Lastly, we will run the executor_runner binary on a Corstone-300 FVP Simulator
platform.

### Example workflow

There are two main scripts, setup.sh and run.sh. Each takes one optional,
positional argument. It is a path to a scratch dir to download and generate
build artifacts. If supplied, the same argument must be supplied to both the scripts.

To run these scripts. On a Linux system, in a terminal, with a working internet connection,
```
# Step [1] - setup necessary tools
$ cd <EXECUTORCH-ROOT-FOLDER>
$ executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula [optional-scratch-dir]

# Step [2] - Setup Patch to tools, The `setup.sh` script has generated a script that you need to source everytime you restart you shell. 
$ source  executorch/examples/arm/ethos-u-scratch/setup_path.sh

# Step [3] - build + run ExecuTorch and executor_runner baremetal application
# suited for Corstone FVP's to run a simple PyTorch model.
$ executorch/examples/arm/run.sh --model_name=mv2 --target=ethos-u85-128 [--scratch-dir=same-optional-scratch-dir-as-before]
```

### Ethos-U minimal example

See the jupyter notebook `ethos_u_minimal_example.ipynb` for an explained minimal example of the full flow for running a
PyTorch module on the EthosUDelegate. The notebook runs directly in some IDE:s s.a. VS Code, otherwise it can be run in
your browser using
```
pip install jupyter
jupyter notebook ethos_u_minimal_example.ipynb
```

### Online Tutorial

We also have a [tutorial](https://pytorch.org/executorch/main/backends-arm-ethos-u) explaining the steps performed in these
scripts, expected results, possible problems and more. It is a step-by-step guide
you can follow to better understand this delegate.
