## ExecuTorch on ARM Cortex-M55 + Ethos-U55

This dir contains scripts to help you prepare setup needed to run a PyTorch
model on an ARM Corstone-300 platform via ExecuTorch. Corstone-300 platform
contains the Cortex-M55 CPU and Ethos-U55 NPU.

We will start from a PyTorch model in python, export it, convert it to a `.pte`
file - A binary format adopted by ExecuTorch. Then we will take the `.pte`
model file and embed that with a baremetal application executor_runner. We will
then take the executor_runner file, which contains not only the `.pte` file but
also necessary software component to run standalone on a baremetal system.
Lastly, we will run the executor_runner binary on a Corstone-300 FVP Simulator
platform.

### Example workflow

There are two main scripts, setup.sh and run.sh. Each takes one optional,
positional argument. It is a path to a scratch dir to download and generate
build artifacts. If supplied, the same argument must be supplied to both the scripts.

To run these scripts. On a Linux system, in a terminal, with a working internet connection,
```
# Step [1] - setup necessary tools
$ ./setup.sh --i-agree-to-the-contained-eula [optional-scratch-dir]

# Step [2] - build + run ExecuTorch and executor_runner baremetal application
# suited for Corstone300 to run a simple PyTorch model.
$ ./run.sh [same-optional-scratch-dir-as-before]
```
### Online Tutorial

We also have a [tutorial](https://pytorch.org/executorch/stable/executorch-arm-delegate-tutorial.html) explaining the steps performed in these
scripts, expected results, and more. It is a step-by-step guide
you can follow to better understand this delegate.
