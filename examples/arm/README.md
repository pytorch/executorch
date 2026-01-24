## ExecuTorch for Arm backends Ethos-U, VGF and Cortex-M

This project contains scripts to help you setup and run a PyTorch
model on a Arm backend via ExecuTorch. This backend supports Ethos-U and VGF as 
targets (using TOSA) but you can also use the Ethos-U example runner as an example
on Cortex-M if you do not delegate the model.

The main scripts are `setup.sh`, `run.sh` and `aot_arm_compiler.py`.

`setup.sh` will install the needed tools and with --root-dir <FOLDER> 
you can change the path to a scratch folder where it will download and generate build
artifacts. If supplied, you must also supply the same folder to run.sh with
--scratch-dir=<FOLDER> If not supplied both script will use examples/arm/arm-scratch

`run.sh` can be used to build, run and test a model in an easy way and it will call cmake for you
and in cases you want to run a simulator it will start it also. The script will call `aot_arm_compiler.py`
to convert a model and include it in the build/run.

Build and test artifacts are by default placed under the folder arm_test folder
this can be changed with --et_build_root=<FOLDER>

`aot_arm_compiler.py` is used to convert a Python model or a saved .pt model to a PTE file and is used by `run.sh`
and other test script but can also be used directly.

If you prefer to use the ExecuTorch API, there is also the `ethos_u_minimal_example.ipynb` notebook example.
This shows the workflow if you prefer to integrate a python torch.export and ExecuTorch flow directly into your
model codebase. This is particularly useful if you want to perform more complex training, such as quantization
aware training using the ArmQuantizer.

## Create a PTE file for Arm backends

There is an easy to use example flow to compile your PyTorch model to a PTE file for the Arm backend called `aot_arm_compiler.py`
that you can use to generate PTE files, it can generate PTE files for the supported targets `-t` or even non delegated (Cortex-M)
using different memory modes and can both use a python file as input or just use the models from examples/models with `--model_input`.
It also supports generating Devtools artifacts like BundleIO BPTE files, and ETRecords. Run it with `--help` to check its capabilities.

You point out the model to convert with `--model_name=<MODELNAME/FILE>` It supports running a model from examples/models or models
from a python file if you just specify `ModelUnderTest` and `ModelInput` in it.

```
$ python3 -m examples.arm.aot_arm_compiler --help
```

This is how you generate a BundleIO BPTE of a simple add example

```
$ python3 -m examples.arm.aot_arm_compiler --model_name=examples/arm/example_modules/add.py --target=ethos-u55-128 --bundleio
```

The example model used has added two extra variables that is picked up to make this work.

`ModelUnderTest` should be a `torch.nn.module` instance.

`ModelInputs` should be a tuple of inputs to the forward function.


You can also use the models from example/models directly by just using the short name e.g.

```
$ python3 -m examples.arm.aot_arm_compiler --model_name=mv2 --target=ethos-u55-64
```


The `aot_arm_compiler.py` is called from the scripts below so you don't need to, but it can be useful to do by hand in some cases.


## ExecuTorch on Arm Ethos-U55/U65 and U85

This example code will help you get going with the Corstone&trade;-300/320 platforms and
run on the FVP and can be used a a starting guide in your porting to your board/HW

We will start from a PyTorch model in python, export it, convert it to a `.pte`
file - A binary format adopted by ExecuTorch. Then we will take the `.pte`
model file and embed that with a baremetal application executor_runner. We will
then take the executor_runner file, which contains not only the `.pte` binary but
also necessary software components to run standalone on a baremetal system.
The build flow will pick up the non delegated ops from the generated PTE file and 
add CPU implementation of them. 
Lastly, we will run the executor_runner binary on a Corstone&trade;-300/320 FVP Simulator platform.


### Example workflow

Below is example workflow to build an application for Ethos-U55/85. The script below requires an internet connection:

```
# Step [1] - setup necessary tools
$ cd <EXECUTORCH-ROOT-FOLDER>
$ ./examples/arm/setup.sh --i-agree-to-the-contained-eula

# Step [2] - Setup path to tools, The `setup.sh` script has generated a script that you need to source every time you restart you shell.
$ source  examples/arm/arm-scratch/setup_path.sh

# Step [3] - build and run ExecuTorch and executor_runner baremetal example application
# on a Corstone(TM)-320 FVP to run a simple PyTorch model from a file.
$ ./examples/arm/run.sh --model_name=examples/arm/example_modules/add.py --target=ethos-u85-128
```

The argument `--model_name=<MODEL>` is passed to `aot_arm_compiler.py` so you can use it in the same way
e.g. you can also use the models from example/models directly in the same way as above.

```
$ ./examples/arm/run.sh --model_name=mv2 --target=ethos-u55-64
```

The runner will by default set all inputs to "1" and you are supposed to add/change the code
handling the input for your hardware target to give the model proper input, maybe from your camera
or mic hardware.

While testing you can use the --bundleio flag to use the input from the python model file and
generate a .bpte instead of a .pte file. This will embed the input example data and reference output
in the bpte file/data, which is used to verify the model's output. You can also use --etdump to generate
an ETRecord and a ETDump trace files from your target (they are printed as base64 strings in the serial log).

Just keep in mind that CPU cycles are NOT accurate on the FVP simulator and it can not be used for
performance measurements, so you need to run on FPGA or actual ASIC to get good results from --etdump.
As a note the printed NPU cycle numbers are still usable and closer to real values if the timing
adaptor is setup correctly.

```
# Build + run with BundleIO and ETDump
$ ./examples/arm/run.sh --model_name=lstm --target=ethos-u85-128 --bundleio --etdump
```


### Ethos-U minimal example

See the jupyter notebook `ethos_u_minimal_example.ipynb` for an explained minimal example of the full flow for running a
PyTorch module on the EthosUDelegate. The notebook runs directly in some IDE:s s.a. VS Code, otherwise it can be run in
your browser using
```
pip install jupyter
jupyter notebook ethos_u_minimal_example.ipynb
```

## ExecuTorch on ARM Cortex-M

For Cortex-M you run the script without delegating e.g `--no_delegate` as the build flow already supports picking up
the non delegated ops from the generated PTE file and add CPU implementation of them this will work out of the box in
most cases.

To run mobilenet_v2 on the Cortex-M55 only, without using the Ethos-U try this:

```
$ ./examples/arm/run.sh --model_name=mv2 --target=ethos-u55-128 --no_delegate
```


### Online Tutorial

We also have a [tutorial](https://pytorch.org/executorch/stable/backends/arm-ethos-u/arm-ethos-u-overview.html) explaining the steps performed in these
scripts, expected results, possible problems and more. It is a step-by-step guide
you can follow to better understand this delegate.
