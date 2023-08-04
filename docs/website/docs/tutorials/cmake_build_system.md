# Building with CMake

Although buck2 is the main build system for the ExecuTorch project, it's also
possible to build core pieces of the runtime using [CMake](https://cmake.org/)
for easier integration with other build systems. Even if you don't use CMake
directly, CMake can emit scripts for other format like Make or Ninja. (see
[cmake-generators(7)](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)).

## Targets built by the CMake build system

ExecuTorch's CMake build system doesn't cover everything that the buck2 build
system covers. It can only build pieces of the runtime that are likely to be
useful to embedded systems users.

- `libexecutorch.a`: The core of the ExecuTorch runtime. Does not contain any
  operator/kernel definitions or backend definitions.
- `libportable_kernels.a`: The implementations of ATen-compatible operators,
  following the signatures in `//kernels/portable/functions.yaml`.
- `libportable_kernels_bindings.a`: Generated code that registers the contents
  of `libportable_kernels.a` with the runtime.
  - NOTE: This must be linked into your application with a flag like
    `-Wl,-force_load` or `-Wl,--whole-archive`. It contains load-time functions
    that automatically register the kernels, but linkers will often prune those
    functions by default because there are no direct calls to them.
- `executor_runner`: An example tool that runs a `.pte` program file using all
  `1` values as inputs, and prints the outputs to stdout. It is linked with
  `libportable_kernels.a`, so the program may use any of the operators it
  implements.

## One-time setup

1. Clone the repo and install buck2 as described in the "Runtime Setup" section
   of [Setting up Executorch](00_setting_up_executorch.md#runtime-setup)
   - `buck2` is necessary because the CMake build system runs `buck2` commands
     to extract source lists from the primary build system. It will be possible
     to configure the CMake system to avoid calling `buck2`, though.
1. If your system's version of python3 is older than 3.11:
   - Run `pip install tomli`
   - This provides an import required by a script that the CMake build system
     calls to extract source lists from `buck2`. Consider doing this `pip
     install` inside your conda environment if you created one during AOT Setup
     (see [Setting up
     Executorch](00_setting_up_executorch.md#aot-setup-open-on-google-colab)).
1. Install CMake version 3.13 or later

## Configure the CMake build

Follow these steps after cloning or pulling the upstream repo, since the build
dependencies may have changed.

```bash
# cd to the root of the executorch repo
cd executorch

# Clean and configure the CMake build system. It's good practice to do this
# whenever cloning or pulling the upstream repo.
#
# NOTE: If your `buck2` binary is not on the PATH, you can change this line to
# say something like `-DBUCK2=/tmp/buck2` to point directly to the tool.
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DBUCK2=buck2 ..)
```

Once this is done, you don't need to do it again until you pull from the
upstream repo again.

## Build the runtime components

Build all targets with

```bash
# cd to the root of the executorch repo
cd executorch

# Build using the configuration that you previously generated under the
# `cmake-out` directory.
#
# NOTE: The `-j` argument specifies how many jobs/processes to use when
# building, and tends to speed up the build significantly. It's typical to use
# "core count + 1" as the `-j` value.
cmake --build cmake-out -j9
```

## Try using `executor_runner`

First, generate an `add.pte` or other ExecuTorch program file using the
instructions in the "AOT Setup" section of
[Setting up Executorch](00_setting_up_executorch.md#aot-setup-open-on-google-colab).

Then, pass it to the commandline tool:

```bash
./cmake-out/executor_runner --model_path path/to/add.pte
```

If it worked, you should see the message "Model executed successfully" followed
by the output values.

```
I 00:00:00.002052 executorch:executor_runner.cpp:75] Model file add.pte is loaded.
I 00:00:00.002086 executorch:executor_runner.cpp:85] Running method forward
I 00:00:00.002092 executorch:executor_runner.cpp:140] Setting up non-const buffer 1, size 48.
I 00:00:00.002149 executorch:executor_runner.cpp:181] Method loaded.
I 00:00:00.002154 executorch:util.h:105] input already initialized, refilling.
I 00:00:00.002157 executorch:util.h:105] input already initialized, refilling.
I 00:00:00.002159 executorch:executor_runner.cpp:186] Inputs prepared.
I 00:00:00.011684 executorch:executor_runner.cpp:195] Model executed successfully.
I 00:00:00.011709 executorch:executor_runner.cpp:210] 8.000000
```
