# Building with CMake

Although buck2 is the main build system for the ExecuTorch project, it's also
possible to build core pieces of the runtime using [CMake](https://cmake.org/)
for easier integration with other build systems. Even if you don't use CMake
directly, CMake can emit scripts for other format like Make or Ninja. For information, see
[cmake-generators(7)](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).

## Targets Built by the CMake Build System

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

## One-time setup to prepare for CMake Build

Follow the steps below to have the tools ready before using CMake to build on your machine.

1. Clone the repo and install buck2 as described in the "Runtime Setup" section
   of [Setting Up ExecuTorch](getting-started-setup.md#building-a-runtime)
   - `buck2` is necessary because the CMake build system runs `buck2` commands
     to extract source lists from the primary build system. It will be possible
     to configure the CMake system to avoid calling `buck2`, though.
2. If your system's version of python3 is older than 3.11:
   - Run `pip install tomli`
   - This provides an import required by a script that the CMake build system
     calls to extract source lists from `buck2`. Consider doing this `pip
     install` inside your conda environment if you created one during ExecuTorch setup. see [Setting up
     ExecuTorch](getting-started-setup.md).
3. Install CMake version 3.19 or later
   - conda install cmake


## Configure the CMake Build

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

Once this is done, you don't need to do it again until you pull from the upstream repo again, or if you modify any CMake-related files.

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

## Use an example app `executor_runner` to execute a .pte file

First, generate an `add.pte` or other ExecuTorch program file using the
instructions as described in
[Setting up ExecuTorch](getting-started-setup.md#building-a-runtime).

Then, pass it to the command line tool:

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


## Cross compilation

Follwing are instruction on how to perform cross compilation for Android and iOS.

### Android
- Prerequisite: [Android NDK](https://developer.android.com/ndk), choose one of the following:
  - Option 1: Download Android Studio by following the instructions to [install ndk](https://developer.android.com/studio/projects/install-ndk).
  - Option 2: Download Android NDK directly from [here](https://developer.android.com/ndk/downloads).

Assuming Android NDK is available, run:
```bash
# Run the following lines from the `executorch/` folder
rm -rf cmake-android-out && mkdir cmake-android-out && cd cmake-android-out

# point -DCMAKE_TOOLCHAIN_FILE to the location where ndk is installed
# Run `which buck2`, if it returns empty (meaning the system doesn't know where buck2 is installed), pass in pass in this flag `-DBUCK2=/path/to/buck2` pointing to buck2
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/{user_name}/Library/Android/sdk/ndk/25.2.9519653/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a ..

cd  ..
cmake --build  cmake-android-out  -j9

# push the binary to an Android device
adb push  cmake-android-out/executor_runner  /data/local/tmp/executorch
# push the model file
adb push  add.pte  /data/local/tmp/executorch

adb shell  "/data/local/tmp/executorch/executor_runner --model_path /data/local/tmp/executorch/add.pte"
```

### iOS
```{note}
 While we're working on making it a smoother experience, here is an early workflow to try out cross compilation for iOS.
```
Only supported in macOS:

Prerequisites:
-   [XCode](https://developer.apple.com/xcode/)

After XCode is installed,

1. Get the iOS cmake toolchain by using one of the following options:
  - Option 1 [recommended] : use the `ios.toolchain.cmake` from the following github repo:

    ```bash
    git clone https://github.com/leetal/ios-cmake.git
    ```

  - Option2 [wip], use the `iOS.cmake` from PyTorch, the
    toolchain is located in `executorch/third-party/pytorch/pytorch/blob/main/cmake/iOS.cmake`


2.  Use the tool chain provided in the repro to build the ExecuTorch library.
    ```bash
    rm -rf cmake-ios-out && mkdir cmake-ios-out && cd cmake-ios-out

    # change the platform accordingly, please refer to the table listed in Readme
    cmake . -G Xcode -DCMAKE_TOOLCHAIN_FILE=~/ios-cmake/ios.toolchain.cmake -DPLATFORM=SIMULATOR

    # Create an include folder in cmake-ios-out to include all header files
    mkdir include
    cp -r ../runtime include
    cp -r ../extension include
    cp -r ../utils include
    ```


3. XCode setup

If using the iOS cmake tool chain from `https://github.com/leetal/ios-cmake.git`, after build is complete, perform the following steps:

1. Open the project in XCode, drag the `executorch.xcodeproj` generated from Step 2 to `Frameworks`.
1. Go to project Target’s  `Build Phases`  -  `Link Binaries With Libraries`, click the **+** sign and add all the library files located in  `cmake-ios-out/build`.
1. Navigate to the project  `Build Settings`:
  - Set the value  **Header Search Paths**  to  `cmake-ios-out/include`.
  - Set **Library Search Paths**  to  `cmake-ios-out/build`.
  - In **other linker flags**, add a custom linker flag `-all_load.`


## Next steps

You have successfully cross-compiled `executor_runner` binary to iOS and Android platforms. You can start exploring advanced features and capabilities. Here is a list of sections you might want to read next:

* [Selective build](./kernel-library-selective_build) to build the runtime that links to only kernels used by the program, which can provide significant binary size savings.
* Tutorials on building [Android](./demo-apps-android.md) and [iOS](./demo-apps-ios.md) demo apps.
* Tutorials on deploying applications to embedded devices such as [ARM Cortex-M/Ethos-U](./executorch-arm-delegate-tutorial.md) and [XTensa HiFi DSP](./build-run-xtensa.md).
