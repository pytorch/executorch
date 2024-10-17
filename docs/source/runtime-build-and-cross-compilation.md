# Building with CMake

ExecuTorch uses [CMake](https://cmake.org/)  as its primary build system.
Even if you don't use CMake directly, CMake can emit scripts for other format
like Make, Ninja or Xcode. For information, see [cmake-generators(7)](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).

## Targets Built by the CMake Build System

ExecuTorch's CMake build system covers the pieces of the runtime that are
likely to be useful to embedded systems users.

- `libexecutorch_core.a`: The core of the ExecuTorch runtime. Does not contain any
  operator/kernel definitions or backend definitions.
- `libportable_kernels.a`: The implementations of ATen-compatible operators,
  following the signatures in `[functions.yaml](https://github.com/pytorch/executorch/blob/release/0.4/kernels/portable/functions.yaml)`.
- `libportable_ops_lib.a`: Generated code that registers the contents
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

1. If your system's version of python3 is older than 3.11:
   - Run `pip install tomli`
3. Install CMake version 3.19 or later:
   - Run `conda install cmake` or `pip install cmake`.


## Configure the CMake Build

Follow these steps after cloning or pulling the upstream repo, since the build
dependencies may have changed.

```bash
# cd to the root of the executorch repo
cd executorch

# Clean cmake cache directory (cmake-out). It's a good practice to do this
# whenever cloning or pulling the upstream repo.
bash install_requirements.sh --clean
```

Once this is done, you don't need to do it again until you pull from the upstream repo again, or if you modify any CMake-related files.

### CMake Build Options

The release build offers optimizations intended to improve performance and reduce binary size. It disables program verification and executorch logging, and adds optimizations flags.
```bash
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
```

To further optimize the release build for size, add:
```bash
CMAKE_FLAGS="$CMAKE_FLAGS -DOPTIMIZE_SIZE=ON"
```

See [CMakeLists.txt](https://github.com/pytorch/executorch/blob/release/0.4/CMakeLists.txt)

## Build the runtime components

Build all targets with

```bash
# Build using the configuration that you previously generated under the
# `cmake-out` directory.
cmake "$CMAKE_FLAGS" -Bcmake-out .

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
I 00:00:00.000526 executorch:executor_runner.cpp:82] Model file add.pte is loaded.
I 00:00:00.000595 executorch:executor_runner.cpp:91] Using method forward
I 00:00:00.000612 executorch:executor_runner.cpp:138] Setting up planned buffer 0, size 48.
I 00:00:00.000669 executorch:executor_runner.cpp:161] Method loaded.
I 00:00:00.000685 executorch:executor_runner.cpp:171] Inputs prepared.
I 00:00:00.000764 executorch:executor_runner.cpp:180] Model executed successfully.
I 00:00:00.000770 executorch:executor_runner.cpp:184] 1 outputs: 
Output 0: tensor(sizes=[1], [2.])
```


## Cross compilation

Following are instruction on how to perform cross compilation for Android and iOS.

### Android
- Prerequisite: [Android NDK](https://developer.android.com/ndk), choose one of the following:
  - Option 1: Download Android Studio by following the instructions to [install ndk](https://developer.android.com/studio/projects/install-ndk).
  - Option 2: Download Android NDK directly from [here](https://developer.android.com/ndk/downloads).

Assuming Android NDK is available, run:
```bash
# Run the following lines from the `executorch/` folder
rm -rf cmake-android-out && mkdir cmake-android-out && cd cmake-android-out

# point -DCMAKE_TOOLCHAIN_FILE to the location where ndk is installed
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/{user_name}/Library/Android/sdk/ndk/25.2.9519653/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a ..

cd  ..
cmake --build  cmake-android-out  -j9

adb shell mkdir -p /data/local/tmp/executorch
# push the binary to an Android device
adb push  cmake-android-out/executor_runner  /data/local/tmp/executorch
# push the model file
adb push  add.pte  /data/local/tmp/executorch

adb shell  "/data/local/tmp/executorch/executor_runner --model_path /data/local/tmp/executorch/add.pte"
```

### iOS

For iOS we'll build [frameworks](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) instead of static libraries, that will also contain the public headers inside.

1. Install Xcode from the
[Mac App Store](https://apps.apple.com/app/xcode/id497799835) and then install
the Command Line Tools using the terminal:

```bash
xcode-select --install
```

2. Build the frameworks:

```bash
./build/build_apple_frameworks.sh
```

Run the above command with `--help` flag to learn more on how to build additional backends
(like [Core ML](build-run-coreml.md), [MPS](build-run-mps.md) or XNNPACK), etc.
Note, some backends may require additional dependencies and certain versions of Xcode and iOS.

3. Copy over the generated `.xcframework` bundles to your Xcode project, link them against
your targets and don't forget to add an extra linker flag `-all_load`.

Check out the [iOS Demo App](demo-apps-ios.md) tutorial for more info.


## Next steps

You have successfully cross-compiled `executor_runner` binary to iOS and Android platforms. You can start exploring advanced features and capabilities. Here is a list of sections you might want to read next:

* [Selective build](./kernel-library-selective_build) to build the runtime that links to only kernels used by the program, which can provide significant binary size savings.
* Tutorials on building [Android](./demo-apps-android.md) and [iOS](./demo-apps-ios.md) demo apps.
* Tutorials on deploying applications to embedded devices such as [ARM Cortex-M/Ethos-U](./executorch-arm-delegate-tutorial.md) and [XTensa HiFi DSP](./build-run-xtensa.md).
