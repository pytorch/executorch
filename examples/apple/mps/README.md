This README gives some examples on backend-specific model workflow.

# MPS Backend

[MPS](https://developer.apple.com/documentation/metalperformanceshaders) is a framework of highly optimized compute and graphics shaders, specially tuned to take advantage of the unique hardware characteristics of each GPU family to ensure optimal performance.
**MPS** backend takes advantage of [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc) to build, compile, and execute customized multidimensional graphs from the edge dialect ops.

## Prerequisite

Please finish the following tutorials:
- [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup).
- [Setting up MPS backend](../../../backends/apple/mps/setup.md).

## Delegation to MPS backend

The following command will lower the EdgeIR to MPS delegate:

```bash
# For MobileNet V2
python3 -m examples.apple.mps.scripts.mps_example --model_name="mv2" --bundled
```
To see all the options when exporting a model to MPS delegate, use the following command:
```
python3 -m examples.apple.mps.scripts.mps_example --help
```

Once we have the model binary file, then let's run it with the ExecuTorch runtime using the `mps_executor_runner`.

```bash
# Build and install executorch
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_DEVTOOLS=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
          -DEXECUTORCH_BUILD_MPS=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
cmake --build cmake-out -j9 --target install --config Release
CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
# build mps_executor_runner
rm -rf cmake-out/examples/apple/mps
cmake \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -Bcmake-out/examples/apple/mps \
    examples/apple/mps

cmake --build cmake-out/examples/apple/mps -j9 --config Release

# Run the mv2 generated model using the mps_executor_runner
./cmake-out/examples/apple/mps/mps_executor_runner --model_path mv2_mps_bundled.pte --bundled_program
```

## Profiling

The following arguments can be used alongside the mps_executor_runner to benchmark a model:
- `--num-runs`: Number of total iterations.
- `--profile`: Show execution time for each iteration.
- `--bundled_program`: Load the inputs and outputs from the bundled program (note that during export, `--bundled` flag must be passed)

For example:
```bash
./cmake-out/examples/apple/mps/mps_executor_runner --model_path mv2_mps_bundled.pte --profile --num_runs 10
```

## Limitation

1. MPS backend is currently supported from iOS 17 and macOS Sonoma and newer.
