# Direct Mode

## Introduction
This tutorial will cover **Direct Mode**, also known as the **Native DSP Backend** in the QNN SDK. The QNN SDK provides predefined protocols for general use cases. However, there may be situations where users want to go further and define their own RPC calls for customized workflows. For example, a user might want to perform an RPC call that handles model loading, input loading and setting, execution, and output saving in a single call. This is something not possible with QNN's predefined RPC protocol. This approach can improve performance by giving users control over where resources are loaded and by reducing the number of RPC calls. To address this need, **Direct Mode** was introduced, providing flexibility for users to define their own FastRPC protocol. For more information about FastRPC, please refer to the [Hexagon SDK](https://www.qualcomm.com/developer/software/hexagon-npu-sdk) for details on setup, building, and defining custom protocols.

## Requirments
Below are the required files to enable **Direct Mode**. Example files are also provided for reference.
1. A **.idl** file that defines the interface. A sample self-defined protocol can be found under [qnn_executorch.idl](qnn_executorch.idl).
This file specifies how the AP and DSP communicate. It can be compiled into header, stub, and skel files using the Hexagon SDK’s **qaic** compiler. **qaic** compiler and more information about **qaic** can be found under `$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu`

2. Implementation for the skel. An example for skel implementation for [qnn_executorch.idl](qnn_executorch.idl) can be found in [qnn_executorch_imp.cpp](qnn_executorch_imp.cpp).

3. Implementation to control session and perform RPC calls. An example runner can be found in [qnn_executor_direct_runner.cpp](../../../../../examples/qualcomm/direct_executor_runner/qnn_executor_direct_runner.cpp)

## Instructions
Below are the steps to build **Direct Mode** artifacts and execute with **Direct Mode**.

1. Export required environment variables. Please export the following 3 variables:
 - `HEXAGON_SDK_ROOT`: Path to Hexagon SDK root directory.
 - `HEXAGON_TOOLS_ROOT`: Hexagon SDK includes 1 or more toolchains. If you are unsure which toolchain to use, you can check `$QNN_SDK_ROOT/share/QNN/OpPackageGenerator/makefiles/HTP/Makefile`. Inside, you will find a mapping between devices and toolchains. The path to `HEXAGON_TOOLS_ROOT` should look similar to `$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04`
 - `DSP_VERSION`: The target DSP architecture (e.g., `v79`).

2. Build necessary artifacts
```bash
backends/qualcomm/scripts/build.sh --enable_hexagon
```
3. Execution
Below is an example to execute a unit test with direct mode using qnn_executor_direct_runner.
```
python backends/qualcomm/tests/test_qnn_delegate.py -k TestQNNQuantizedOperator.test_qnn_backend_adaptive_avg_pool2d --model SM8750  --device $DEVICE_ID --build_folder build-android --direct_build_folder build-hexagon/
```

### Note
The model execution time for `qnn_executor_direct_runner` is expected to be faster than `qnn_executor_runner` because it reduces DMA usage and minimizes the number of RPC calls.
However, you may observe that the total completion time for `qnn_executor_direct_runner` appears longer. This is expected in the demo script, since the runner performs file loading and saving on the DSP side. These operations can be slightly slower compared to when the AP handles them.
In production scenarios, this difference should not be a concern. Typically, inputs will be accessed directly from memory and outputs will be handled in more optimized ways. The file I/O in the demo is included only to align the behavior of `qnn_executor_direct_runner` with `qnn_executor_runner`, and to simplify testing. It is not representative of the intended performance characteristics in real-world usage.