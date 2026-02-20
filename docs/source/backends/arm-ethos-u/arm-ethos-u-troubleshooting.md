# Arm Ethos-U Troubleshooting

This page describes common issues that you may encounter when using the Arm Ethos-U backend and how to debug and resolve them.

## Understanding memory footprint using the Ethos-U compiler

As part of the `to_edge_transform_and_lower` step, you will see a memory footprint information presented as:

```
Total SRAM used                               2467.27 KiB
Total Off-chip Flash used                       12.20 KiB
```

The `Total SRAM used` indicates the peak SRAM utilization needed by the NPU in order to perform an inference. In the snippet above, the Ethos-U compiler requires 2467.27 KiB of SRAM in order to schedule the inference.
Therefore, from an application standpoint, you need to ensure you have at least 2467.27 KiB of SRAM on the SoC to run this model. The Ethos-U compiler provides a scheduling algorithm allowing to
lower the peak SRAM usage within reasonable limits, you need to add the `--optimise Size` or `--arena-cache-size` CLI options for to the compile spec. You can read more about the options of the
Ethos-U compiler in the documentation [here](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/OPTIONS.md#optimise). If the peak SRAM usage remains too high in
Shared Sram memory mode, you would need to us the Dedicated Sram mode in order to store the Neural Network and the Ethos-U scratch buffer in the external memory.
The main advantage of the Dedicated_Sram memory mode is that you can run large models and still benefit from the low-latency/high-bandwidth of the SRAM, used as a cache.
It is important to highlight that when you specify a memory mode in the compile spec, in the runtime, the user is expected to place the scratch buffer and NN in the correct memory location.
In other words, when you specify for ex. Shared Sram memory mode, the runtime application logic should place the ethos-U scratch buffer in the on-chip memory and the NN in the external memory for optimal performance.
You can see how  this coupling between the memory mode and runtime application is done in the
[Ethos-U porting guide](https://github.com/pytorch/executorch/blob/main/examples/arm/ethos-u-porting-guide.md)

## Using Bundled.io and ETdump

The arm_executor_runner supports [bundled-io](https://docs.pytorch.org/executorch/0.4/bundled-io.html) and [ETdump](https://docs.pytorch.org/executorch/stable/etdump.html) debugging tools.

To enable bundled-io, set `EXECUTORCH_BUILD_DEVTOOLS` when building Executorch and `DET_BUNDLE_IO` when building the executor_runner. To enable ETdump, set `EXECUTORCH_BUILD_ARM_ETDUMP` when building Executorch and `DEXECUTORCH_ENABLE_EVENT_TRACER` when building the executor_runner.

## Issues with memory formats

Tensors of rank 4 and higher have two differing [memory format](https://pytorch.org/blog/tensor-memory-format-matters/) standards used.
PyTorch defaults to contiguous/ channels first/ NCHW memory formats, compared to TOSA which only supports channels last/NHWC memory format.
To support this, the backend inserts a transpose in the beginning if the incoming memory format is contiguous, and correspondingly a
transpose in the end if the outgoing memory format is contiguous. Note that this means that you may avoid transposing the data unneccessarily if the runtime integration and
full network is converted to use channels last. A word of caution must be given here however - changing memory format has been noted to have side effects such as
unsupported ops being inserted into the graph, and it is currently not widely tested, so the feature must so far be viewed as experimental.
