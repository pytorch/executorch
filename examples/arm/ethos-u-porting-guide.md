# Introduction

As you have seen in in the [Arm&reg; Ethos&trade;-U NPU backend tutorial](../../docs/source/backends/arm-ethos-u/tutorials/ethos-u-getting-started.md), ExecuTorch has two distinct parts:
- Ahead-of-time(AoT) compile flow
- Ethos-U on-device runtime

In the porting guide, we guide you through the main steps to port your SoC with an Ethos-U to the ExecuTorch Ethos-U backend in order to leverage the ExecuTorch Ethos-U enablement. We assume
you are familiar with the concepts introduced in `backends-arm-ethos-u.md`, you have already generated a pte in the AoT flow and want to deploy the ML model on device.
Fundamentally, there are two big approaches you can take in porting a SoC with an Ethos-U NPU towards the ExecuTorch runtime.
- You can use the enablement we have done in ExecuTorch for the Arm&reg; Corstone&trade;-300(Arm&reg; Cortex&reg;-M55 and Arm&reg; Ethos&trade;-U55 reference design) and
Arm&reg; Corstone&trade;-320(Arm&reg; Cortex&reg;-M85 and Arm&reg; Ethos&trade;-U85 reference design) and migrate from the Corstone platform towards a new platform.
- If the SoC comes with an SDK that is not based on ExecuTorch, you can replace the runtime SDKs with the corresponding APIs from ExecuTorch runtime.

It is important to understand that irrespective of whether the SoC comes with or without SDK, there is
[a single Ethos-U driver](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver) that any SoC with Ethos-U relies on. For that reason, there will be
overlap between the two approaches when it comes to the enablement of the Ethos-U NPU.

## Functioning of a system with an Ethos-U
A system with an Ethos-U and Cortex-M or Arm&reg; Cortex&reg;-A functions in the following way:
- The CPU(Cortex-M or Cortex-A) dispatches an inference job to the Ethos-U NPU. The inference job is in the form of command stream - a sequence of instructions that the NPU
executes. The command stream is generated as part of the `to_edge_transform_and_lower` AoT compile stage. The command stream is embedded within the pte file and at runtime,
the pte file is stored in the memory of the SoC.
- The Ethos-U NPU autonomously reads the command stream from memory. When the NPU finishes processing the command stream, it raises an interrupt to the CPU to signal
that the inference job is complete. The CPU executes the interrupt handler and resumes its normal execution.

### Ethos-U memory regions
In order to allow this functioning, the Ethos-U driver defines the following regions that the NPU hardware will access:
- Ethos-U scratch buffer - a contiguous block of memory used by the NPU to store the intermediate tensors produced and consumed during inference. Applicable for any Ethos-U NPU.
- Neural Network - a contiguous block of memory holding constant data such as weights, biases, quantization parameters required to run an inference. Applicable for
any Ethos-U NPU.
- Ethos-U fast scratch buffer -  a contiguous block of memory for the case when the Ethos-U scratch buffer and Neural Network are both in the external memory.
Applicable only for Ethos-U65 and Ethos-U85 in Dedicated_Sram memory mode.

### Ethos-U driver
The key function of the Ethos-U driver enabling the interaction with the NPU is
(ethosu_invoke_v3)[https://github.com/pytorch/executorch/blob/main/backends/arm/runtime/EthosUBackend.cpp#L324]. The `ethosu_invoke_v3` function takes as input a driver handle,
a pointer towards  the command stream and the size of the command stream, base address as well as the size of the base addresses. For a system with Cortex-M, there is a 1:1
mapping between base pointer and region, so we will pass three base pointers and each base pointer will correspond to one region. Then, as part of the compilation stage
in `to_edge_transform_and_lower`, the Ethos-U compiler will generate command stream taking into account the three regions. In other words, at runtime, the ethos-U driver
knows the address of the command stream, its size, as well as the address for the locations in memory needed to store the intermediate tensors.
The Ethos-U driver will pass these address to the NPU and the NPU will issue memory requests to the on-chip or external memories in order to access the necessary data
(e.g. read weights, store an intermediate result into the scratch buffer, etc). The `backends/arm/runtime/EthosUBackend.cpp` already integrates the Ethos-U driver and it
already supports the three memory modes of the Ethos-U. Therefore, you should reuse `backends/arm/runtime/EthosUBackend.cpp` as is, without modifications.
The key question for any porting is how to initialize the NPU and
make sure it works. Let's analyse these questions in the following section.

**Note:** Interrupts work differently between Cortex-M and Cortex-A and a system with Cortex-A will use more base pointers and won't have a 1:1 mapping between Ethos-U
driver base pointer and Ethos-U region. The `backends/arm/runtime/EthosUBackend.cpp` is for a system with Cortex-M. Going forward, we assume we have a system with Cortex-M, similar to the Corstone platforms.

## NPU initialization
In order to initialize the NPU hardware, the software needs to provide correct information about:
- The base address of the Ethos-U NPU on the memory map of the SoC.
- The interrupt assignment for the Ethos-U. You also need to provide interrupt priority.

In the `executorch/examples/arm/executor_runner/arm_executor_runner.cpp` sample application, we inherit the Corstone-300/Corstone-320 NPU initialization done in the [core-platform project](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform).
We include the core-platform project as a dependency in the `backends/arm/scripts/corstone_utils.cmake` script. Note that in [corstone_utils.sh](https://github.com/pytorch/executorch/blob/main/backends/arm/scripts/corstone_utils.cmake#L69)
depending on whether we target Ethos-U55 or Ethos-U85, we include the  corresponding target from core-platform. Then, inside core-platform, the NPU base address and interrupt assignment are defined in the
[target.cpp](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform/-/blob/main/targets/corstone-320/target.cpp?ref_type=heads#L44) as per the memory map of the Corstone-300/Corstone-320.
It's worth mentioning that the code in core-platform(code we reuse in the `examples/arm/executor_runner/arm_executor_runner.cpp`) also calls the `ethosu_init`
function to initialize the NPU. The `ethosu_init` function is [defined in the Ethos-U driver](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver/-/blob/main/src/ethosu_driver.c?ref_type=heads#L409). The ethos-u driver itself is
included within the core-platform CMake. In other words, to initialize the NPU in the ExecuTorch executor runner application, we reuse the Ethos-U initialization that has been done in the core-platform project.
Core-platform includes a [tutorial](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform/-/blob/main/PORTING.md?ref_type=heads) on porting a new target. If you port your target to
core-platform, you can then easily reuse it in the ExecuTorch runtime.

Also, as explained in the comments in `backends/arm/scripts/corstone_utils.cmake`, note that REGIONCFG register of the Ethos-U controls the memory(on-chip or external memory) used by the NPU to
access the the Ethos-U scratch buffer, ML model and Ethos-U fast scratch buffer. The REGIONCFG is defined in the Ethos-U driver and you need to configure it differently depending on the memory mode.
You can see in `backends/arm/scripts/corstone_utils.cmake` how we overwrite the register to the correct value depending on the memory mode.

For the Corstone platforms, we make use of the Timing Adapters to model different memory latencies. The Timing Adapter is only applicable to the Corstone targets and should not be included for another SoC.

## Corstone linker scripts

The linker scripts point to the linker where to place various objects in memory when the application is loaded onto the target.
In the `arm_executor_runner.cpp` application, we reuse the linker scripts from the core-platform project. Note that the Global Offset Table(.got symbols) needs to be 16-byte aligned. The linker scripts are highly specific to the memory map of system.
For example on the Corstone-300,  in order to allow us to build a lot of portable kernels, we relocate the portable kernels from the .text section, living in the ITCM, to a bigger memory.  Also on the Corstone-300. the linker script defines two
load regions- rom_exec and rom_dram corresponding to loading the application in the ITCM and in the DDR. When you deploy the application, the boot loader copies the two binaries from the rom_exec/rom_dram regions to their physical address in memory -
a process known as scatter loader. Upon powering on the device, the very first instruction that the Cortex-M executes is the ResetHandler function. The ResetHandler is the first entry in the Vector Interrupt Table, and the location of the Vector Interrupt 
Table is specified in the linker script(the `KEEP(*(.vectors))` symbol). The assembly boot code powering on the Cortex-M is itself defined in `core_software/cmsis_6/CMSIS/CoreValidation/Layer/Target/CM55S/RTE/Device/ARMCM55/startup_ARMCM55.c`.
The CMSIS start-up code for Cortex-M is added as part of the build system of the core-platform applications.

## Coupling between the AoT compile specification memory mode, linker script and the application logic
It is important to note that when you specify a memory mode in the Python script to generate the pte file, in the runtime, the user is
expected to place the scratch buffer and NN in the correct memory location.

For example, if you generate a pte file with compile specification for Shared Sram, the scratch buffer should be placed in the SRAM and the NN in the external memory in the runtime application code. You can see we are following
this approach in the `examples/arm/executor_runner/arm_executor_runner.cpp` example application. In the linker scripts for the application(`examples/arm/executor_runner/Corstone-320.ld` and
`examples/arm/executor_runner/Corstone-300.ld`) we check the value of `ETHOSU_ARENA` to determine whether the ethos-u scratch buffer is placed in the on-chip memory or in the external memory. In this
way, depending on the `ETHOSU_ARENA` parameter, the linker knows whether the symbol is to be placed in the .ddr or .sram.bss sections. The `ETHOSU_ARENA` parameter is set in the `backends/arm/scripts/corstone_utils.cmake` and
its value is derived based on the memory mode parameter that is passed to the `examples/arm/run.sh` shell script. Then, at link time, the .ddr section is always placed in the external memory and the .sram.bss is always placed in the SRAM.
Finally, note that in the `examples/arm/executor_runner/arm_executor_runner.cpp` application code, we place the buffers for the Ethos-u scratch and the neural network in the correct symbol from the linker script. For instance,
the Ethos-u scratch buffer corresponds to the the `.bss.tensor_arena` section in the linker script, In the application code, when we allocate memory for the Ethos-u scratch buffer, we place this array in the .bss.tensor_arena section in the memory map.

```
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];
```
and the `.bss.tensor_arena` section is placed in the correct location in the memory map thanks to
the `ETHOSU_ARENA` parameter.

There is a tight coupling between the memory mode for the Ethos-U and the placement of the ethos-u scratch buffer,
ethos-u-fast scratch buffer (only applicable for Dedicated_Sram) and the neural network in the memory map of the
SoC. The `arm_executor_runner.cpp` application built with the `examples/arm/run.sh` shell script and corresponding linker scripts are aimed to serve as
example implementation for the correct placement of the various objects in memory.

It's also worth mentioning that in the AoT Python flow, by default the input to the pte file is in FP32. Therefore, the pte file contains a Quantize node, an Ethos-U custom delegate and a Dequantize node.
Sometimes, you may want to feed quantized input to the Ethos-U custom delegate straight away, for example if you have a camera input outputting RGB data in (u)int8. You can apply the `QuantizeInputs` and
`QuantizeOutputs` passes in the AoT flow for that purpose. Here is a snippet showing how to achieve it:
```
edge_program_manager = to_edge_transform_and_lower(...)
from executorch.exir.passes.quantize_io_pass import QuantizeInputs
from executorch.exir.passes.quantize_io_pass import QuantizeOutputs
# Apply the QuantizeInputs & QuantizeOutputs passes to input & output tensor 0
edge_program_manager.transform(passes=[QuantizeInputs(edge_program_manager, [0]),
                      QuantizeOutputs(edge_program_manager, [0])])
# Convert edge program to executorch
executorch_program_manager = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
```
If you apply  the `QuantizeInputs` pass in the AoT flow, when you populate the input tensor in the runtime application logic, you need to use int8 numbers and not FP32. In the `examples/arm/executor_runner/arm_executor_runner.cpp` application, you can see
how we populate the input tensor depending on its data type.

## Conclusion
The ExecuTorch project already provides an Arm Ethos-U backend in `executorch/backends/arm/runtime/` that you can reuse as is. The key steps to bring up a new platform is to reuse the Ethos-U driver
and ensure that the NPU base address and interrupt assignment matches your SoC.  The `examples/arm/executor_runner/arm_executor_runner.cpp` is an example application running on the Corstone platform.
For the `arm_executor_runner.cpp` application, we are relying on the NPU initialization done in the core-platform project and we integrate core-platform in the
`backends/arm/scripts/corstone_utils.cmake` script. Then, we inherit the core-platform integration of the Ethos-U driver and the CMSIS boot code for the Cortex-M core.