# Arm(R) Ethos(TM)-U NPU Backend

The Arm Ethos-U backend is the ExecuTorch solution for executing quantized models on [Ethos-U55](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55), [Ethos-U65](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u65), and [Ethos-U85](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85) NPUs. It leverages the TOSA operator set which can be compiled by the ethos-u-vela graph compiler.

## Features

- Wide operator support for delegating large parts of models to highly optimized and low power Ethos-U NPUs.
- A quantizer that optimizes quantization for the NPU target.

## Target Requirements

The target system must include an Ethos-U NPU.

## Development Requirements

To compile for the NPUs, the Ethos-U Vela compiler is needed. A target-specific toolchain is also needed for building the runtime. Finally, to test models, Arm provides freely available Fixed Virtual Platforms (FVP), allowing running code on the Ethos-U without a a physical development board by emulating reference designs. For Ethos-U55, there is [Corstone-300](https://developer.arm.com/Processors/Corstone-300), and for Ethos-U85, there is [Corstone-320](https://developer.arm.com/Processors/Corstone-320).

These dependencies can easily be downloaded using the script `examples/arm/setup.sh`.

## Using the Arm Ethos-U backend
The example below demonstrates the lowering processs of a MobileNet V2 model from torchvision for a Ethos-U55 target. Since the model is a floating point model, first quantize it using the `EthosUQuantizer`. Then, pass an instance of the `EthosUPartitioner` to `to_edge_transform_and_lower`. Both the quantizer and the partitioner need a compilation specification created using `ArmCompileSpecBuilder`.

```python
import torch
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision.models import mobilenetv2
import executorch.kernels.quantized

mobilenet_v2 = mobilenetv2.mobilenet_v2(
    weights=mobilenetv2.MobileNet_V2_Weights.DEFAULT
).eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

compile_spec = ArmCompileSpecBuilder().ethosu_compile_spec(
        "ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags="--output-format=raw --debug-force-regor",
    ).build()

# Post training quantization
graph_module = torch.export.export(mobilenet_v2, example_inputs).module()
quantizer = EthosUQuantizer(compile_spec)
operator_config = get_symmetric_quantization_config(is_per_channel=False)
quantizer.set_global(operator_config)
graph_module = prepare_pt2e(graph_module, quantizer)
graph_module(*example_inputs)
graph_module = convert_pt2e(graph_module)
exported_program = torch.export.export(graph_module, example_inputs)

# Lower the exported program to the Ethos-U backend and save pte file.
edge_program_manager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[EthosUPartitioner(compile_spec)],
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
    ),
).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

with open("mv2_arm_ethos_u55.pte", "wb") as file:
    edge_program_manager.write_to_file(file)
```

### Partitioner API

`EthosUPartitioner` tries to partition as much of the model as possible. It will never delegate unsupported operators, but a user can pass additional checks to the constructor to avoid partitioning additional operators. To do this, subclass `OperatorSupportBase` and implement the function `is_node_supported`. A few such checks exist in `executorch.exir.backend.operator_support`:

- `DontPartition`: Don't partition operators based on operator type.
- `DontPartitionModule`: Don't partition operators based on which python module the operator comes from.
- `DontPartitionName`: Don't partition opertors based on the operator name.

### Quantization

A fully integer model is required for using the Arm Ethos-U backend. As discussed above, you can quantize floating point models with the the `EthosUQuantizer`. Quantizers are backend specific, which means the `EthosUQuantizer` is configured to quantize models correctly for the target.

## Runtime Integration

To run the model on-device, build the executorch library and EthosUDelegate using the script
`executorch/backends/arm/scripts/build_executorch.sh`.
Then build the arm executorch runtime using the script
`executorch/backends/arm/scripts/build_executor_runner.sh --pte=mv2_arm_ethos_u55.pte --target=ethos-u55-128`.

Finally, run the elf file on FVP using the script
`executorch/backends/arm/scripts/run_fvp.sh --elf=executorch/mv2_arm_ethos_u55/cmake-out/arm_executor_runner --target=ethos-u55-128`.

## See Also
- [Arm Ethos-U Backend Tutorial](tutorial-arm.md)
