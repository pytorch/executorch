# Quantization

The Vulkan backend currently supports execution of quantized linear layers,
where weights are symmetrically quantized to 8-bit or 4-bit with per output
channel or per group quantization scales.

Support for additional quantized operators and quantization schemes (i.e. static
+ dynamic quantized convolution, support for statically quantized linear) is
under active development and will be added soon.

### 4-bit quantization with torchao `quantize_`

The `quantize_` API from [torchao](https://github.com/pytorch/ao) allows for
more advanced quantization schemes, and is the quantization workflow needed to
access 4-bit quantization. 4-bit quantization is commonly used for LLMs.

Two options are available to execute linear layers with 4-bit quantization:

1. Dynamically quantized activations via `Int8DynamicActivationIntxWeightConfig`
2. Weight only quantization via `IntxWeightOnlyConfig`

Dynamically quantized activations can provide a significant boost in latency
compared to weight only quantization, since it allows GPUs to leverage
accelerated integer dot product instructions when computing matrix
multiplication.

Below is a simple example of quantizing a simple sequence of linear layers using
the `quantize_` API.

```python
import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.utils import unwrap_tensor_subclass


class LinearSequenceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64, bias=False)
        self.linear2 = torch.nn.Linear(64, 32, bias=False)
        self.linear3 = torch.nn.Linear(32, 16, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


linear_sequence_module = LinearSequenceModule()

M = 32
sample_inputs = (torch.randn(M, 128),)

group_size = 32

q_config_8da4w = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4, weight_granularity=PerGroup(group_size)
)

q_config_4w = IntxWeightOnlyConfig(
    weight_dtype=torch.int4, granularity=PerGroup(group_size)
)

quantize_(linear_sequence_module, q_config_8da4w)
unwrap_tensor_subclass(linear_sequence_module)

# Regular export path from here
exported_program = torch.export.export(linear_sequence_module, sample_inputs)

etvk_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[VulkanPartitioner()],
).to_executorch()
```

### 8-bit quantization with PT2E quantization

For 8-bit quantized linear layers, currently the only quantization scheme
supported is weight only quantization, with weights that are symmetrically
quantized to 8 bits with per output channel quantization scales.

To access this quantization mode, the PT2E quantization flow must be used. At a
high level, the steps to quantize a model are:

1) Create an instance of the `VulkanQuantizer` class and specify desired quantization behaviour
2) Use `torch.export.export` to prepare for quantization.
3) Call `prepare_pt2e` to prepare the exported graph for quantization.
4) Execute the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

For example:

```python
import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config,
    VulkanQuantizer,
)

from executorch.exir import to_edge_transform_and_lower

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torchao.utils import unwrap_tensor_subclass


class LinearSequenceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64, bias=False)
        self.linear2 = torch.nn.Linear(64, 32, bias=False)
        self.linear3 = torch.nn.Linear(32, 16, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


linear_sequence_module = LinearSequenceModule()

M = 32
# Create sample inputs
sample_inputs = (torch.randn(M, 128),)

# Setup quantizer
quantizer = VulkanQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_dynamic=False, weight_bits=8))

# Export the model
exported_program = torch.export.export(linear_sequence_module, sample_inputs)
graph_module = exported_program.module()

# Quantize the exported program with PT2E quantization flow
quantized_module = prepare_pt2e(graph_module, quantizer)
# Calibrate. In practice, this would be done by iterating over a real dataset
quantized_module(*sample_inputs)
quantized_module = convert_pt2e(quantized_module)

# Export once more
exported_program = torch.export.export(quantized_module, sample_inputs)

# Lower to vulkan
etvk_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[VulkanPartitioner()],
).to_executorch()
```
