# Quantization

The XNNPACK delegate can also be used as a backend to execute symmetrically quantized models. To quantize a PyTorch model for the XNNPACK backend, use the `XNNPACKQuantizer`. `Quantizers` are backend specific, which means the `XNNPACKQuantizer` is configured to quantize models to leverage the quantized operators offered by the XNNPACK Library.

### Supported Quantization Schemes
The XNNPACK delegate supports the following quantization schemes:

- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
  - Supports both static and dynamic activations.
  - Supports per-channel and per-tensor schemes.
  - Supports linear, convolution, add, mul, cat, and adaptive avg pool 2d operators.

Weight-only quantization is not currently supported on XNNPACK.

### 8-bit Quantization using the PT2E Flow

To perform 8-bit quantization with the PT2E flow, perform the following steps prior to exporting the model:

1) Create an instance of the `XnnpackQuantizer` class. Set quantization parameters.
2) Use `torch.export.export` to prepare for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) For static quantization, run the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

qparams = get_symmetric_quantization_config(is_per_channel=True) # (1)
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

et_program = to_edge_transform_and_lower( # (6)
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[XnnpackPartitioner()],
).to_executorch()
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### LLM quantization with quantize_

The XNNPACK backend also supports quantizing models with the [torchao](https://github.com/pytorch/ao) quantize_ API.  This is most commonly used for LLMs, requiring more advanced quantization.  Since quantize_ is not backend aware, it is important to use a config that is compatible with CPU/XNNPACK:

* Quantize embeedings with IntxWeightOnlyConfig (with weight_dtype torch.int2, torch.int4, or torch.int8, using PerGroup or PerAxis granularity)
* Quantize linear layers with Int8DynamicActivationIntxWeightConfig (with weight_dtype=torch.int4, using PerGroup or PerAxis granularity)

Below is a simple example, but a more detailed tutorial including accuracy evaluation on popular LLM benchmarks can be found in the [torchao documentation](https://docs.pytorch.org/ao/main/serving.html#mobile-deployment-with-executorch).

```python
from torchao.quantization.granularity import PerGroup, PerAxis
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    quantize_,
)

# Quantize embeddings with 8-bits, per channel
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int8,
    granularity=PerAxis(0),
)
qunatize_(
    eager_model,
    lambda m, fqn: isinstance(m, torch.nn.Embedding),
)


# Quatize linear layers with 8-bit dynamic activations and 4-bit weights
linear_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
)
quantize_(eager_model, linear_config)
```
