# Quantization

To quantize a PyTorch model for the Core ML backend, use the `CoreMLQuantizer`.  `Quantizers` are backend specific, which means the `CoreMLQuantizer` is configured to quantize models to leverage the quantized operators offered by the Core ML backend.

### Supported Quantization Schemes

The CoreML delegate supports the following quantization schemes:

- 8-bit static and weight-only quantization via the PT2E flow; dynamic quantization is not supported by CoreML.
- 4-bit weight-only affine quantization (per-group or per-channel) via the quantize_ flow
- 1-8 bit weight-only LUT quantization (per grouped-channel) via the quantize_ flow

### 8-bit Quantization using the PT2E Flow

Quantization with the Core ML backend requires exporting the model for iOS 17 or later.
To perform 8-bit quantization with the PT2E flow, follow these steps:

1) Create a [`coremltools.optimize.torch.quantization.LinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig) and use it to create an instance of a `CoreMLQuantizer`.
2) Use `torch.export.export` to export a graph module that will be prepared for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) Run the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

```python
import torch
import coremltools as ct
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.compiler import CoreMLBackend

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# Step 1: Define a LinearQuantizerConfig and create an instance of a CoreMLQuantizer
# Note that "linear" here does not mean only linear layers are quantized, but that linear (aka affine) quantization
# is being performed
static_8bit_config = ct.optimize.torch.quantization.LinearQuantizerConfig(
    global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        weight_per_channel=True,
    )
)
quantizer = CoreMLQuantizer(static_8bit_config)

# Step 2: Export the model for training
training_gm = torch.export.export(mobilenet_v2, sample_inputs).module()

# Step 3: Prepare the model for quantization
prepared_model = prepare_pt2e(training_gm, quantizer)

# Step 4: Calibrate the model on representative data
# Replace with your own calibration data
for calibration_sample in [torch.randn(1, 3, 224, 224)]:
    prepared_model(calibration_sample)

# Step 5: Convert the calibrated model to a quantized model
quantized_model = convert_pt2e(prepared_model)

# Step 6: Export the quantized model to Core ML
et_program = to_edge_transform_and_lower(
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[
        CoreMLPartitioner(
             # iOS17 is required for the quantized ops in this example
            compile_specs=CoreMLBackend.generate_compile_specs(
                minimum_deployment_target=ct.target.iOS17
            )
        )
    ],
).to_executorch()
```

The above does static quantization (activations and weights are quantized).

You can see a full description of available quantization configs in the [coremltools documentation](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig).  For example, the config below will perform weight-only quantization:

```
weight_only_8bit_config = ct.optimize.torch.quantization.LinearQuantizerConfig(
    global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        activation_dtype=torch.float32,
        weight_dtype=torch.qint8,
        weight_per_channel=True,
    )
)
quantizer = CoreMLQuantizer(weight_only_8bit_config)
```

Quantizing activations requires calibrating the model on representative data.  Also note that PT2E currently requires passing at least 1 calibration sample before calling `convert_pt2e`, even for data-free weight-only quantization.

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### LLM quantization with quantize_

The Core ML backend also supports quantizing models with the [torchao](https://github.com/pytorch/ao) quantize_ API.  This is most commonly used for LLMs, requiring more advanced quantization.  Since quantize_ is not backend aware, it is important to use a config that is compatible with Core ML:

* Quantize embedding/linear layers with IntxWeightOnlyConfig (with weight_dtype torch.int4 or torch.int8, using PerGroup or PerAxis granularity).  Using 4-bit or PerGroup quantization requires exporting with minimum_deployment_target >= ct.target.iOS18.  Using 8-bit quantization with per-axis granularity is supported on ct.target.IOS16+.  See [Core ML `CompileSpec`](coreml-partitioner.md#coreml-compilespec) for more information on setting the deployment target.
* Quantize embedding/linear layers with CodebookWeightOnlyConfig (with dtype torch.uint1 through torch.uint8, using various block sizes).  Quantizing with CodebookWeightOnlyConfig requires exporting with minimum_deployment_target >= ct.target.iOS18, see [Core ML `CompileSpec`](coreml-partitioner.md#coreml-compilespec) for more information on setting the deployment target.

Below is an example that quantizes embeddings to 8-bits per-axis and linear layers to 4-bits using group_size=32 with affine quantization:

```python
from torchao.quantization.granularity import PerGroup, PerAxis
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    quantize_,
)

# Quantize embeddings with 8-bits, per channel
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int8,
    granularity=PerAxis(0),
)
quantize_(
    eager_model,
    embedding_config,
    lambda m, fqn: isinstance(m, torch.nn.Embedding),
)

# Quantize linear layers with 4-bits, per-group
linear_config = IntxWeightOnlyConfig(
    weight_dtype=torch.int4,
    granularity=PerGroup(32),
)
quantize_(
    eager_model,
    linear_config,
)
```

Below is another example that uses codebook quantization to quantize both embeddings and linear layers to 3-bits.
In the coremltools documentation, this is called [palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html):

```
from torchao.quantization.quant_api import (
    quantize_,
)
from torchao.prototype.quantization.codebook_coreml import CodebookWeightOnlyConfig

quant_config = CodebookWeightOnlyConfig(
    dtype=torch.uint3,
    # There is one LUT per 16 rows
    block_size=[16, -1],
)

quantize_(
    eager_model,
    quant_config,
    lambda m, fqn: isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.Linear),
)
```

Both of the above examples will export and lower to Core ML with the to_edge_transform_and_lower API.
