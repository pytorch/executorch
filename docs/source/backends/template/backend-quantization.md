# {BACKEND_NAME} Quantization

Document quantization schemes and flows for the backend. This should include a description of each scheme and a code example to perform quantization. Example sections for PT2E and quantize_ are included below, to be replaced with details for the target backend.

### Supported Quantization Schemes
The {BACKEND_NAME} delegate supports the following quantization schemes:

- {QUANTIZATION_SCHEME_1}
- {QUANTIZATION_SCHEME_2}

### {QUANTIZATION_METHOD_1} using the PT2E Flow

To perform {QUANTIZATION_METHOD_1} with the PT2E flow, perform the following steps prior to exporting the model:

1) Create an instance of the `{BackendName}Quantizer` class. Set quantization parameters.
2) Use `torch.export.export` to prepare for quantization.
3) Call `prepare_pt2e` to prepare the model for quantization.
4) For static quantization, run the prepared model with representative samples to calibrate the quantized tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

```python
import torch
import {MODEL_IMPORT_PATH} as models
from {MODEL_WEIGHTS_IMPORT}
from executorch.backends.{backend_name}.quantizer.{backend_name}_quantizer import {BackendName}Quantizer, {get_quantization_config_function}
from executorch.backends.{backend_name}.partition.{backend_name}_partitioner import {BackendName}Partitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = models.{model_name}.{model_function}(weights={ModelWeights}.DEFAULT).eval()
sample_inputs = ({SAMPLE_INPUT_SHAPE}, )

qparams = {get_quantization_config_function}({QUANTIZATION_PARAMS}) # (1)
quantizer = {BackendName}Quantizer()
quantizer.set_global(qparams)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [{CALIBRATION_SAMPLE}]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

et_program = to_edge_transform_and_lower( # (6)
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[{BackendName}Partitioner()],
).to_executorch()
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### LLM Quantization with quantize_

The {BACKEND_NAME} backend also supports quantizing models with the [torchao](https://github.com/pytorch/ao) quantize_ API. {ADVANCED_QUANTIZATION_DESCRIPTION}

Below is a simple example, but a more detailed tutorial including accuracy evaluation on popular benchmarks can be found in the [torchao documentation]({TORCHAO_DOCS_URL}).

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
