# Quantization

The Exynos backend currently supports executing statically quantized 8-bit models.

### 8-bit quantization with the PT2E quantization flow

To perform 8-bit quantization with the PT2E flow, perform the following steps prior to exporting the model:

1) Create an instance of the `EnnQuantizer` class and set the desired quantization behaviour.
2) Use `torch.export.export` to obtain a graph module representation of the source model.
3) Use `prepare_pt2e` to prepare the model for quantization.
4) Execute the prepared model with representative samples to calibrate the quantizated tensor activation ranges.
5) Use `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard export flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using
the same export flow as non-quantized models. As it is a regular PyTorch model, it can
also be used to evaluate the accuracy of the quantized model using standard PyTorch
techniques.

The below example shows how to quantize a MobileNetV2 model using the PT2E quantization flow.

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer, Precision

from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# Currently, "A8W8" is the only supported precision mode
precision = "A8W8"
is_per_channel = True
is_qat = False

quantizer = EnnQuantizer()
quantizer.set_quant_params(precision, is_per_channel, is_qat) # (1)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

et_program = to_edge_transform_and_lower( # (6)
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[EnnPartitioner()],
).to_executorch()
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html)
for more information.
