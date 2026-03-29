# NXP eIQ Neutron Quantization

The eIQ Neutron NPU requires the operators delegated to be quantized. To quantize the PyTorch model for the Neutron backend, use the `NeutronQuantizer` from `backends/nxp/quantizer/neutron_quantizer.py`.
The `NeutronQuantizer` is configured to quantize the model with quantization scheme supported by the eIQ Neutron NPU.

### Supported Quantization Schemes

The Neutron delegate supports the following quantization schemes:

- Static quantization with 8-bit symmetric weights and 8-bit asymmetric activations (via the PT2E quantization flow), per-tensor granularity.
    - Following operators are supported at this moment: 
      - `aten.abs.default`
      - `aten.adaptive_avg_pool2d.default`
      - `aten.addmm.default`
      - `aten.add.Tensor`
      - `aten.avg_pool2d.default`
      - `aten.cat.default`
      - `aten.conv1d.default`
      - `aten.conv2d.default`
      - `aten.dropout.default`
      - `aten.flatten.using_ints`
      - `aten.hardtanh.default`
      - `aten.hardtanh_.default`
      - `aten.linear.default`
      - `aten.max_pool2d.default`
      - `aten.mean.dim`
      - `aten.mul.Tensor`
      - `aten.pad.default`
      - `aten.permute.default`
      - `aten.relu.default` and `aten.relu_.default`
      - `aten.reshape.default`
      - `aten.view.default`
      - `aten.softmax.int`
      - `aten.tanh.default`,  `aten.tanh_.default`
      - `aten.sigmoid.default`
      - `aten.slice_copy.Tensor`

### Static 8-bit Quantization Using the PT2E Flow

To perform 8-bit quantization with the PT2E flow, perform the following steps prior to exporting the model to edge:

1) Create an instance of the `NeutronQuantizer` class.
2) Use `torch.export.export` to export the model to ATen Dialect.
3) Call `prepare_pt2e` with the instance of the `NeutronQuantizer` to annotate the model with observers for quantization.
4) As static quantization is required, run the prepared model with representative samples to calibrate the quantized tensor activation ranges.
5) Call `convert_pt2e` to quantize the model.
6) Export and lower the model using the standard flow.

The output of `convert_pt2e` is a PyTorch model which can be exported and lowered using the normal flow. As it is a regular PyTorch model, it can also be used to evaluate the accuracy of the quantized model using standard PyTorch techniques.

To quantize the model, you can use the PT2E workflow: 

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

neutron_target_spec = NeutronTargetSpec(target="imxrt700")
quantizer = NeutronQuantizer(neutron_target_spec) # (1)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

compile_spec = generate_neutron_compile_spec(
    "imxrt700",
    operators_not_to_delegate=None,
)

et_program = to_edge_transform_and_lower( # (6)
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[NeutronPartitioner(compile_spec=compile_spec)],
).to_executorch()
```

Or you can use the predefined function for post training quantization from NXP Backend implementation: 
```python
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize

...

target_spec = NeutronTargetSpec(target="imxrt700")
quantized_graph_module = calibrate_and_quantize(
    aten_model,
    calibration_inputs,
    NeutronQuantizer(neutron_target_spec=target_spec),
)
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.

### Quantization Aware Training

The NeutronQuantizer supports two modes of quantization: *Post‑Training Quantization (PTQ)* and *Quantization Aware Training (QAT)*.
PTQ uses a calibration phase to tune quantization parameters on an already‑trained model in order to obtain a model with integer weights.
While this optimization reduces model size, it introduces quantization noise and can degrade the model's performance.
Compared to PTQ, QAT enables the model to adapt its weights to the introduced quantization noise.
In QAT, instead of calibration we run training to optimize both quantization parameters and model weights at the same time.

See the [Quantization Aware Training blog post](https://pytorch.org/blog/quantization-aware-training/) for an introduction to the QAT method.

To use QAT with the Neutron backend, toggle the `is_qat` parameter:

```python
from executorch.backends.nxp.quantizer.neutron_quantizer import (
    NeutronQuantizer,
    NeutronTargetSpec,
)

target_spec = NeutronTargetSpec(target="imxrt700")
neutron_quantizer = NeutronQuantizer(neutron_target_spec=target_spec, is_qat=True)
```

The rest of the quantization pipeline works similarly to the PTQ workflow.
The most significant change is that the calibration step is replaced by training.

<div class="admonition tip">
Note: QAT uses <code>prepare_qat_pt2e</code> prepare function instead of <code>prepare_pt2e</code>.
</div>

```python
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e import (
    move_exported_model_to_eval,
    move_exported_model_to_train,
    disable_observer,
)

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()

neutron_target_spec = NeutronTargetSpec(target="imxrt700")
quantizer = NeutronQuantizer(neutron_target_spec, is_qat=True) # (1)

sample_inputs = (torch.randn(1, 3, 224, 224),)
training_ep = torch.export.export(model, sample_inputs).module() # (2)

## Steps different from PTQ (3–6)
prepared_model = prepare_qat_pt2e(training_ep, quantizer) # (3) !!! Different prepare function
prepared_model = move_exported_model_to_train(prepared_model) # (4)

# ---------------- Training phase (5) ----------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-2, momentum=0.9)

train_data = datasets.ImageNet("./", split="train", transform=...)
train_loader = DataLoader(train_data, batch_size=5)

# Training replaces calibration in QAT
for epoch in range(num_epochs):
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = prepared_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # It is recommended to disable quantization params
    # updates after few epochs of training.
    if epoch >= num_epochs / 3:
        model.apply(disable_observer)
# --------------- End of training phase ---------------

prepared_model = move_exported_model_to_eval(prepared_model) # (6)
quantized_model = convert_pt2e(prepared_model) # (7)

# Optional step - fixes biasless convolution (see Known Limitations of QAT)
quantized_model = QuantizeFusedConvBnBiasAtenPass(
    default_zero_bias=True
)(quantized_model).graph_module

...
```

Moving from PTQ to QAT check-list:
- Set `is_qat=True` in `NeutronQuantizer`
- Use `prepare_qat_pt2e` instead of `prepare_pt2e`
- Call `move_exported_model_to_train()` before training
- Train the model instead of calibrating
- Call `move_exported_model_to_eval()` after training

#### Known limitations of QAT

In the current ExecuTorch/TorchAO implementation, there is an issue when quantizing biasless convolutions during QAT.
The pipeline produces a non‑quantized empty bias, which causes the Neutron Converter to fail.
To mitigate this issue, use the `QuantizeFusedConvBnBiasAtenPass` post‑quantization:

```python
...

# training

prepared_model = move_exported_model_to_eval(prepared_model) # (6)
quantized_model = convert_pt2e(prepared_model) # (7)

quantized_model = QuantizeFusedConvBnBiasAtenPass(
    default_zero_bias=True
)(quantized_model).graph_module

...
```
