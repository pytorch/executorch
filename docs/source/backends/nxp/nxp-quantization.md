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

target_spec = NeutronTargetSpec(target="imxrt700", converter_flavor="SDK_25_09")
quantizer = NeutronQuantizer(neutron_target_spec) # (1)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

compile_spec = generate_neutron_compile_spec(
    "imxrt700",
    operators_not_to_delegate=None,
    neutron_converter_flavor="SDK_25_06",
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

target_spec = NeutronTargetSpec(target="imxrt700", converter_flavor="SDK_25_09")
quantized_graph_module = calibrate_and_quantize(
    aten_model,
    calibration_inputs,
    NeutronQuantizer(neutron_target_spec=target_spec),
)
```

See [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) for more information.
