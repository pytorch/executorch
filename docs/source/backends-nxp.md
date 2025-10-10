# NXP eIQ Neutron Backend

This manual page is dedicated to introduction of using the ExecuTorch with NXP eIQ Neutron Backend.
NXP offers accelerated machine learning models inference on edge devices.
To learn more about NXP's machine learning acceleration platform, please refer to [the official NXP website](https://www.nxp.com/applications/technologies/ai-and-machine-learning:MACHINE-LEARNING).

<div class="admonition tip">
For up-to-date status about running ExecuTorch on Neutron Backend please visit the <a href="https://github.com/pytorch/executorch/blob/main/backends/nxp/README.md">manual page</a>.
</div>

## Features

ExecuTorch v1.0 supports running machine learning models on selected NXP chips (for now only i.MXRT700).
Among currently supported machine learning models are:
- Convolution-based neutral networks
- Full support for MobileNetV2 and CifarNet

## Prerequisites (Hardware and Software)

In order to successfully build ExecuTorch project and convert models for NXP eIQ Neutron Backend you will need a computer running Linux.

If you want to test the runtime, you'll also need:
- Hardware with NXP's [i.MXRT700](https://www.nxp.com/products/i.MX-RT700) chip or a testing board like MIMXRT700-AVK
- [MCUXpresso IDE](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-integrated-development-environment-ide:MCUXpresso-IDE) or [MCUXpresso Visual Studio Code extension](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-for-visual-studio-code:MCUXPRESSO-VSC)

## Using NXP backend 

To test converting a neural network model for inference on NXP eIQ Neutron Backend, you can use our example script:

```shell
# cd to the root of executorch repository
./examples/nxp/aot_neutron_compile.sh [model (cifar10 or mobilenetv2)]
```

For a quick overview how to convert a custom PyTorch model, take a look at our [example python script](https://github.com/pytorch/executorch/tree/release/1.0/examples/nxp/aot_neutron_compile.py).

### Partitioner API

The partitioner is defined in `NeutronPartitioner` in `backends/nxp/neutron_partitioner.py`. It has the following 
arguments:
* `compile_spec` - list of key-value pairs defining compilation. E.g. for specifying platform (i.MXRT700) and Neutron Converter flavor.
* `custom_delegation_options` - custom options for specifying node delegation.

### Quantization

The quantization for Neutron Backend is defined in `NeutronQuantizer` in `backends/nxp/quantizer/neutron_quantizer.py`. 
The quantization follows PT2E workflow, INT8 quantization is supported. Operators are quantized statically, activations
follow affine and weights symmetric per-tensor quantization scheme.

#### Supported operators

List of Aten operators supported by Neutron quantizer:

`abs`, `adaptive_avg_pool2d`, `addmm`, `add.Tensor`, `avg_pool2d`, `cat`, `conv1d`, `conv2d`, `dropout`,
`flatten.using_ints`, `hardtanh`, `hardtanh_`, `linear`, `max_pool2d`, `mean.dim`, `pad`, `permute`, `relu`, `relu_`,
`reshape`, `view`, `softmax.int`, `sigmoid`, `tanh`, `tanh_`

#### Example
```python
import torch
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Prepare your model in Aten dialect
aten_model = get_model_in_aten_dialect()
# Prepare calibration inputs, each tuple is one example, example tuple has items for each model input
calibration_inputs: list[tuple[torch.Tensor, ...]] = get_calibration_inputs()
quantizer = NeutronQuantizer()

m = prepare_pt2e(aten_model, quantizer)
for data in calibration_inputs:
    m(*data)
m = convert_pt2e(m)
```

## Runtime Integration

To learn how to run the converted model on the NXP hardware, use one of our example projects on using ExecuTorch runtime from MCUXpresso IDE example projects list.
For more finegrained tutorial, visit [this manual page](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/example_applications.html).
