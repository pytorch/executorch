# Quantization Overview

Quantization is a technique that reduces the precision of numbers used in a model’s computations and stored weights—typically from 32-bit floats to 8-bit integers. This reduces the model’s memory footprint, speeds up inference, and lowers power consumption, often with minimal loss in accuracy.

Quantization is especially important for deploying models on edge devices such as wearables, embedded systems, and microcontrollers, which often have limited compute, memory, and battery capacity. By quantizing models, we can make them significantly more efficient and suitable for these resource-constrained environments.


# Quantization in ExecuTorch
ExecuTorch uses [torchao](https://github.com/pytorch/ao/tree/main/torchao) as its quantization library. This integration allows ExecuTorch to leverage PyTorch-native tools for preparing, calibrating, and converting quantized models.


Quantization in ExecuTorch is backend-specific. Each backend defines how models should be quantized based on its hardware capabilities. Most ExecuTorch backends use the torchao [PT2E quantization](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html) flow, which works on models exported with torch.export and enables quantization that is tailored for each backend.

The PT2E quantization workflow has three main steps:

1. Configure a backend-specific quantizer.
2. Prepare, calibrate, convert, and evaluate the quantized model in PyTorch
3. Lower the model to the target backend

## 1. Configure a Backend-Specific Quantizer

Each backend provides its own quantizer (e.g., XNNPACKQuantizer, CoreMLQuantizer) that defines how quantization should be applied to a model in a way that is compatible with the target hardware.
These quantizers usually support configs that allow users to specify quantization options such as:

* Precision (e.g., 8-bit or 4-bit)
* Quantization type (e.g., dynamic, static, or weight-only quantization)
* Granularity (e.g., per-tensor, per-channel)

Not all quantization options are supported by all backends. Consult backend-specific guides for supported quantization modes and configuration, and how to initialize the backend-specific PT2E quantizer:

* [XNNPACK quantization](backends/xnnpack/xnnpack-quantization.md)
* [CoreML quantization](backends/coreml/coreml-quantization.md)
* [QNN quantization](backends-qualcomm.md#step-2-optional-quantize-your-model)



## 2. Quantize and evaluate the model

After the backend specific quantizer is defined, the PT2E quantization flow is the same for all backends.  A generic example is provided below, but specific examples are given in backend documentation:

```python
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

training_gm = torch.export.export(model, sample_inputs).module()

# Prepare the model for quantization using the backend-specific quantizer instance
prepared_model = prepare_pt2e(training_gm, quantizer)


# Calibrate the model on representative data
for sample in calibration_data:
	prepared_model(sample)

# Convert the calibrated model to a quantized model
quantized_model = convert_pt2e(prepared_model)
```

The quantized_model is a PyTorch model like any other, and can be evaluated on different tasks for accuracy.
Tasks specific benchmarks are the recommended way to evaluate your quantized model, but as crude alternative you can compare to outputs with the original model using generic error metrics like SQNR:

```python
from torchao.quantization.utils import compute_error
out_reference = model(sample)
out_quantized = quantized_model(sample)
sqnr = compute_error(out_reference, out_quantized) # SQNR error
```

Note that numerics on device can differ those in PyTorch even for unquantized models, and accuracy evaluation can also be done with pybindings or on device.


## 3. Lower the model

The final step is to lower the quantized_model to the desired backend, as you would an unquantized one.  See [backend-specific pages](backends-overview.md) for lowering information.
