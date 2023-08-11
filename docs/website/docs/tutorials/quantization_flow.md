# Quantization Flow in Executorch

## 1. Capture the model with `export.capture_pre_autograd_graph`
### Process
The flow uses `PyTorch 2.0 Export Quantization` to quantize the model, that works on a model captured by `exir.capture`. If the model is not traceable, please see [here](https://pytorch.org/docs/main/generated/exportdb/index.html) for supported constructs in `export.capture_pre_autograd_graph` and how to make the model exportable.

```
# program capture
from torch._export import export

m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
```
### Result
The result in this step will be a `fx.GraphModule`

## 2. Quantization
### Process
Note: Before quantizing models, each backend need to implement their own `Quantizer` by following [this tutorial](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html).

Please take a look at the [pytorch 2.0 export post training static quantization tutorial](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html) to learn about all the steps of quantization. Main APIs that's used to quantize the model would be:
* `prepare_pt2e`: used to insert observers to the model, it takes a backend specific `Quantizer` as argument, which will annotate the nodes with informations needed to quantize the model properly for the backend
* (not an api) calibration: run the model through some sample data
* `convert_pt2e`: convert a observed model to a quantized model.


### Result
The result after these steps will be a reference quantized model, with quantize/dequantize operators being further decomposed. Example:

#### Q/DQ Representation (default)
We'll have (dq -> float32_op -> q) representation for all quantized operators

```
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    weight_permuted = torch.ops.aten.permute_copy.default(weight_fp32, [1, 0]);
    out_fp32 = torch.ops.aten.addmm.default(bias_fp32, x_fp32, weight_permuted)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8
```


#### Reference Quantized Model Representation
(WIP, expected to be ready at end of August): we have special representation for selected ops (e.g. quantized linear), other ops are represented as (dq -> float32_op -> q), and q/dq are decomposed into more primitive operators.

You can get this representation by:
`convert_pt2e(..., use_reference_representation=True)`

```
# Reference Quantized Pattern for quantized linear
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_int16 = x_int8.to(torch.int16)
    weight_int16 = weight_int8.to(torch.int16)
    acc_int32 = torch.ops.out_dtype(torch.mm, torch.int32, (x_int16 - x_zero_point), (weight_int16 - weight_zero_point))
    acc_rescaled_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, torch.int32, acc_int32, x_scale * weight_scale / output_scale)
    bias_scale = x_scale * weight_scale
    bias_int32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, bias_fp32, bias_scale / out_scale)
    out_int8 = torch.ops.aten.clamp(acc_rescaled_int32 + bias_int32 + output_zero_point, qmin, qmax).to(torch.int8)
    return out_int8
```

See [here](https://docs.google.com/document/d/17h-OEtD4o_hoVuPqUFsdm5uo7psiNMY8ThN03F9ZZwg/edit#heading=h.ov8z39149wy8) for some operators that has integer operator representations.

## 4. Lowering to Executorch
You can lower the quantized model to executorch by following [this tutorial](https://github.com/pytorch/executorch/blob/main/docs/website/docs/tutorials/exporting_to_executorch.md#12-lower-to-exir-edge-dialect).
