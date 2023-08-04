# Quantization Flow in Executorch

High level flow for short term quantization flow in exeuctorch looks like the following: https://docs.google.com/document/d/1UuktDffiMH0rXRuiL0e8bQaS3X0XfkIHEF1VtpFmA5A/edit#heading=h.mywdosyratgh

## 1. Capture the model with `exir.capture`
### Process
The flow uses `PyTorch 2.0 Export Quantization` to quantize the model, that works on a model captured by `exir.capture`. If the model is not traceable, please follow the [User Guide](TBD) to make changes to model, and see [here](https://pytorch.org/docs/main/generated/exportdb/index.html) for supported constructs in `exir.capture`.

```
# program capture
from executorch import exir
from executorch.exir import CaptureConfig
exported_program = exir.capture(m, example_inputs)
m = exported_program.graph_module
```
### Result
The result in this step will be a capturable model (`fx.GraphModule`)
## 2. Quantization
### Process
Please take a look at the [pytorch 2.0 export post training static quantization tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html) to learn about all the steps of quantization. Main APIs that's used to quantize the model would be:
* `prepare_pt2e`: used to insert observers to the model, it takes a backend specific `Quantizer` as argument, which will annotate the nodes with informations needed to quantize the model properly for the backend
* (not an api) calibration: run the model through some sample data
* `convert_pt2e`: convert a observed model to a quantized model, we have special representation for selected ops (e.g. quantized linear), other ops are represented as (dq -> float32_op -> q), and q/dq are decomposed into more primitive operators.

### Result
The result after these steps will be a reference quantized model, with quantize/dequantize operators being further decomposed. Example:

(TODO): update
```
# Reference Quantized Pattern for quantized add
x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8)
y = torch.ops.quantized_decomposed.dequantize_per_tensor(y, y_scale, y_zero_point, y_qmin, y_qmax, torch.uint8)
out = x + y
out = torch.ops.quantized_decomposed.quantize_per_tensor(out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8)
```

see https://docs.google.com/document/d/17h-OEtD4o_hoVuPqUFsdm5uo7psiNMY8ThN03F9ZZwg/edit#heading=h.ov8z39149wy8 for some operators that has integer operator representations.

## 4. Lowering to Executorch
TODO: link
