# [Deprecated, Please Don't Use] Short Term Quantization Flow in Executorch

Note: this is deprecated, pelase use [this](./quantization_flow.md) instead.

High level flow for short term quantization flow in exeuctorch looks like the following: https://fburl.com/8pspa022

## 1. Make the model symbolically traceable with torch.fx
### Process
The flow uses [FX Graph Mode Quantization](https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization) to quantize the model, so we need to make the model symbolically traceable first. If the model is not symbolically traceable (failed when running step 2), please follow the [User Guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) to make changes to model.
### Result
The result in this step will be a symbolically traceable model, or a refactored model with the non-traceable parts being factored out in a separate module or function so that we can skip tracing them in the next step.
## 2. Quantization
### Process
Please take a look at the [post training static quantization tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html) or [post training dynamic quantization tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html) to learn about all the steps of quantization (some of the apis will need to be updated).
[Here](https://www.internalfb.com/code/fbsource/[ea0e2ae0a4a88529f17342e656e820a528ed5bcd]/fbcode/executorch/exir/tests/test_quant_fusion_pass.py?lines=26) is the most up to date flow, main APIs that's used to quantize the model would be:
* [`prepare_fx`](https://pytorch.org/docs/master/generated/torch.quantization.quantize_fx.prepare_fx.html#torch.quantization.quantize_fx.prepare_fx): used to insert observers to the model

  * The main argument that should be configured by user is [QConfigMapping](https://pytorch.org/docs/master/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping), which describes how a model should be quantized, e.g. quantize all linear modules with int8 static quantization/int8 dynamic quantization etc.
  * Please use [PrepareCustomConfig](https://pytorch.org/docs/master/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig) to skip the non traceable modules.
  * Another important argument is [BackendConfig](https://pytorch.org/docs/master/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig), which is a config the encodes the quantization capabilities of a specific backend, we're using a [default one for executorch](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/executorch.py) right now.
* (not an api) calibration: run the model through some sample data
* `_convert_to_reference_decomposed_fx`: short term private convert function to convert a observed model to a [reference quantized model](https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md#reference-quantized-model)
### Result
The result after these steps will be a reference quantized model, with quantize/dequantize operators that use decomposed Tensors. Example:

```
# Reference Quantized Pattern for quantized add
x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8)
y = torch.ops.quantized_decomposed.dequantize_per_tensor(y, y_scale, y_zero_point, y_qmin, y_qmax, torch.uint8)
out = x + y
out = torch.ops.quantized_decomposed.quantize_per_tensor(out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8)
```


* What do we mean by decomposed quantized tensor?
  * currently in PyTorch we have a quantized Tensor as a separate abstraction, see [this doc](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor) for more details, taking int8 quantized Tensor as an example, it stores int8 data, and some quantization parameters like scale and zero_point in the Tensor object
  * Using decomposed quantized Tensor means instead of using quantized Tensor as a separate abstraction as input and output of quantized ops, we just use the decomposed int8 data Tensor, scale and zero_point to represent a quantized Tensor, and we will pass around int8 Tensor, and store scale and zero_point in the signature of quantized operator

* Why using decomposed quantized tensor?
  * Current quantized Tensor has some drawbacks, the main ones are
    * coupling with core, which means whenever user wants to add new type of quantization, they will need to modify core
    * more burden for downstream systems to support new Tensor types, e.g. executorch has a separate runtime, if we use the current quantized Tensor abastraction, executorch runtime (lean mode) will also need to support this abstraction
  * For more informations about decomposed Tensor (prototypes and demos), please see https://fb.workplace.com/groups/2322282031156145/permalink/5674821579235490/ and https://fb.workplace.com/notes/1403734133487714
Note: for dynamic quantized linear, the pattern will be:
```
act_scale, act_zero_point = torch.ops.quantized_decomposed.choose_qparams(act, quant_min, quant_max, dtype)
act = torch.ops.quantized_decomposed.quantize_per_tensor(act, act_scale, act_zero_point, act_qmin, act_qmax, torch.uint8)
act = torch.ops.quantized_decomposed.dequantize_per_tensor(act, act_scale, act_zero_point, act_qmin, act_qmax, torch.uint8)
weight = torch.ops.quantized_decomposed.dequantize_per_tensor(y, y_scale, y_zero_point, y_qmin, y_qmax, torch.uint8)
out = torch.nn.functional.linear(act, weight, bias)
```
Also here we have ops in torch API (e.g. `torch.nn.functional.linear`), but in the fusion passes, we'll be working with aten operators and all the torch ops, modules will be traced as aten ops.

## 3. exir.capture
### Process
`m = exir.capture(m, example_inputs)`

We'll call exir.capture to capture the graph to a representation using aten operators.
### Result
A model with aten operators (in [EXIR - ATen dialect](https://www.figma.com/file/l1f1UXfjofLT6D1HqDwp93/Executorch-Compilation-Flow?node-id=0%3A1&t=1c2UKQXUZsNeENDR-0)), reference quantized pattern will be expressed with aten operators as well.

## 4. Lowering through Delegation
### Process
In this step we need to recognize reference quantized pattern for quantized operators, e.g. "dq - linear - q" and lower the pattern to delegation modules, since some quantized operators needs to run in special libraries (e.g. xnnpack) or runtime (e.g. GPU), so we have a delegation flow for them, see [this test](https://www.internalfb.com/code/fbsource/[ea0e2ae0a4a88529f17342e656e820a528ed5bcd]/fbcode/executorch/exir/tests/test_quant_lowering_custom_backend_pass.py?lines=404) for a end to end example with delegation. We are still working on how to do weight prepacking in delegation at the moment and will have an update for the a bit later. But [here](https://fb.workplace.com/notes/1307520240058002) is the delegation API.
Main things are (1) implement delegation module (2) implement lowering (partitioner) to delegation module.
Code (extracted from the [test](https://www.internalfb.com/code/fbsource/[ea0e2ae0a4a88529f17342e656e820a528ed5bcd]/fbcode/executorch/exir/tests/test_quant_lowering_custom_backend_pass.py?lines=404)):
```
# duplicate dequant op
m = m.to_edge(exir.EdgeCompileConfig(passes=[DuplicateDequantNodePass()]))
m = to_backend(m, QuantizedConvAddOpPartitioner)
```
### Result
A partially lowered model with some reference quantized patterns been replaced by calls to delegation modules (lowered_module.execute(...))

## 5. Lowering to quantized operators
### Process
Another path is to implement quantized operator and lower the reference quantized pattern to quantized operators.
See [this diff](https://www.internalfb.com/diff/D39974289) for an example implementation for quantized add. Basically we need to

(1). Implement a functional quantized operator in quantized_decomposed namespace in [//exir/passes/_quant_patterns_and_replacements.py](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py). Notice that these operators are categorized as backend operators since they are meaningful to the target backends.

(2). Implement the out variant quantized operator in quantized_decomposed namespace in //kernels/quantized/op_QOP.cpp and add test to //kernels/quantized/test/op_QOP_test.cpp
Also we need to make sure the operator here matches the operator in (1) in signature so that ToOutVar pass can establish the connection between these two ops. Example [here](https://github.com/pytorch/executorch/blob/main/kernels/quantized/cpu/op_add.cpp).

(3). Implement lowering pass from reference quantized pattern to the functional quantized operator we write in (1), some examples for quantized add can be found in [//exir/passes/_quant_patterns_and_replacements.py](https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py)

Example Code (extracted from [this test](https://github.com/pytorch/executorch/blob/main/exir/tests/test_quant_fusion_pass.py)):
```
m = exir.capture(m, example_inputs).to_edge().to_executorch(ExecutorchBackendConfig(passes=[QuantFusionPass()]))
```
### Result
A fully lowered quantized model, with both delegated quantized modules and functioanl quantized operators

## 6. to_executorch
### Process
In this last step, we just call to_executorch to convert the functional variant operators to out variant operators.
### Result
A fully lowered quantized model, with both delegated quantized modules and out variant quantized operators.
