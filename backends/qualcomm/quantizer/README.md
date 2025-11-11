# Contribution for Operator Annotation
Thank you for contributing to Qualcomm AI Engine Direct delegate for ExecuTorch. Reading and following these guidelines will help you quickly get the essentials of annotating an operator in `QnnQuantizer` to unblock yourself and land pull requests more efficiently.

## Sections
* [References](#references)
* [Getting Started](#getting-started)
* [Issues](#issues)
* [Pull Requests](#pull-requests)

## References
### Qualcomm AI Engine Direct
- [Operator Definitions for HTP](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html)

### PyTorch
- [ATen Operator Definitions](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native)

## Getting Started
Before extending operator for quantization annotation, please make sure the operator builder has been well-implemented (learn more on this [tutorial](../builders/README.md)).
### Behavior of Annotation
In order to conduct PTQ for floating point precision graph, observers are required to be inserted after each graph nodes. The observed numeric range will go through different algorithms and return statistics of `scale`, `offset` to represent data in fixed point.<br/><br/>
**Stages could be shown as**:
- Floating point `nn.Module` after `torch.export.export`
    ```mermaid
    flowchart TB
        input & kernel & bias --> id1(convolution) --> output
    ```

- Inserting observers for inspecting numeric range
    ```mermaid
    flowchart TB
        input --> id2(input_act_obs) --> id1(convolution) --> id3(output_act_obs) --> output
        kernel --> id4(weight_obs) --> id1(convolution)
        bias --> id5(bias_obs) --> id1(convolution)
    ```

- Cascade QDQ pairs after landing encodings
    ```mermaid
    flowchart TB
        input --> id2(Q_i) --> id3(DQ_i) --> id1(convolution) --> id4(Q_o) --> id5(DQ_o) --> output
        kernel --> id6(Q_k) --> id7(DQ_k) --> id1(convolution)
        bias --> id8(Q_b) --> id9(DQ_b) --> id1(convolution)
    ```
Qualcomm backend will consume the generated encodings and lower operators with fixed precision. This tutorial will guide you through the details of inserting observer and some useful utilies.

### Register Annotation via Operator Type
Let's start with hooking callback for designated operator target:
```python
def register_annotator(ops: List[OpOverload]):
    def decorator(annotator: Callable):
        for op in ops:
            OP_ANNOTATOR[op] = annotator

    return decorator
```
The `register_annotator` decorator provides a convenient way to attach your own annotation logic, which requires list of operator type as its input argument.<br/> For example, the torch activation functions have `copy`, `in-place` implementation with small difference appears in naming (an extra `_` postfix), which will map to the same [Core ATen](https://pytorch.org/docs/stable/torch.compiler_ir.html) operators after `to_edge`:
```python
@register_annotator([torch.ops.aten.relu.default, torch.ops.aten.relu_.default])
```
Where `torch.ops.aten.relu.default` / `torch.ops.aten.relu_.default` map to `copy` / `in-place` version and both will be converted into `torch.ops.aten.relu.default` ultimately.<br/><br>

The function signature is defined as follow with two arguments:
```python
def annotate_xxx(node: Node, quantization_config: QuantizationConfig) -> None:
```
- __node__: graph node required to be observed
- __quantization_config__: data structure describing quantization configurations for IO activation / weight / bias

### Example of Conv2d Annotation
Conv2d accepts up to three input tensors: `input activation`, `kernel`, `bias`. There are constraints imposed by [Qualcomm AI Engine Direct Manual](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html#conv2d).<br/>
Take 8-bit fixed point as example:
- __weight__: must be symmetrically quantized if per-channel observer is applied
- __bias__: must have `QNN_DATATYPE_SFIXED_POINT_32` and be symmetrically quantized with expected encoding `scales = weight.scales * input.scale`, `offset = 0` if per-channel observer is applied.

Let's look at the simplified per-channel quantization configuration used in `QnnQuantizer`:
```python
def ptq_per_channel_quant_config(
    act_dtype=torch.uint8, weight_dtype=torch.int8
) -> QuantizationConfig:
    ...
    act_quantization_spec = QuantizationSpec(
        dtype=act_dtype,
        quant_min=torch.iinfo(act_dtype).min,
        quant_max=torch.iinfo(act_dtype).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(weight_dtype).min + 1,
        quant_max=torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = _derived_bias_quant_spec

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config
```
Here we choose `torch.uint8` + `MinMaxObserver` for better converage of IO activation and apply rules to `weight` w/`PerChannelMinMaxObserver`, `bias` w/`_derived_bias_quant_spec` (a callable method to calculate encoding in desired way) to meet aforementioned constraints. The well-defined `quantizaton_config` will then be shipped to callback for annotation.<br/>

Now, we can start to fill in the function body:
- Register annotator
    ```python
    @register_annotator(
        [
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv_transpose2d.input,
        ]
    )
    def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
    ```
    There are multiple targets expected to meet our annotation criteria, it's encouraged to do so for code reuse.

- Define map of input quantization spec
    ```python
        if _is_annotated([node]):
            return

        input_qspec_map = {}

        # annotate input activation
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        # annotate kernel
        kernel = node.args[1]
        input_qspec_map[kernel] = quantization_config.weight

        # annotate bias
        if len(node.args) > 2:
            bias = node.args[2]
            input_qspec_map[bias] = quantization_config.bias(node)
    ```
    We first check if current graph node has been annotated. If not, an `input_qspec_map` dictionary required by PyTorch framework will be declared for providing mapping between graph nodes and their configurations.<br/>
    The parameters' order could be found [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp) mentioned in [ATen Operator Definitions](#pytorch). Since bias node is optional, the implementation will invoke `_derived_bias_quant_spec` to calculate the per-channel bias encoding only if it exists.

- Update node's meta with framework compatible data structure
    ```python
        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )
    ```
    After done processing `input_qspec_map`, it's required to have it in node's meta with special tag (`QUANT_ANNOTATION_KEY`) for `convert_pt2e` to properly insert observers.

### Common Annotators
For operators without extra parameters to be observed, there are pre-defined annotation method for convenience:
- Single in single out operators, e.g.:
    ```python
    @register_annotator([torch.ops.aten.relu.default, torch.ops.aten.relu_.default])
    def annotate_relu(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_single_in_single_out(node, quantization_config)
    ```

- Binary in single out operators, e.g.:
    ```python
    @register_annotator([torch.ops.aten.add, torch.ops.aten.add.Tensor])
    def annotate_add(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)
    ```

- Shared encodings between input / output, e.g.:<br/>
    ```python
    # For operators without arithmetical function, IOs are expected to own the same encodings.
    @register_annotator([torch.ops.aten.transpose.int])
    def annotate_transpose(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_single_out(node, quantization_config)
    ```
    This annotator only works for single-in-single-out scenario with node's input that has already been annotated. If not, we still need to invoke `annotate_single_in_single_out` again (this path should be less likely).

## Issues
Please refer to the [issue section](../README.md#issues) for more information.

## Pull Requests
Please refer to the [PR section](../README.md#pull-requests) for more information.
