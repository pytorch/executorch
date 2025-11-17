# Quantization

The Arm Ethos-U delegate only supports the execution of quantized models. To quantize a model so that is supported by this delegate, the `EthosUQuantizer` should be used.

Currently, the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is the main config available to use with the Ethos-U quantizer.

### Supported Quantization Schemes

The Arm Ethos-U delegate supports the following quantization schemes:

- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
- Limited support for 16-bit quantization with 16-bit activations and 8-bit weights (a.k.a 16x8 quantization). This is under development.

### Quantization API

```python
class EthosUQuantizer(compile_spec: 'EthosUCompileSpec') -> 'None'
```
Quantizer supported by the Arm Ethos-U backend.

Args:
- **compile_spec**: A EthosUCompileSpec instance.

```python
def EthosUQuantizer.set_global(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for submodules that are not already annotated by name or type filters.

Args:
- **quantization_config**: The QuantizationConfig to set as global configuration.

```python
def EthosUQuantizer.set_io(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config**: The QuantizationConfig to set for input and output nodes.

```python
def EthosUQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with name: `module_name`, for example:
quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
patterns in the submodule with this module name with the given `quantization_config`

Args:
- **module_name**: The name of the submodule to set the quantization config for.
- **quantization_config**: The QuantizationConfig to set for the submodule.

```python
def EthosUQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with type: `module_type`, for example:
quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
patterns in the submodule with this module type with the given `quantization_config`.

Args:
- **module_type**: The type of the submodule to set the quantization config for.
- **quantization_config**: The QuantizationConfig to set for the submodule.

```python
def EthosUQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
An initial pass for transforming the graph to prepare it for annotation.
Currently transforms scalar values to tensor attributes.

Args:
- **model**: The model to transform.
Returns:
    The transformed model.
