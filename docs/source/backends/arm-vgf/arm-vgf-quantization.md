# Quantization

The Arm VGF delegate can be used to execute quantized models. To quantize a model so that is supported by this delegate, the `VgfQuantizer` should be used.

Currently the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is the main config available to use with the VGF quantizer.

### Supported Quantization Schemes

The quantization schemes supported by the VGF Backend are:
- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
    - Supports both static and dynamic activations
    - Supports per-channel and per-tensor schemes

Weight-only quantization is not currently supported on the VGF backend.

### Quantization API

```python
class VgfQuantizer(compile_spec: 'VgfCompileSpec') -> 'None'
```
Quantizer supported by the Arm Vgf backend.

Args:
- **compile_spec**: A VgfCompileSpec instance.

```python
def VgfQuantizer.set_global(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for submodules that are not already annotated by name or type filters.

Args:
- **quantization_config**: The QuantizationConfig to set as global configuration.

```python
def VgfQuantizer.set_io(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config**: The QuantizationConfig to set for input and output nodes.

```python
def VgfQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with name: `module_name`, for example:
quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
patterns in the submodule with this module name with the given `quantization_config`

Args:
- **module_name**: The name of the submodule to set the quantization config for.
- **quantization_config**: The QuantizationConfig to set for the submodule.

```python
def VgfQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with type: `module_type`, for example:
quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
patterns in the submodule with this module type with the given `quantization_config`.

Args:
- **module_type**: The type of the submodule to set the quantization config for.
- **quantization_config**: The QuantizationConfig to set for the submodule.

```python
def VgfQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
An initial pass for transforming the graph to prepare it for annotation.
Currently transforms scalar values to tensor attributes.

Args:
- **model**: The model to transform.
Returns:
    The transformed model.
