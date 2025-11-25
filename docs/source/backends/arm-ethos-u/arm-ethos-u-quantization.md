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
- **compile_spec (EthosUCompileSpec)**: Backend compile specification for
        Ethos-U targets.

```python
def EthosUQuantizer.quantize_with_submodules(self, model: 'GraphModule', calibration_samples: 'list[tuple]', is_qat: 'bool' = False):
```
Quantizes a GraphModule in a way such that conditional submodules are handled properly.

Args:
- **model (GraphModule)**: The model to quantize.
- **calibration_samples (list[tuple])**: A list of inputs to used to
        calibrate the model during quantization. To properly calibrate a
        model with submodules, at least one sample per code path is
        needed.
- **is_qat (bool)**: Whether to do quantization aware training or not.

Returns:
- **GraphModule**: The quantized model.

```python
def EthosUQuantizer.set_global(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for submodules not matched by other filters.

Args:
- **quantization_config (QuantizationConfig)**: Configuration to apply to
        modules that are not captured by name or type filters.

```python
def EthosUQuantizer.set_io(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config (QuantizationConfig)**: Configuration describing
        activation quantization for model inputs and outputs.

```python
def EthosUQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for submodules with a given module name.

For example, calling set_module_name("blocks.sub") quantizes supported
patterns for that submodule with the provided quantization_config.

Args:
- **module_name (str)**: Fully qualified module name to configure.
- **quantization_config (QuantizationConfig)**: Configuration to apply to
        the named submodule.

```python
def EthosUQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for submodules with a given module type.

For example, calling set_module_type(Sub) quantizes supported patterns
in each Sub instance with the provided quantization_config.

Args:
- **module_type (Callable)**: Type whose submodules should use the
        provided quantization configuration.
- **quantization_config (QuantizationConfig)**: Configuration to apply to
        submodules of the given type.

```python
def EthosUQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
Transform the graph to prepare it for quantization annotation.

Currently transforms scalar values to tensor attributes.

Args:
- **model (GraphModule)**: Model whose graph will be transformed.

Returns:
- **GraphModule**: Transformed model prepared for annotation.
