# Quantization

The Arm Ethos-U delegate only supports the execution of quantized models. To quantize a model so that is supported by this delegate, the `EthosUQuantizer` should be used.

Currently, the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is the main config available to use with the Ethos-U quantizer.

### Supported Quantization Schemes

The Arm Ethos-U delegate supports the following quantization schemes:

- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
- Limited support for 16-bit quantization with 16-bit activations and 8-bit weights (a.k.a 16x8 quantization).
- Limited support for 8-bit quantization with 8-bit activations and 4-bit weights (a.k.a. 8x4 quantization). 
- Partial quantization is supported by the quantizer, but non-quantized operators won't be delegated to the Ethos-U backend.

### Quantization API

```python
class EthosUQuantizer(compile_spec: 'EthosUCompileSpec', use_composable_quantizer: 'bool' = False) -> 'None'
```
Quantizer supported by the Arm Ethos-U backend.

.. warning::
    Setting ``use_composable_quantizer=True`` enables an experimental API
    surface that may change without notice.

Args:
- **compile_spec (EthosUCompileSpec)**: Backend compile specification for
        Ethos-U targets.
- **use_composable_quantizer (bool)**: Whether to use the composable quantizer implementation. See https://github.com/pytorch/executorch/issues/17701" for details.

```python
def EthosUQuantizer.add_quantizer(self, quantizer: 'Quantizer') -> 'TOSAQuantizer':
```
Insert a quantizer with highest precedence.

```python
def EthosUQuantizer.quantize_with_submodules(self, model: 'GraphModule', calibration_samples: 'list[tuple]', is_qat: 'bool' = False, fold_quantize: 'bool' = True):
```
Quantizes a GraphModule in a way such that conditional submodules are
handled properly.

Note: torchao's prepare_pt2e and convert_pt2e natively handle
while_loop body_fn submodules, so we only manually process cond
branches and while_loop cond_fn here.

Args:
- **model (GraphModule)**: The model to quantize.
- **calibration_samples (list[tuple])**: A list of inputs to used to
        calibrate the model during quantization. To properly calibrate a
        model with submodules, at least one sample per code path is
        needed.
- **is_qat (bool)**: Whether to do quantization aware training or not.
- **fold_quantize (bool)**: Enables or disables constant folding when quantization
        is completed.

Returns:
- **GraphModule**: The quantized model.

```python
def EthosUQuantizer.set_global(self, quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for submodules not matched by other filters.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration to
        apply to modules that are not captured by name or type filters.
        ``None`` indicates no quantization.

```python
def EthosUQuantizer.set_io(self, quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration
        describing activation quantization for model inputs and outputs.
        ``None`` indicates no quantization.

```python
def EthosUQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for submodules with a given module name.

For example, calling set_module_name("blocks.sub") quantizes supported
patterns for that submodule with the provided quantization_config.

Args:
- **module_name (str)**: Fully qualified module name to configure.
- **quantization_config (Optional[QuantizationConfig])**: Configuration
        applied to the named submodule. ``None`` indicates no
        quantization.

```python
def EthosUQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for submodules with a given module type.

For example, calling set_module_type(Softmax) quantizes supported
patterns in each Softmax instance with the provided quantization_config.

Args:
- **module_type (Callable)**: Type whose submodules should use the
        provided quantization configuration.
- **quantization_config (Optional[QuantizationConfig])**: Configuration to
        apply to submodules of the given type. ``None`` indicates no
        quantization.

```python
def EthosUQuantizer.set_node_finder(self, quantization_config: 'Optional[QuantizationConfig]', node_finder: 'NodeFinder') -> 'TOSAQuantizer':
```
Set quantization_config for nodes matched by a custom NodeFinder.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration
        describing quantization settings for nodes matched by the provided
        NodeFinder. ``None`` indicates no quantization.

```python
def EthosUQuantizer.set_node_name(self, node_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization config for a specific node name.

```python
def EthosUQuantizer.set_node_target(self, node_target: 'OpOverload', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization config for a specific operator target.

```python
def EthosUQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
Transform the graph to prepare it for quantization annotation.

Decomposes all operators where required to get correct quantization parameters.

Args:
- **model (GraphModule)**: Model whose graph will be transformed.

Returns:
- **GraphModule**: Transformed model prepared for annotation.
