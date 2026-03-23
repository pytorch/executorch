# Quantization

The Arm VGF delegate can be used to execute quantized models. To quantize a model so that is supported by this delegate, the `VgfQuantizer` should be used.

Currently the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is the main config available to use with the VGF quantizer.

### Supported Quantization Schemes

The quantization schemes supported by the VGF Backend are:
- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
    - Supports both static and dynamic activations
    - Supports per-channel and per-tensor schemes

Weight-only quantization is not currently supported on the VGF backend.

### Partial Quantization

The VGF backend supports partial quantization, where only parts of the model
are quantized while others remain in floating-point. This can be useful for
models where certain layers are not well-suited for quantization or when a
balance between performance and accuracy is desired.

For every node (op) in the graph, the quantizer looks at the *quantization
configuration* set for that specific node. If the configuration is set to
`None`, the node is left in floating-point; if it is provided (not `None`), the
node is quantized according to that configuration.

With the [Quantization API](#quantization-api), users can specify the
quantization configurations for specific layers or submodules of the model. The
`set_global` method is first used to set a default quantization configuration
(could be `None` as explained above) for all nodes in the model. Then,
configurations for specific layers or submodules can override the global
setting using the `set_module_name` or `set_module_type` methods.

### Quantization API

```python
class VgfQuantizer(compile_spec: 'VgfCompileSpec', use_composable_quantizer: 'bool' = False) -> 'None'
```
Quantizer supported by the Arm Vgf backend.

.. warning::
    Setting ``use_composable_quantizer=True`` enables an experimental API
    surface that may change without notice.

Args:
- **compile_spec (VgfCompileSpec)**: Backend compile specification for Vgf
        targets.
- **use_composable_quantizer (bool)**: Whether to use the composable quantizer implementation. See https://github.com/pytorch/executorch/issues/17701" for details.

```python
def VgfQuantizer.add_quantizer(self, quantizer: 'Quantizer') -> 'TOSAQuantizer':
```
Insert a quantizer with highest precedence.

```python
def VgfQuantizer.quantize_with_submodules(self, model: 'GraphModule', calibration_samples: 'list[tuple]', is_qat: 'bool' = False, fold_quantize: 'bool' = True):
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
def VgfQuantizer.set_global(self, quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for submodules not matched by other filters.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration to
        apply to modules that are not captured by name or type filters.
        ``None`` indicates no quantization.

```python
def VgfQuantizer.set_io(self, quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration
        describing activation quantization for model inputs and outputs.
        ``None`` indicates no quantization.

```python
def VgfQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
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
def VgfQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
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
def VgfQuantizer.set_node_finder(self, quantization_config: 'Optional[QuantizationConfig]', node_finder: 'NodeFinder') -> 'TOSAQuantizer':
```
Set quantization_config for nodes matched by a custom NodeFinder.

Args:
- **quantization_config (Optional[QuantizationConfig])**: Configuration
        describing quantization settings for nodes matched by the provided
        NodeFinder. ``None`` indicates no quantization.

```python
def VgfQuantizer.set_node_name(self, node_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization config for a specific node name.

```python
def VgfQuantizer.set_node_target(self, node_target: 'OpOverload', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization config for a specific operator target.

```python
def VgfQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
Transform the graph to prepare it for quantization annotation.

Decomposes all operators where required to get correct quantization parameters.

Args:
- **model (GraphModule)**: Model whose graph will be transformed.

Returns:
- **GraphModule**: Transformed model prepared for annotation.
