# Quantization

The Arm Ethos-U delegate only supports the execution of quantized models. To quantize a model so that is supported by this delegate, the `EthosUQuantizer` should be used.

Currently, the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is the main config available to use with the Ethos-U quantizer.

### Supported Quantization Schemes

The Arm Ethos-U delegate supports the following quantization schemes:

- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
- Limited support for 16-bit quantization with 16-bit activations and 8-bit weights (a.k.a 16x8 quantization).
- Limited support for 8-bit quantization with 8-bit activations and 4-bit weights (a.k.a. 8x4 quantization). 
- Partial quantization is supported by the quantizer, but non-quantized operators won't be delegated to the Ethos-U backend.

### Selecting Attention Precision

Use the composable quantizer's node-finder API to configure native SDPA.
Later registrations take precedence, so register broad regions before their
more specific overrides. The quantizer decomposes supported SDPA calls before
applying the matmul and score finders; role finders reject ambiguous
decompositions rather than silently selecting a different precision.
An ambiguous decomposition encountered by a role-specific `SDPAMatmulFinder`
or by `SDPAScoreFinder` raises `ValueError` and stops quantization of the whole
model, not just that SDPA call. The score finder also classifies QK internally.

For native SDPA, a quantized `SDPAFinder()` or global configuration is required
to enable decomposition. `SDPAMatmulFinder` and `SDPAScoreFinder` only select
nodes after decomposition; registering either alone does not enable native
SDPA quantization. They intentionally return no matches on native calls during
the initial annotation pass. Register the broad policy first, as below, then
apply the more specific overrides.
Each recipe below is independent and constructs a new quantizer.

For A16 attention:

```python
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.cortex_m.quantizer.node_finders import SDPAFinder

quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u85-256"))
quantizer.set_node_finder(
    get_symmetric_a16w8_quantization_config(), SDPAFinder()
)
```

For A8 QK/AV matmuls with INT32 QK accumulation and A16 score processing:

```python
from dataclasses import replace
from functools import partial

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.quantization_config import (
    get_int8_matmul_accumulator_qspec,
)
from executorch.backends.cortex_m.quantizer.node_finders import (
    SDPAFinder,
    SDPAMatmulFinder,
    SDPAScoreFinder,
)

quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u85-256"))
a8 = get_symmetric_quantization_config()
quantizer.set_node_finder(a8, SDPAFinder())
quantizer.set_node_finder(
    get_symmetric_a16w8_quantization_config(), SDPAScoreFinder()
)
qk_input = replace(a8.input_activation, qscheme=torch.per_tensor_symmetric)
qk_config = replace(
    a8,
    input_activation=qk_input,
    output_activation=partial(
        get_int8_matmul_accumulator_qspec, input_qspec=qk_input
    ),
)
quantizer.set_node_finder(qk_config, SDPAMatmulFinder("qk"))
```

An output-qspec factory receives the selected FX node and explicitly overrides
the config's usual output rules, including fixed and shared qparams. Scope it
only to operators it supports. The accumulator factory requires static,
symmetric INT8 inputs and a static reduction dimension; it checks the actual
input observers and conservatively rejects possible INT32 overflow. Its scale
comes from the product of the actual input scales.

Calling `get_output_act_qspec()` without a node remains supported for concrete
specs, but is an error for a factory. Factory configs are not a drop-in global
configuration for clients that inspect output specs without a node, such as
Cortex-M pattern checkers.

Attention precision and persistent KV storage are independent. Configure cache
replacement separately. `StaticQuantizedKVCache` defaults its scale to
`1 / torch.iinfo(quantized_cache_dtype).max` when no scale is supplied; explicit
scales and calibrated scales are preserved. INT16 persistent variables remain
an experimental Vela-specific path, not a portable TOSA capability.

The repository currently pins Vela 5.2.0. Tests using that pin check INT16 TOSA
variable lowering with a stubbed compiler, but skip the real INT16 Vela
persistent-region compilation case. That case requires Vela 5.3.0 or a
compatible development build exposing `--separate-persistent-region`; the skip
reason identifies this requirement. TOSA dtype checks are not evidence of
successful Vela compilation or device execution.

### Quantization API

```python
class EthosUQuantizer(compile_spec: 'EthosUCompileSpec', use_composable_quantizer: 'bool' = True) -> 'None'
```
Quantizer supported by the Arm Ethos-U backend.

.. warning::
    The composable quantizer is now the default implementation. Setting
    ``use_composable_quantizer=False`` is deprecated and will be removed in
    two minor releases.

Args:
- **compile_spec (EthosUCompileSpec)**: Backend compile specification for
        Ethos-U targets.
- **use_composable_quantizer (bool)**: Whether to use the composable
        quantizer implementation. Setting this to ``False`` is deprecated
        and will be removed in two minor releases. See
        [issue #17701](https://github.com/pytorch/executorch/issues/17701)
        for details.

```python
def EthosUQuantizer.add_quantizer(self, quantizer: 'Quantizer') -> 'TOSAQuantizer':
```
Insert a quantizer with highest precedence.

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
- **node_finder (NodeFinder)**: Predicate used to select nodes.

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
