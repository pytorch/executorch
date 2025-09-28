# Arm&reg; VGF Backend

The Arm VGF backend is the ExecuTorch solution for lowering PyTorch models to VGF compatible hardware.
It leverages the TOSA operator set and the [ML SDK for Vulkan&reg;](https://github.com/arm/ai-ml-sdk-for-vulkan?tab=readme-ov-file) to produce a .PTE file.
The VGF backend also supports execution from a .PTE file and provides functionality to extract the corresponding VGF file for integration into various applications.

## Features

- Wide operator support for delegating large parts of models to the VGF target.
- A quantizer that optimizes quantization for the VGF target.

## Target Requirements
The target system must include ML SDK for Vulkan and a Vulkan driver with Vulkan API >= 1.3.

## Development Requirements

```{tip}
All requirements can be downloaded using `examples/arm/setup.sh --enable-mlsdk-deps --disable-ethos-u-deps` and added to the path using
`source examples/arm/ethos-u-scratch/setup_path.sh`
```

For the AOT flow, compilation of a model to `.pte` format using the VGF backend, the requirements are:
- [TOSA Serialization Library](https://www.mlplatform.org/tosa/software.html) for serializing the Exir IR graph into TOSA IR.
- [ML SDK Model Converter](https://github.com/arm/ai-ml-sdk-model-converter) for converting TOSA flatbuffers to VGF files.

And for building and running your application using the generic executor_runner:
- [Vulkan API](https://www.vulkan.org) should be set up locally for GPU execution support.
- [ML Emulation Layer for Vulkan](https://github.com/arm/ai-ml-emulation-layer-for-vulkan) for testing on Vulkan API.

## Using the Arm VGF Backend
The [VGF Minimal Example](https://github.com/pytorch/executorch/blob/main/examples/arm/vgf_minimal_example.ipynb) demonstrates how to lower a module using the VGF backend.

The main configuration point for the lowering is the `VgfCompileSpec` consumed by the partitioner and quantizer.
The full user-facing API is documented below.

```python
class VgfCompileSpec(tosa_spec: executorch.backends.arm.tosa.specification.TosaSpecification | str | None = None, compiler_flags: list[str] | None = None)
```
Compile spec for VGF compatible targets.

Attributes:
- **tosa_spec**: A TosaSpecification, or a string specifying a TosaSpecification.
- **compiler_flags**: Extra compiler flags for converter_backend.

```python
def VgfCompileSpec.dump_debug_info(self, debug_mode: executorch.backends.arm.common.arm_compile_spec.ArmCompileSpec.DebugMode | None):
```
Dump debugging information into the intermediates path.

```python
def VgfCompileSpec.dump_intermediate_artifacts_to(self, output_path: str | None):
```
Sets a path for dumping intermediate results during lowering such as tosa and pte.

```python
def VgfCompileSpec.get_intermediate_path(self) -> str | None:
```
Returns the path for dumping intermediate results during lowering such as tosa and pte.

```python
def VgfCompileSpec.get_output_format() -> str:
```
Returns a constant string that is the output format of the class.



### Partitioner API
```python
class VgfPartitioner(compile_spec: executorch.backends.arm.vgf.compile_spec.VgfCompileSpec, additional_checks: Optional[Sequence[torch.fx.passes.operator_support.OperatorSupportBase]] = None) -> None
```
Partitions subgraphs supported by the Arm Vgf backend.

Attributes:
- **compile_spec**:List of CompileSpec objects for Vgf backend.
- **additional_checks**: Optional sequence of additional operator support checks.

```python
def VgfPartitioner.ops_to_not_decompose(self, ep: torch.export.exported_program.ExportedProgram) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.node.Node], bool]]]:
```
Returns a list of operator names that should not be decomposed. When these ops are
registered and the `to_backend` is invoked through to_edge_transform_and_lower it will be
guaranteed that the program that the backend receives will not have any of these ops
decomposed.

Returns:
- **List[torch._ops.OpOverload]**: a list of operator names that should not be decomposed.
- **Optional[Callable[[torch.fx.Node], bool]]]**: an optional callable, acting as a filter, that users can provide
    which will be called for each node in the graph that users can use as a filter for certain
    nodes that should be continued to be decomposed even though the op they correspond to is
    in the list returned by ops_to_not_decompose.

```python
def VgfPartitioner.partition(self, exported_program: torch.export.exported_program.ExportedProgram) -> executorch.exir.backend.partitioner.PartitionResult:
```
Returns the input exported program with newly created sub-Modules encapsulating
specific portions of the input "tagged" for delegation.

The specific implementation is free to decide how existing computation in the
input exported program should be delegated to one or even more than one specific
backends.

The contract is stringent in that:
* Each node that is intended to be delegated must be tagged
* No change in the original input exported program (ExportedProgram) representation can take
place other than adding sub-Modules for encapsulating existing portions of the
input exported program and the associated metadata for tagging.

Args:
- **exported_program**: An ExportedProgram in Edge dialect to be partitioned for backend delegation.

Returns:
- **PartitionResult**: includes the tagged graph and the delegation spec to indicate what backend_id and compile_spec is used for each node and the tag created by the backend developers.



### Quantizer
The VGF quantizer supports [Post Training Quantization (PT2E)](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html)
and [Quantization-Aware Training (QAT)](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_qat.html) quantization.

Currently the symmetric `int8` config defined by `executorch.backends.arm.quantizer.arm_quantizer.get_symmetric_quantization_config` is
the main config available to use with the VGF quantizer.

```python
class VgfQuantizer(compile_spec: 'VgfCompileSpec') -> 'None'
```
Quantizer supported by the Arm Vgf backend.

Attributes:
- **compile_spec**: VgfCompileSpec, specifies the compilation configuration.

```python
def VgfQuantizer.set_global(self, quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for submodules that are not already annotated by name or type filters.

Args:
- **quantization_config**: Specifies the quantization scheme for the weights and activations

```python
def VgfQuantizer.set_io(self, quantization_config):
```
Set quantization_config for input and output nodes.

Args:
- **quantization_config**: Specifies the quantization scheme for the weights and activations

```python
def VgfQuantizer.set_module_name(self, module_name: 'str', quantization_config: 'Optional[QuantizationConfig]') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with name: `module_name`, for example:
quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
patterns in the submodule with this module name with the given `quantization_config`

Args:
- **module_name**: Name of the module to which the quantization_config is set.
- **quantization_config**: Specifies the quantization scheme for the weights and activations.

Returns:
- **TOSAQuantizer**: The quantizer instance with the updated module name configuration

```python
def VgfQuantizer.set_module_type(self, module_type: 'Callable', quantization_config: 'QuantizationConfig') -> 'TOSAQuantizer':
```
Set quantization_config for a submodule with type: `module_type`, for example:
quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
patterns in the submodule with this module type with the given `quantization_config`

Args:
- **module_type**: Type of module to which the quantization_config is set.
- **quantization_config**: Specifies the quantization scheme for the weights and activations.

Returns:
- **TOSAQuantizer**: The quantizer instance with the updated module type configuration

```python
def VgfQuantizer.transform_for_annotation(self, model: 'GraphModule') -> 'GraphModule':
```
An initial pass for transforming the graph to prepare it for annotation.
Currently transforms scalar values to tensor attributes.

Args:
- **model**: Module that is transformed.

Returns:
    The transformed model.


### Supported Quantization Schemes
The quantization schemes supported by the VGF Backend are:
- 8-bit symmetric weights with 8-bit asymmetric activations (via the PT2E quantization flow).
    - Supports both static and dynamic activations
    - Supports per-channel and per-tensor schemes

Weight-only quantization is not currently supported on VGF

## Runtime Integration

The VGF backend can use the default ExecuTorch runner. The steps required for building and running it are explained in the previously mentioned [VGF Backend Tutorial](https://docs.pytorch.org/executorch/stable/tutorial-arm-ethos-u.html).
The example application is recommended to use for testing basic functionality of your lowered models, as well as a starting point for developing runtime integrations for your own targets.

### VGF Adapter for Model Explorer

The [VGF Adapter for Model Explorer](https://github.com/arm/vgf-adapter-model-explorer) enables visualization of
VGF files and can be useful for debugging.
