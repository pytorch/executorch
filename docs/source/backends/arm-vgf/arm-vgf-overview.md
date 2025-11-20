# Arm VGF Backend

The Arm&reg; VGF backend is the ExecuTorch solution for lowering PyTorch models to VGF compatible hardware.
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

Args:
- **tosa_spec**: TOSA specification that should be targeted.
- **compiler_flags**: Extra compiler flags for converter_backend.

```python
def VgfCompileSpec.dump_debug_info(self, debug_mode: executorch.backends.arm.common.arm_compile_spec.ArmCompileSpec.DebugMode | None):
```
Dump debugging information into the intermediates path.

Args:
- **debug_mode**: The debug mode to use for dumping debug information.

```python
def VgfCompileSpec.dump_intermediate_artifacts_to(self, output_path: str | None):
```
Sets a path for dumping intermediate results during such as tosa and pte.

Args:
- **output_path**: Path to dump intermediate results to.

```python
def VgfCompileSpec.get_intermediate_path(self) -> str | None:
```
Gets the path used for dumping intermediate results such as tosa and pte.

Returns:
    Path where intermediate results are saved.

```python
def VgfCompileSpec.get_output_format() -> str:
```
Returns a constant string that is the output format of the class.



### Partitioner API

See [Partitioner API](arm-vgf-partitioner.md) for more information of the Partitioner API.

## Quantization

The VGF quantizer supports [Post Training Quantization (PT2E)](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html)
and [Quantization-Aware Training (QAT)](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_qat.html).

For more information on quantization, see [Quantization](arm-vgf-quantization.md).

## Runtime Integration

The VGF backend can use the default ExecuTorch runner. The steps required for building and running it are explained in the [VGF Backend Tutorial](tutorials/vgf-getting-started.md).
The example application is recommended to use for testing basic functionality of your lowered models, as well as a starting point for developing runtime integrations for your own targets.

## Reference

**→{doc}`/backends/arm-vgf/arm-vgf-partitioner` — Partitioner options.**

**→{doc}`/backends/arm-vgf/arm-vgf-quantization` — Supported quantization schemes.**

**→{doc}`/backends/arm-vgf/arm-vgf-troubleshooting` — Debug common issues.**

**→{doc}`/backends/arm-vgf/tutorials/arm-vgf-tutorials` — Tutorials.**


```{toctree}
:maxdepth: 2
:hidden:
:caption: Arm VGF Backend

arm-vgf-partitioner
arm-vgf-quantization
arm-vgf-troubleshooting
tutorials/arm-vgf-tutorials
```
