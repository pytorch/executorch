# Arm Custom Operators

As a practical extension of `torch.library`, the Arm backends provide a way to
keep selected custom operators inside delegated partitions and lower them to
backend-specific implementations such as shaders or other target-side code.

Arm custom operators are lowered through the Arm TOSA dialect as `tosa.CUSTOM`
nodes. In practice this means a user-visible library op is first captured in the
graph, then rewritten to `tosa.CUSTOM` with a stable `operator_name`,
`domain_name`, and `implementation_attrs` payload that describes the shader or
other backend-specific implementation contract.

The main APIs involved are:
- register the operator with the Arm partitioner using `partitioner.register_custom_partition_op(...)` so it can stay inside the delegated graph
- add a pass that rewrites the `torch.library` op to `tosa.CUSTOM` in the Arm backend.
- provide the target-side implementation, for example a GLSL shader
- provide a function that builds the `tosa.CUSTOM` definition and payload

For a minimal end-to-end example showing the required pieces in Python, see
`examples/arm/custom_operators.py`.


## Resource Layout

### Overview

#### Useful Mental Model
- Tensor/buffer resources: scalar view, channels in shape.
- Image resources: packed texel view, channels in format.
- If you alias tensor and image over the same backing, both views must describe the same logical data consistently.

#### General EValue Tensor Rules
- Treat shader resources as dense, contiguous tensors in the layout declared by the compiled resource contract.
- For current 4D shader-local feature tensors, that means `NHWC`.
- For tensor-like grid and buffer resources, channels remain in the shape and storage is scalar-contiguous in that declared order.
- Do not rely on row padding, channel padding, or partial copies.
- Runtime copies raw bytes only. It does not repack, pad, or reinterpret layout for you.
- If the shader ABI wants a different order, lowering must permute before the `tosa.CUSTOM` node and permute back after it.

### Contract

#### Channels-Last Rules For Current `tosa.CUSTOM` Shader Paths
- To comply with Vulkan texture layout requirements, we focus on channels last.
- For the current Arm/VGF 4D custom-shader ABI, shader-local feature tensors are channels-last.
- That means the internal shader contract is `NHWC`, not graph-visible `NCHW`.
- Lowering is responsible for inserting `NCHW -> NHWC` before the custom node and `NHWC -> NCHW` after it when needed.
- Shader authors should implement against the shader-local layout, not the surrounding graph layout.
- Adjacent shader regions may optimize away redundant permutes, but that is an optimization. The ABI remains explicit.

#### `VK_DESCRIPTOR_TYPE_TENSOR_ARM`
- This is a scalar tensor contract.
- `VkFormat` means scalar element format, not packed channel format.
- For fp32 tensors coming from EValues, use `VK_FORMAT_R32_SFLOAT`.
- Channels stay in the shape.
- Example: a grid tensor is `[N, Hout, Wout, 2]` with `VK_FORMAT_R32_SFLOAT`.
- A 3-channel tensor is fine here as shape `[..., 3]` with scalar format.
- If tensor/image aliasing is used, tensor-like alias members must use scalar formats.

#### `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`
- Same practical data contract as tensor-like resources: scalar, linear, contiguous bytes.
- `VkFormat` is scalar element format.
- Channels stay in the shape.
- If the shader ABI is NHWC, the buffer contents are NHWC scalar linearization.
- Do not use this as an implicit packed-image contract.

#### `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`
- This is a packed image contract.
- Logical shape must be `[H, W, C]` or `[1, H, W, C]`.
- If rank 4, batch must be `1`.
- The image extent is `W x H`.
- Channels are packed into the image `VkFormat`.
- Channel count must exactly match the image format component count.
- Supported current packed image cases are:
  - `C=1` -> `VK_FORMAT_R32_SFLOAT`
  - `C=2` -> `VK_FORMAT_R32G32_SFLOAT`
  - `C=4` -> `VK_FORMAT_R32G32B32A32_SFLOAT`
- `C=3` is not supported for image-backed resources in the current contract.

#### `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`
- Same packing rules as sampled images.
- Writable image-backed output.
- Shape must be `[H, W, C]` or `[1, H, W, C]`, with `N=1` if rank 4.
- `C` must exactly match the image format component count.
- No implicit `3 -> 4` promotion or padding is allowed.
- If you need image-backed output, output channels must be `1`, `2`, or `4`.

#### 3-Channel Limitation
- `C=3` is allowed for tensor/buffer paths because channels remain in the shape.
- `C=3` is rejected for image-backed resources because the current contract only supports exact 1/2/4-component packed image formats.
- If you need image semantics for 3-channel data, you must either:
  - pad to 4 channels explicitly before the custom node, or
  - stay on a tensor/buffer path
