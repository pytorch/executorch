# Writing ATen Operators for ExecuTorch Vulkan Backend

This guide provides a comprehensive walkthrough for contributing new ATen operators to the ExecuTorch Vulkan (ET-VK) backend. The ET-VK backend implements PyTorch operators using GLSL compute shaders executed via the Vulkan graphics API.

## Overview

Adding a new operator to ET-VK involves several key components:
- **C++ Host Code**: Manages tensor metadata, dispatches compute shaders, and handles graph operations
- **GLSL Compute Shaders**: Implement the actual computation logic on the GPU
- **YAML Configuration**: Defines shader compilation parameters and variants
- **Test Cases**: Validate operator correctness and performance

## Step 1: Identifying the Operator Schema

### 1.1 Find the Operator in PyTorch's native_functions.yaml

First, locate your target operator in PyTorch's `native_functions.yaml` file:

```bash
# In PyTorch repository
grep -n "your_operator_name" aten/src/ATen/native/native_functions.yaml
```

Example for `group_norm`:
```yaml
- func: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
  variants: function
  dispatch:
    CPU: native_group_norm
    CUDA: native_group_norm_cuda
```

### 1.2 Understand the Operator Signature

Key information to extract:
- **Input tensors**: Number, types, and optional parameters
- **Output tensors**: Number and types (some operators return tuples)
- **Scalar parameters**: Integer, float, or boolean configuration values
- **Variants**: Different implementations (function, method, etc.)

### 1.3 Check Existing ET-VK Support

Search the ET-VK codebase to see if the operator is already implemented:

```bash
# In executorch/backends/vulkan/
grep -r "your_operator_name" runtime/graph/ops/impl/
```

## Step 2: Adding Test Cases

### 2.1 Add Test Cases in cases.py

Add comprehensive test cases in `test/op_tests/cases.py`:

```python
def sample_inputs_group_norm():
    return [
        OpTestCase(
            op=torch.nn.functional.group_norm,
            sample_input=SampleInput(
                torch.randn(2, 16, 32, 32),  # input tensor
                args=(4, torch.randn(16), torch.randn(16), 1e-5),  # num_groups, weight, bias, eps
            ),
            reference_fn=torch.nn.functional.group_norm,
        ),
        # Add more test cases with different configurations
        OpTestCase(
            op=torch.nn.functional.group_norm,
            sample_input=SampleInput(
                torch.randn(1, 8, 16, 16),
                args=(2, None, None, 1e-5),  # Test without weight/bias
            ),
            reference_fn=torch.nn.functional.group_norm,
        ),
    ]
```

### 2.2 Test Case Best Practices

- **Multiple Configurations**: Test different tensor sizes, parameter combinations
- **Edge Cases**: Test with/without optional parameters, boundary conditions
- **Data Types**: Test supported data types (float32, int32, etc.)
- **Dynamic Shapes**: Include tests with varying batch sizes or spatial dimensions

## Step 3: Writing C++ Host Code

### 3.1 Create the Implementation File

Create `runtime/graph/ops/impl/YourOperator.cpp`:

```cpp
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_your_operator_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  // Extract tensor references using ComputeGraph APIs
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  // Get input tensor sizes using ComputeGraph APIs (avoid vTensorPtr)
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  // Calculate output sizes based on operator logic
  std::vector<int64_t> out_sizes = calculate_output_sizes(in_sizes, /* other params */);

  // Resize output tensor
  graph->virtual_resize(out, out_sizes);
}

void add_your_operator_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out,
    /* other parameters */) {

  // Extract scalar parameters
  const int64_t param_val = graph.extract_scalar<int64_t>(param_ref);
  const float epsilon = graph.extract_scalar<float>(eps_ref);

  // Validate tensor properties
  VK_CHECK_COND(graph.is_standard_channels_packed_texture_tensor(in));
  VK_CHECK_COND(graph.is_standard_channels_packed_texture_tensor(out));

  // Build shader name with appropriate suffixes
  std::string kernel_name("your_operator");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // Prepare uniform data
  const struct {
    int32_t param1;
    float param2;
  } params_uniform = {static_cast<int32_t>(param_val), epsilon};

  // Create dispatch node
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers (UBOs)
      {
          graph.sizes_ubo(in),
          graph.strides_ubo(out),
          graph.numel_ubo(out),
      },
      // Push Constants
      {
          PushConstantDataInfo(&params_uniform, sizeof(params_uniform)),
      },
      // Specialization Constants
      {
          graph.hashed_layout_of(in),
      },
      // Resize Args
      {param_ref},
      // Resizing Logic
      resize_your_operator_node));
}

void your_operator(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Extract arguments according to operator signature
  const ValueRef in = args.at(0);
  const ValueRef param_ref = args.at(1);
  const ValueRef out = args.at(2);

  return add_your_operator_node(graph, in, out, param_ref);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.your_operator.default, your_operator);
}

} // namespace vkcompute
```

### 3.2 C++ Best Practices

#### Memory Management
- **Avoid vTensorPtr**: Use `ComputeGraph` APIs like `graph->sizes_of()`, `graph->extract_scalar()`
- **Use UBOs over Push Constants**: For tensor metadata that changes with dynamic shapes
- **Const Correctness**: Mark variables `const` when possible (except `ComputeGraph*` parameters)

#### Error Handling
- **Validate Inputs**: Use `VK_CHECK_COND()` for tensor properties and constraints
- **Bounds Checking**: Use `.at(i)` instead of `[i]` for vector access

#### Shader Dispatch
- **Uniform Buffers**: Use `graph.sizes_ubo()`, `graph.strides_ubo()` for dynamic metadata
- **Push Constants**: Use for small, static data that doesn't change with tensor shapes
- **Specialization Constants**: Use for compile-time shader variants

## Step 4: Writing GLSL Shader Templates and YAML Configuration

### 4.1 Understanding the Preprocessing System

ET-VK uses `gen_vulkan_spv.py` as a Python preprocessor for GLSL templates:

- **Python Code Blocks**: Marked with `$` character for control flow
- **Python Functions**: Available for generating GLSL code dynamically
- **Template Variables**: Replaced during preprocessing (e.g., `${DTYPE}`, `${PRECISION}`)

### 4.2 Create the GLSL Shader Template

Create `runtime/graph/ops/glsl/your_operator_buffer.glsl`:

```glsl
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

// Declare input/output tensors
${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

// Declare UBOs for tensor metadata
${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}
${layout_declare_ubo(B, "int", "out_numel")}

// Push constants for operator parameters
layout(push_constant) uniform restrict Block {
  int param1;
  float param2;
};

// Specialization constants
${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= out_numel) {
    return;
  }

  // Convert buffer index to tensor coordinates
  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_dim_order);

  // Calculate input coordinates (operator-specific logic)
  const ivec4 in_tidx = calculate_input_coords(out_tidx);
  const int in_bufi = tidx_to_bufi(in_tidx, /* in_strides from push constants */);

  // Perform computation
  const T input_val = t_in[in_bufi];
  const T result = perform_operation(input_val, param1, param2);

  // Write result
  t_out[out_bufi] = result;
}
```

### 4.3 Create the YAML Configuration

Create `runtime/graph/ops/glsl/your_operator_buffer.yaml`:

```yaml
# Shader compilation configuration
glsl_file: your_operator_buffer.glsl
parameter_names_with_default_values:
  PRECISION: "highp"
  DTYPE: "float"

# Generate variants for different data types
generate_variant_forall:
  - parameter: DTYPE
    values: [float, half, int]

# Optional: Custom shader name patterns
shader_name: your_operator_buffer_${DTYPE}
```

### 4.4 GLSL Best Practices

#### Tensor Access Patterns
- **Use UBOs for Metadata**: Tensor sizes, strides, and element counts
- **Efficient Indexing**: Use `indexing_utils.h` functions for coordinate conversion
- **Bounds Checking**: Always check `gl_GlobalInvocationID.x < numel`

#### Memory Layout
- **WHCN Order**: All tensor metadata is in Width-Height-Channels-Batch order
- **4D Assumption**: All tensors treated as 4D, missing dimensions have size 1
- **Texture vs Buffer**: Implement separate shaders for texture and buffer storage

#### Performance Optimization
- **Vectorization**: Use `VEC4_T` for texture-backed tensors when possible
- **Memory Coalescing**: Access memory in patterns that align with GPU architecture
- **Shared Memory**: Use for reduction operations or data sharing between threads

### 4.5 Texture-Backed Tensor Implementation

For texture-backed tensors, create `your_operator_texture.glsl`:

```glsl
#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_type(DTYPE)}

${define_active_storage_type("texture3d")}
${define_required_extensions(DTYPE)}

#include "indexing_utils.h"

// Declare texture tensors
${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

// UBOs for texture limits and sizes
${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(push_constant) uniform restrict Block {
  int param1;
  float param2;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  // Load texel (4 elements at once)
  const VEC4_T input_texel = load_texel(t_in, pos);

  // Process all 4 components
  VEC4_T result_texel;
  for (int i = 0; i < 4; i++) {
    result_texel[i] = perform_operation(input_texel[i], param1, param2);
  }

  // Write result
  write_texel(t_out, pos, result_texel);
}
```

## Step 5: Integration and Testing

### 5.1 Build System Integration

The build system automatically discovers and compiles GLSL shaders. Ensure your files follow the naming convention:
- `your_operator_buffer.glsl` + `your_operator_buffer.yaml`
- `your_operator_texture.glsl` + `your_operator_texture.yaml`

### 5.2 Register the Operator

Add your operator to the appropriate registry files and ensure it's included in the build targets.

### 5.3 Testing

Run comprehensive tests:

```bash
# Run operator-specific tests
python -m pytest test/op_tests/ -k your_operator

# Run integration tests
python test/test_vulkan_delegate.py::TestVulkanBackend::test_your_operator
```

## Advanced Topics

### Dynamic Shapes and Resize Functions

For operators that need to handle dynamic input shapes:

```cpp
void resize_your_operator_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  // Extract parameters that affect output shape
  const int64_t param = graph->extract_scalar<int64_t>(resize_args.at(0));

  // Calculate new output sizes
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  std::vector<int64_t> out_sizes = in_sizes;  // Copy and modify as needed
  out_sizes[2] = in_sizes[2] / param;  // Example: reduce channel dimension

  // Resize output tensor
  graph->virtual_resize(out, out_sizes);
}
```

### Multi-Pass Algorithms

Some operators require multiple shader passes:

```cpp
// First pass: compute statistics
graph.execute_nodes().emplace_back(new DynamicDispatchNode(
    graph,
    VK_KERNEL_FROM_STR("your_operator_reduce"),
    // ... configuration for reduction pass
));

// Second pass: apply normalization
graph.execute_nodes().emplace_back(new DynamicDispatchNode(
    graph,
    VK_KERNEL_FROM_STR("your_operator_apply"),
    // ... configuration for application pass
));
```

### Quantized Operators

For quantized variants, implement additional logic for scale/zero-point handling:

```glsl
// In GLSL shader
${layout_declare_ubo(B, "float", "input_scale")}
${layout_declare_ubo(B, "int", "input_zero_point")}
${layout_declare_ubo(B, "float", "output_scale")}
${layout_declare_ubo(B, "int", "output_zero_point")}

// Dequantize input
float dequantized = (input_val - input_zero_point) * input_scale;

// Perform computation
float result = perform_operation(dequantized);

// Quantize output
int quantized_result = int(result / output_scale + output_zero_point);
```

## Common Pitfalls and Debugging

### 1. Shader Compilation Errors
- Check GLSL syntax and version compatibility
- Verify all template variables are properly defined
- Ensure proper include paths for utility headers

### 2. Runtime Errors
- Validate tensor dimensions and memory layouts
- Check bounds in both C++ and GLSL code
- Verify UBO and push constant data alignment

### 3. Performance Issues
- Profile memory access patterns
- Optimize workgroup sizes for target hardware
- Consider using shared memory for reduction operations

### 4. Numerical Accuracy
- Be aware of precision differences between CPU and GPU
- Test with appropriate tolerances in validation
- Consider using higher precision for intermediate calculations

## Conclusion

Contributing operators to ET-VK requires understanding both the high-level graph operations and low-level GPU programming. By following this guide and the established patterns in the codebase, you can successfully implement efficient and robust operators for the ExecuTorch Vulkan backend.

For additional examples, refer to existing operator implementations in `runtime/graph/ops/impl/` and their corresponding GLSL shaders in `runtime/graph/ops/glsl/`.
