#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT    ${FORMAT}

layout(std430) buffer;

// TODO: 1. Define bindings for Q, K, V, Decay, Beta, State, and Output buffers.
// e.g., layout(set = 0, binding = 0) buffer Q { float q_data[]; };

// TODO: 2. Define uniforms (batch size, dimensions, etc.)
layout(set = 0, binding = 7) uniform UniformParams {
    int batch;
    // ...
} params;

// TODO: 3. Define local workgroup size
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
    // Get the global ID of this specific thread
    const ivec3 pos = ivec3(gl_GlobalInvocationID);

    // TODO: 4. Implement the gated delta rule math here
    // - Read inputs based on pos
    // - Update state
    // - Compute output
    // - Write to output buffer
}
