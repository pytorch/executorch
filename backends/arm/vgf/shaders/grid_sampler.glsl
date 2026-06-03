#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input0 {
    float input0[];
};

layout(set = 0, binding = 1) readonly buffer Input1 {
    float input1[];
};

layout(set = 0, binding = 2) writeonly buffer Output0 {
    float output0[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    output0[index] = input0[index];
}
