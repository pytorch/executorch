#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "out_buf", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "in_buf", DTYPE, STORAGE)}
${layout_declare_ubo(2, "int", "numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  int tid = int(gl_GlobalInvocationID.x);
  if (tid >= numel) {
    return;
  }
  out_buf[tid] = in_buf[tid];
}
