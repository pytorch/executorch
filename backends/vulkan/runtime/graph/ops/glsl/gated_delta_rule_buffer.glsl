/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core
#define PRECISION ${PRECISION}

${define_required_extensions("buffer", DTYPE)}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "out_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "final_state_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "q_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "k_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "v_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "decay_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "beta_buf", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "initial_state_buf", DTYPE, "buffer")}

layout(set = 0, binding = 8) uniform UniformParamsOut {
  ivec4 sizes;     // out sizes in WHCN: x=v_dim, y=seq_len, z=heads, w=batch_size
} out_params;

layout(set = 0, binding = 9) uniform UniformParamsK {
  ivec4 sizes;     // k sizes in WHCN: x=k_dim, y=seq_len, z=heads, w=batch_size
} k_params;

layout(set = 0, binding = 10) uniform UniformParamsState {
  ivec4 sizes;     // initial_state sizes in WHCN: x=v_dim, y=k_dim, z=heads, w=batch_size
} state_params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define MAX_K 128

void main() {
    const ivec3 pos = ivec3(gl_GlobalInvocationID);

    // out_params.sizes is in WHCN order:
    // x = v_dim
    // y = seq_len
    // z = heads
    // w = batch_size
    int v_dim = out_params.sizes.x;
    int seq_len = out_params.sizes.y;
    int heads = out_params.sizes.z;
    int batch_size = out_params.sizes.w;
    int k_dim = k_params.sizes.x;

    int v_base = pos.x * 4;

    // Bounds check
    if (v_base >= v_dim || pos.z >= batch_size * heads) {
        return;
    }

    int batch = pos.z / heads;
    int head = pos.z % heads;
    int bh = batch * heads + head;

    // In Kev prefix evaluation, initial recurrent state comes from the single prefix (batch 0)
    int init_batch = 0;
    int init_bh = init_batch * heads + head;

    int k_limit = min(k_dim, MAX_K);
    vec4 state_col[MAX_K];

    // Load initial recurrent state
    int state_bh_stride = k_dim * v_dim;
    for (int row = 0; row < k_limit; row++) {
        int idx = init_bh * state_bh_stride + row * v_dim + v_base;
        state_col[row] = vec4(
            float(initial_state_buf[idx + 0]),
            float(initial_state_buf[idx + 1]),
            float(initial_state_buf[idx + 2]),
            float(initial_state_buf[idx + 3])
        );
    }

    int gate_bh_stride = seq_len;
    int kv_bh_stride = seq_len * k_dim;
    int out_bh_stride = seq_len * v_dim;

    for (int t = 0; t < seq_len; t++) {
        int gate_idx = bh * gate_bh_stride + t;
        float g_t = float(decay_buf[gate_idx]);
        float b_t = float(beta_buf[gate_idx]);

        // 1. Decay state column
        for (int row = 0; row < k_limit; row++) {
            state_col[row] *= g_t;
        }

        // 2. Compute v_pred = sum_k (k_t[k] * state[k])
        int k_base = bh * kv_bh_stride + t * k_dim;
        vec4 v_pred = vec4(0.0);
        for (int k = 0; k < k_limit; k++) {
            float k_val = float(k_buf[k_base + k]);
            v_pred += k_val * state_col[k];
        }

        // 3. Compute delta = (v_val - v_pred) * b_t
        int v_idx = bh * out_bh_stride + t * v_dim + v_base;
        vec4 v_val = vec4(
            float(v_buf[v_idx + 0]),
            float(v_buf[v_idx + 1]),
            float(v_buf[v_idx + 2]),
            float(v_buf[v_idx + 3])
        );
        vec4 delta = (v_val - v_pred) * b_t;

        // 4. Update state with delta and compute y_t = sum_k (q_t[k] * state_new[k])
        vec4 y_t = vec4(0.0);
        for (int k = 0; k < k_limit; k++) {
            float k_val = float(k_buf[k_base + k]);
            float q_val = float(q_buf[k_base + k]);

            state_col[k] += k_val * delta;
            y_t += q_val * state_col[k];
        }

        // 5. Store output
        int out_idx = bh * out_bh_stride + t * v_dim + v_base;
        out_buf[out_idx + 0] = T(y_t.x);
        out_buf[out_idx + 1] = T(y_t.y);
        out_buf[out_idx + 2] = T(y_t.z);
        out_buf[out_idx + 3] = T(y_t.w);
    }

    // Write final recurrent state
    for (int row = 0; row < k_limit; row++) {
        int idx = bh * state_bh_stride + row * v_dim + v_base;
        final_state_buf[idx + 0] = T(state_col[row].x);
        final_state_buf[idx + 1] = T(state_col[row].y);
        final_state_buf[idx + 2] = T(state_col[row].z);
        final_state_buf[idx + 3] = T(state_col[row].w);
    }
}
