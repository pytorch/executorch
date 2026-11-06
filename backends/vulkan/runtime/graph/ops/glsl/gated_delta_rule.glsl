/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core
#define PRECISION ${PRECISION}
${layout_declare_tensor(B, "w", "out_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "w", "final_state_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "q_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "k_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "v_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "decay_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "beta_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "initial_state_tex", DTYPE, K_CACHE_STORAGE)}

layout(set = 0, binding = 8) uniform UniformParamsOut {
  ivec4 sizes;     // out sizes in WHCN: x=v_dim, y=seq_len, z=heads, w=batch_size
} out_params;

layout(set = 0, binding = 9) uniform UniformParamsK {
  ivec4 sizes;     // k sizes in WHCN: x=k_dim, y=seq_len, z=heads, w=batch_size
} k_params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define MAX_K 128

void main() {
    const ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    // out_params.sizes is in WHCN order:
    // x = v_dim (dim 3)
    // y = seq_len (dim 2)
    // z = heads (dim 1)
    // w = batch_size (dim 0)
    int v_dim = out_params.sizes.x;
    int seq_len = out_params.sizes.y;
    int heads = out_params.sizes.z;
    int batch_size = out_params.sizes.w;
    int k_dim = k_params.sizes.x;
    
    // Bounds check
    if (pos.x * 4 >= v_dim || pos.z >= batch_size * heads) {
        return;
    }

    int batch = pos.z / heads;
    int head = pos.z % heads;
    int z = pos.z;

    int k_limit = min(k_dim, MAX_K);
    vec4 state_col[MAX_K];

    for (int row = 0; row < k_limit; row++) {
        state_col[row] = texelFetch(initial_state_tex, ivec3(pos.x, row, z), 0);
    }

    int num_k_chunks = k_dim / 4;

    for (int t = 0; t < seq_len; t++) {
        // decay and beta are 3D textures [Batch, Heads, Seq_len]
        // With kWidthPacked: X=seq_len/4, Y=heads, Z=batch
        ivec3 gate_pos = ivec3(t / 4, head, batch);
        float g_t = texelFetch(decay_tex, gate_pos, 0)[t % 4];
        float b_t = texelFetch(beta_tex, gate_pos, 0)[t % 4];

        // Apply decay to state column
        for (int row = 0; row < k_limit; row++) {
            state_col[row] *= g_t;
        }

        // Compute v_pred = sum_k (k_t[k] * state[k])
        vec4 v_pred = vec4(0.0);
        for (int k_idx = 0; k_idx < num_k_chunks; k_idx++) {
            vec4 k_val = texelFetch(k_tex, ivec3(k_idx, t, z), 0);
            v_pred += k_val.x * state_col[k_idx * 4 + 0];
            v_pred += k_val.y * state_col[k_idx * 4 + 1];
            v_pred += k_val.z * state_col[k_idx * 4 + 2];
            v_pred += k_val.w * state_col[k_idx * 4 + 3];
        }

        // v_tex: [Batch, Heads, Seq_len, V_Dim] -> X=pos.x, Y=t, Z=z
        vec4 v_val = texelFetch(v_tex, ivec3(pos.x, t, z), 0);
        vec4 delta = (v_val - v_pred) * b_t;

        // Update state with delta and compute y_t = sum_k (q_t[k] * state_new[k])
        vec4 y_t = vec4(0.0);
        for (int k_idx = 0; k_idx < num_k_chunks; k_idx++) {
            vec4 k_val = texelFetch(k_tex, ivec3(k_idx, t, z), 0);
            vec4 q_val = texelFetch(q_tex, ivec3(k_idx, t, z), 0);

            state_col[k_idx * 4 + 0] += k_val.x * delta;
            state_col[k_idx * 4 + 1] += k_val.y * delta;
            state_col[k_idx * 4 + 2] += k_val.z * delta;
            state_col[k_idx * 4 + 3] += k_val.w * delta;

            y_t += q_val.x * state_col[k_idx * 4 + 0];
            y_t += q_val.y * state_col[k_idx * 4 + 1];
            y_t += q_val.z * state_col[k_idx * 4 + 2];
            y_t += q_val.w * state_col[k_idx * 4 + 3];
        }

        imageStore(out_tex, ivec3(pos.x, t, z), y_t);
    }

    // Write final state
    for (int row = 0; row < k_limit; row++) {
        imageStore(final_state_tex, ivec3(pos.x, row, z), state_col[row]);
    }
}
