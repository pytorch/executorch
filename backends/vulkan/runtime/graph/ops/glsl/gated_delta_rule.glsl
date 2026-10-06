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
${layout_declare_tensor(B, "rw", "final_state_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "q_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "k_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "v_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "decay_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "beta_tex", DTYPE, IO_STORAGE)}
${layout_declare_tensor(B, "r", "initial_state_tex", DTYPE, K_CACHE_STORAGE)}

layout(set = 0, binding = 8) uniform UniformParams {
  ivec4 sizes;     // out sizes
} params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
    const ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    // sizes: [Dim, H, Seq, B]
    int v_dim = params.sizes.x;
    int heads = params.sizes.y;
    int seq_len = params.sizes.z;
    int batch_size = params.sizes.w;
    int k_dim = params.sizes.x * 4;
    
    // Bounds check
    if (pos.x * 4 >= v_dim || pos.z >= batch_size * heads) {
        return;
    }

    int batch = pos.z / heads;
    int head = pos.z % heads;

    for (int t = 0; t < seq_len; t++) {
        // v_tex: [Dim, H, Seq, B] -> x=Dim/4, y=H, z=Seq*B
        ivec3 v_pos = ivec3(pos.x, head, batch * seq_len + t);
        vec4 v_val = texelFetch(v_tex, v_pos, 0);

        // decay and beta: [H, Seq, B] -> x=H/4, y=Seq, z=B
        ivec3 gate_pos = ivec3(head / 4, t, batch);
        vec4 g_vec = texelFetch(decay_tex, gate_pos, 0);
        vec4 b_vec = texelFetch(beta_tex, gate_pos, 0);
        float g_t = g_vec[head % 4];
        float b_t = b_vec[head % 4];

        vec4 kv_mem = vec4(0.0);
        vec4 first_k_val = vec4(0.0);
        for (int k_idx = 0; k_idx < k_dim / 4; k_idx++) {
            ivec3 k_pos = ivec3(k_idx, head, batch * seq_len + t);
            vec4 k_val = texelFetch(k_tex, k_pos, 0);
            if (k_idx == 0) first_k_val = k_val;
            
            // state_tex: [Dim, Dim, H, B] -> x=Dim/4, y=Dim, z=B*H
            ivec3 s_pos0 = ivec3(k_idx, pos.x * 4 + 0, pos.z);
            ivec3 s_pos1 = ivec3(k_idx, pos.x * 4 + 1, pos.z);
            ivec3 s_pos2 = ivec3(k_idx, pos.x * 4 + 2, pos.z);
            ivec3 s_pos3 = ivec3(k_idx, pos.x * 4 + 3, pos.z);
            
            vec4 s_row0, s_row1, s_row2, s_row3;
            if (t == 0) {
                s_row0 = texelFetch(initial_state_tex, s_pos0, 0);
                s_row1 = texelFetch(initial_state_tex, s_pos1, 0);
                s_row2 = texelFetch(initial_state_tex, s_pos2, 0);
                s_row3 = texelFetch(initial_state_tex, s_pos3, 0);
            } else {
                s_row0 = imageLoad(final_state_tex, s_pos0);
                s_row1 = imageLoad(final_state_tex, s_pos1);
                s_row2 = imageLoad(final_state_tex, s_pos2);
                s_row3 = imageLoad(final_state_tex, s_pos3);
            }
            
            s_row0 *= g_t;
            s_row1 *= g_t;
            s_row2 *= g_t;
            s_row3 *= g_t;
            
            kv_mem.x += dot(s_row0, k_val);
            kv_mem.y += dot(s_row1, k_val);
            kv_mem.z += dot(s_row2, k_val);
            kv_mem.w += dot(s_row3, k_val);
        }
        vec4 delta = (v_val - kv_mem) * b_t;
        
        // Pass 2: compute y_t and update state
        vec4 y_t = vec4(0.0);
        for (int k_idx = 0; k_idx < k_dim / 4; k_idx++) {
            ivec3 k_pos = ivec3(k_idx, head, batch * seq_len + t);
            ivec3 q_pos = ivec3(k_idx, head, batch * seq_len + t);
            vec4 k_val = texelFetch(k_tex, k_pos, 0);
            vec4 q_val = texelFetch(q_tex, q_pos, 0);
            
            ivec3 s_pos0 = ivec3(k_idx, pos.x * 4 + 0, pos.z);
            ivec3 s_pos1 = ivec3(k_idx, pos.x * 4 + 1, pos.z);
            ivec3 s_pos2 = ivec3(k_idx, pos.x * 4 + 2, pos.z);
            ivec3 s_pos3 = ivec3(k_idx, pos.x * 4 + 3, pos.z);
            
            vec4 s_row0, s_row1, s_row2, s_row3;
            if (t == 0) {
                s_row0 = texelFetch(initial_state_tex, s_pos0, 0);
                s_row1 = texelFetch(initial_state_tex, s_pos1, 0);
                s_row2 = texelFetch(initial_state_tex, s_pos2, 0);
                s_row3 = texelFetch(initial_state_tex, s_pos3, 0);
            } else {
                s_row0 = imageLoad(final_state_tex, s_pos0);
                s_row1 = imageLoad(final_state_tex, s_pos1);
                s_row2 = imageLoad(final_state_tex, s_pos2);
                s_row3 = imageLoad(final_state_tex, s_pos3);
            }
            
            s_row0 *= g_t;
            s_row1 *= g_t;
            s_row2 *= g_t;
            s_row3 *= g_t;
            
            s_row0 += k_val * delta.x;
            s_row1 += k_val * delta.y;
            s_row2 += k_val * delta.z;
            s_row3 += k_val * delta.w;
            
            imageStore(final_state_tex, s_pos0, s_row0);
            imageStore(final_state_tex, s_pos1, s_row1);
            imageStore(final_state_tex, s_pos2, s_row2);
            imageStore(final_state_tex, s_pos3, s_row3);
            
            y_t.x += dot(s_row0, q_val);
            y_t.y += dot(s_row1, q_val);
            y_t.z += dot(s_row2, q_val);
            y_t.w += dot(s_row3, q_val);
        }
        
        imageStore(out_tex, ivec3(pos.x, head, batch * seq_len + t), y_t);
        memoryBarrierImage();
    }
}
