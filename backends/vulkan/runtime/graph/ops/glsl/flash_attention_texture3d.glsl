/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

// Flash Attention inputs: Query, Key, Value tensors using texture storage
${layout_declare_tensor(B, "rw", "t_O", DTYPE, "texture3d")}
${layout_declare_tensor(B, "rw", "t_l", "float", "texture3d")}
${layout_declare_tensor(B, "rw", "t_m", "float", "texture3d")}
${layout_declare_tensor(B, "r", "t_Q", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_K", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_V", DTYPE, "texture3d")}

${layout_declare_ubo(B, "ivec4", "Q_sizes")}  // [B, H, N, D]
${layout_declare_ubo(B, "ivec4", "K_sizes")}
${layout_declare_ubo(B, "ivec4", "V_sizes")}
${layout_declare_ubo(B, "ivec4", "O_sizes")}

${layout_declare_ubo(B, "ivec3", "l_sizes")}  // [B, H, N]
${layout_declare_ubo(B, "ivec3", "m_sizes")}  // [B, H, N]

${layout_declare_ubo(B, "float", "scale")}
${layout_declare_ubo(B, "int", "block_size_r")} // Br (num rows in Q block)
${layout_declare_ubo(B, "int", "block_size_c")} // Bc (num cols in K/V block)
${layout_declare_ubo(B, "int", "input_pos")}    // Starting position for causal masking
${layout_declare_ubo(B, "int", "num_heads")}    // Number of query heads
${layout_declare_ubo(B, "int", "num_kv_heads")} // Number of key/value heads

// Axis mapping setup for proper texture indexing
${layout_declare_spec_const(C, "int", "Q_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 Q_axis_map = unhash_axis_map(Q_layout);
const lowp int Q_packed_dim = unhash_packed_dim(Q_layout);

${layout_declare_spec_const(C, "int", "K_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 K_axis_map = unhash_axis_map(K_layout);
const lowp int K_packed_dim = unhash_packed_dim(K_layout);

${layout_declare_spec_const(C, "int", "V_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 V_axis_map = unhash_axis_map(V_layout);
const lowp int V_packed_dim = unhash_packed_dim(V_layout);

${layout_declare_spec_const(C, "int", "O_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 O_axis_map = unhash_axis_map(O_layout);
const lowp int O_packed_dim = unhash_packed_dim(O_layout);

${layout_declare_spec_const(C, "int", "l_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 l_axis_map = unhash_axis_map(l_layout);
const lowp int l_packed_dim = unhash_packed_dim(l_layout);

${layout_declare_spec_const(C, "int", "m_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 m_axis_map = unhash_axis_map(m_layout);
const lowp int m_packed_dim = unhash_packed_dim(m_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Maximum block sizes to prevent array overflow
#define MAX_BR 64
#define MAX_BC 128

// Texture access helper functions using proper axis mapping
// Q_sizes, K_sizes, V_sizes, O_sizes are [D, H, N, B] (UBO layout)
// l_sizes, m_sizes are [B, H, N] (UBO layout)
T load_tensor_Q(int batch, int seq_pos, int head, int dim) {
    ivec4 tidx = ivec4(dim, head, seq_pos, batch);  // Match [D, H, N, B] order
    ivec3 pos = tidx_to_pos(tidx, Q_sizes, Q_axis_map, Q_packed_dim);
    int component = tidx[Q_packed_dim] % 4;
    vec4 texel = texelFetch(t_Q, pos, 0);
    return T(texel[component]);
}

T load_tensor_K(int batch, int seq_pos, int head, int dim) {
    ivec4 tidx = ivec4(dim, head, seq_pos, batch);  // Match [D, H, N, B] order
    ivec3 pos = tidx_to_pos(tidx, K_sizes, K_axis_map, K_packed_dim);
    int component = tidx[K_packed_dim] % 4;
    vec4 texel = texelFetch(t_K, pos, 0);
    return T(texel[component]);
}

T load_tensor_V(int batch, int seq_pos, int head, int dim) {
    ivec4 tidx = ivec4(dim, head, seq_pos, batch);  // Match [D, H, N, B] order
    ivec3 pos = tidx_to_pos(tidx, V_sizes, V_axis_map, V_packed_dim);
    int component = tidx[V_packed_dim] % 4;
    vec4 texel = texelFetch(t_V, pos, 0);
    return T(texel[component]);
}

T load_tensor_O(int batch, int seq_pos, int head, int dim) {
    ivec4 tidx = ivec4(dim, head, seq_pos, batch);  // Match [D, H, N, B] order
    ivec3 pos = tidx_to_pos(tidx, O_sizes, O_axis_map, O_packed_dim);
    int component = tidx[O_packed_dim] % 4;
    vec4 texel = imageLoad(t_O, pos);
    return T(texel[component]);
}

void store_tensor_O(int batch, int seq_pos, int head, int dim, T value) {
    ivec4 tidx = ivec4(dim, head, seq_pos, batch);  // Match [D, H, N, B] order
    ivec3 pos = tidx_to_pos(tidx, O_sizes, O_axis_map, O_packed_dim);
    int component = tidx[O_packed_dim] % 4;
    vec4 texel = imageLoad(t_O, pos);
    texel[component] = float(value);
    imageStore(t_O, pos, texel);
}

float load_tensor_l(int batch, int head, int seq_pos) {
    ivec4 tidx = ivec4(seq_pos, head, batch, 0);  // Match [N, H, B] order (with padding)
    ivec3 pos = tidx_to_pos(tidx, ivec4(l_sizes, 1), l_axis_map, l_packed_dim);
    int component = tidx[l_packed_dim] % 4;
    vec4 texel = imageLoad(t_l, pos);
    return texel[component];
}

void store_tensor_l(int batch, int head, int seq_pos, float value) {
    ivec4 tidx = ivec4(seq_pos, head, batch, 0);  // Match [N, H, B] order (with padding)
    ivec3 pos = tidx_to_pos(tidx, ivec4(l_sizes, 1), l_axis_map, l_packed_dim);
    int component = tidx[l_packed_dim] % 4;
    vec4 texel = imageLoad(t_l, pos);
    texel[component] = value;
    imageStore(t_l, pos, texel);
}

float load_tensor_m(int batch, int head, int seq_pos) {
    ivec4 tidx = ivec4(seq_pos, head, batch, 0);  // Match [N, H, B] order (with padding)
    ivec3 pos = tidx_to_pos(tidx, ivec4(m_sizes, 1), m_axis_map, m_packed_dim);
    int component = tidx[m_packed_dim] % 4;
    vec4 texel = imageLoad(t_m, pos);
    return texel[component];
}

void store_tensor_m(int batch, int head, int seq_pos, float value) {
    ivec4 tidx = ivec4(seq_pos, head, batch, 0);  // Match [N, H, B] order (with padding)
    ivec3 pos = tidx_to_pos(tidx, ivec4(m_sizes, 1), m_axis_map, m_packed_dim);
    int component = tidx[m_packed_dim] % 4;
    vec4 texel = imageLoad(t_m, pos);
    texel[component] = value;
    imageStore(t_m, pos, texel);

}

void main() {
    // Each thread processes one row block - same as buffer version
    const int thread_id = int(gl_GlobalInvocationID.x);

    // Tensor dimensions: Q_sizes = [D, H, N, B]
    const int head_dim = Q_sizes.x;     // D (head dim)
    const int num_heads_val = Q_sizes.y;    // H (num heads)
    const int seq_len = Q_sizes.z;      // N (sequence length)
    const int batch_size = Q_sizes.w;   // B (batch)

    // Block sizes
    const int Br = block_size_r;
    const int Bc = block_size_c;

    const int Tr = (seq_len + Br - 1) / Br;  // Number of row blocks
    const int total_row_blocks = batch_size * num_heads_val * Tr;

    if (thread_id >= total_row_blocks) {
        return;
    }

    // Decode thread_id to (batch, head, row_block)
    const int batch = thread_id / (num_heads_val * Tr);
    const int remaining = thread_id % (num_heads_val * Tr);
    const int head = remaining / Tr;
    const int row_block = remaining % Tr;

    // Calculate row range for this block
    const int row_start = row_block * Br;
    const int row_end = min(row_start + Br, seq_len);
    const int actual_Br = row_end - row_start;

    // STEP 1: Initialize only this thread's row block
    // Each thread initializes its own rows to avoid cross-workgroup synchronization issues
    for (int r = 0; r < actual_Br; r++) {
        const int seq_pos = row_start + r;

        // Initialize l and m textures for this row block's positions
        ivec4 l_tidx = ivec4(batch, head, seq_pos, 0);
        ivec3 l_pos = tidx_to_pos(l_tidx, ivec4(l_sizes, 1), l_axis_map, l_packed_dim);
        vec4 l_texel = vec4(0.0);
        imageStore(t_l, l_pos, l_texel);

        ivec4 m_tidx = ivec4(batch, head, seq_pos, 0);
        ivec3 m_pos = tidx_to_pos(m_tidx, ivec4(m_sizes, 1), m_axis_map, m_packed_dim);
        vec4 m_texel = vec4(-1e10);
        imageStore(t_m, m_pos, m_texel);

        // Initialize output tensor for this row block
        for (int dim = 0; dim < head_dim; dim++) {
            store_tensor_O(batch, seq_pos, head, dim, T(0.0));
        }
    }

    // STEP 5: Outer loop over column blocks (For K, V tensors)
    const int Tc = (seq_len + Bc - 1) / Bc;  // Number of column blocks
    for (int j = 0; j < Tc; j++) {
        const int col_start = j * Bc;
        const int col_end = min(col_start + Bc, seq_len);
        const int actual_Bc = col_end - col_start;

        // Load current statistics for all rows in this block
        float m_i[MAX_BR];
        float l_i[MAX_BR];
        for (int r = 0; r < actual_Br; r++) {
            const int seq_pos = row_start + r;
            m_i[r] = load_tensor_m(batch, head, seq_pos);
            l_i[r] = load_tensor_l(batch, head, seq_pos);
        }

        // STEP 9: Compute Sij = Qi * Kj^T
        T S_block[MAX_BR][MAX_BC];
        float m_tilde_ij[MAX_BR];   // Row maxes
        float l_tilde_ij[MAX_BR];   // Row sums

        // Initialize row statistics
        for (int r = 0; r < actual_Br; r++) {
            m_tilde_ij[r] = -1.0 / 0.0; // -infinity
            l_tilde_ij[r] = 0.0;
        }

        // Compute attention scores Sij = Qi @ Kj^T
        for (int r = 0; r < actual_Br; r++) {
            const int global_row = row_start + r;
            for (int c = 0; c < actual_Bc; c++) {
                const int global_col = col_start + c;

                // For multi-query attention: map query head to KV head
                const int kv_head = (head * num_kv_heads) / num_heads_val;

                // Dot product: Q[seq_pos, :] · K[col_pos, :]
                T score = T(0.0);
                for (int dim = 0; dim < head_dim; dim++) {
                    T q_val = load_tensor_Q(batch, global_row, head, dim);
                    T k_val = load_tensor_K(batch, global_col, kv_head, dim);
                    score += q_val * k_val;
                }
                score *= scale;


                // Apply causal masking: mask if global_col > global_row + input_pos
                bool masked = (global_col > global_row + input_pos);
                if (masked) {
                    score = T(-1.0 / 0.0); // Set to negative infinity
                }

                S_block[r][c] = score;


                // Track row maximum (after masking)
                m_tilde_ij[r] = max(m_tilde_ij[r], float(score));
            }
        }

        // STEP 10: Compute P'ij = exp(Sij − m'ij) and l'ij = rowsum(P'ij)
        for (int r = 0; r < actual_Br; r++) {
            // Handle the case where all scores are -inf (fully masked row)
            if (isinf(m_tilde_ij[r]) && m_tilde_ij[r] < 0.0) {
                // All scores are -inf, so all probabilities are 0
                for (int c = 0; c < actual_Bc; c++) {
                    S_block[r][c] = 0.0;
                }
                l_tilde_ij[r] = 0.0;
            } else {
                // Normal case: compute softmax
                for (int c = 0; c < actual_Bc; c++) {
                    S_block[r][c] = exp(S_block[r][c] - T(m_tilde_ij[r]));
                    l_tilde_ij[r] += float(S_block[r][c]);
                }
            }
        }

        // STEP 11: Softmax update
        float m_new_i[MAX_BR];
        float l_new_i[MAX_BR];
        for (int r = 0; r < actual_Br; r++) {
            m_new_i[r] = max(m_i[r], m_tilde_ij[r]);
            l_new_i[r] = exp(m_i[r] - m_new_i[r]) * l_i[r] + exp(m_tilde_ij[r] - m_new_i[r]) * l_tilde_ij[r];

        }

        // STEP 12: Update Oi
        for (int r = 0; r < actual_Br; r++) {
            const int global_row = row_start + r;
            float alpha = exp(m_i[r] - m_new_i[r]);
            float beta = exp(m_tilde_ij[r] - m_new_i[r]);

            // For multi-query attention: map query head to KV head
            const int kv_head = (head * num_kv_heads) / num_heads_val;

            for (int dim = 0; dim < head_dim; dim++) {
                // Compute P'ij @ Vj for this dimension
                T pv_sum = T(0.0);
                for (int c = 0; c < actual_Bc; c++) {
                    const int global_col = col_start + c;
                    T v_val = load_tensor_V(batch, global_col, kv_head, dim);
                    pv_sum += S_block[r][c] * v_val;
                }

                // Check for division by zero before updating output
                if (l_new_i[r] <= 0.0) {
                    store_tensor_O(batch, global_row, head, dim, T(0.0));
                } else {
                    // Oi = (alpha * l_i * Oi + beta * P'ij @ Vj) / l_new_i
                    T current_o = load_tensor_O(batch, global_row, head, dim);
                    T new_o = (T(alpha) * T(l_i[r]) * current_o + T(beta) * pv_sum) / T(l_new_i[r]);
                    store_tensor_O(batch, global_row, head, dim, new_o);

                }
            }
        }

        // STEP 13: Update li, mi
        for (int r = 0; r < actual_Br; r++) {
            const int seq_pos = row_start + r;
            store_tensor_l(batch, head, seq_pos, l_new_i[r]);
            store_tensor_m(batch, head, seq_pos, m_new_i[r]);
        }

    }
}
