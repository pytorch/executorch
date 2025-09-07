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

// Flash Attention inputs: Query, Key, Value tensors
${layout_declare_tensor(B, "rw", "t_O", DTYPE, "buffer")}
${layout_declare_tensor(B, "rw", "t_l", "float", "buffer")}
${layout_declare_tensor(B, "rw", "t_m", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_Q", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_K", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_V", DTYPE, "buffer")}

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
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Maximum block sizes to prevent array overflow
#define MAX_BR 64
#define MAX_BC 128

void main() {
    // Each thread processes one row block
    const int thread_id = int(gl_GlobalInvocationID.x);

    // Tensor dimensions: Q_sizes = [D, H, N, B] from graph.sizes_ubo()
    // The UBO layout is different from the PyTorch tensor layout
    const int head_dim = Q_sizes.x;     // D (head dim)
    const int num_heads = Q_sizes.y;    // H (num heads)
    const int seq_len = Q_sizes.z;      // N (sequence length)
    const int batch_size = Q_sizes.w;   // B (batch)

    // Block sizes
    const int Br = block_size_r;
    const int Bc = block_size_c;

    const int Tr = (seq_len + Br - 1) / Br;  // Number of row blocks
    const int total_row_blocks = batch_size * num_heads * Tr;

    if (thread_id >= total_row_blocks) {
        return;
    }

    // Decode thread_id to (batch, head, row_block)
    const int batch = thread_id / (num_heads * Tr);
    const int remaining = thread_id % (num_heads * Tr);
    const int head = remaining / Tr;
    const int row_block = remaining % Tr;

    // Calculate row range for this block
    const int row_start = row_block * Br;
    const int row_end = min(row_start + Br, seq_len);
    const int actual_Br = row_end - row_start;

    // Base indices for this batch
    const int q_base = batch * (seq_len * num_heads * head_dim);
    const int k_base = batch * (seq_len * num_heads * head_dim);
    const int v_base = batch * (seq_len * num_heads * head_dim);
    const int o_base = batch * (seq_len * num_heads * head_dim);
    const int lm_base = batch * (seq_len * num_heads);

    // STEP 2: Initialize O = 0, l = 0, m = -inf for this row block
    for (int r = 0; r < actual_Br; r++) {
        const int seq_pos = row_start + r;
        const int lm_idx = lm_base + head * seq_len + seq_pos;

        t_l[lm_idx] = 0.0;
        t_m[lm_idx] = -1.0 / 0.0; // -infinity

        for (int dim = 0; dim < head_dim; dim++) {
            const int o_idx = o_base + seq_pos * (num_heads * head_dim) + head * head_dim + dim;
            t_O[o_idx] = T(0.0);
        }
    }

    // STEP 5: Outer loop over column blocks (For K, V tensors)
    const int Tc = (seq_len + Bc - 1) / Bc;  // Number of column blocks
    for (int j = 0; j < Tc; j++) {
        const int col_start = j * Bc;
        const int col_end = min(col_start + Bc, seq_len);
        const int actual_Bc = col_end - col_start;

        // STEP 6-8 done implicitly below

        // Load current statistics for all rows in this block
        float m_i[MAX_BR];
        float l_i[MAX_BR];
        for (int r = 0; r < actual_Br; r++) {
            const int seq_pos = row_start + r;
            const int lm_idx = lm_base + head * seq_len + seq_pos;
            m_i[r] = t_m[lm_idx];
            l_i[r] = t_l[lm_idx];
        }

        // STEP 9: Compute Sij = Qi * Kj^T
        T S_block[MAX_BR][MAX_BC]; // Use MAX_BR and MAX_BC constants
        float m_tilde_ij[MAX_BR];   // Row maxes (float to match l/m)
        float l_tilde_ij[MAX_BR];   // Row sums (float to match l/m)

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
                const int kv_head = (head * num_kv_heads) / num_heads;

                // Dot product: Q[seq_pos, :] · K[col_pos, :]
                T score = T(0.0);
                for (int dim = 0; dim < head_dim; dim++) {
                    const int q_idx = q_base + global_row * (num_heads * head_dim) + head * head_dim + dim;
                    const int k_idx = k_base + global_col * (num_kv_heads * head_dim) + kv_head * head_dim + dim;
                    score += t_Q[q_idx] * t_K[k_idx];
                }
                score *= scale;


                // Apply causal masking: mask if global_col > global_row + input_pos
                if (global_col > global_row + input_pos) {
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
                    S_block[r][c] = T(0.0);
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
            const int kv_head = (head * num_kv_heads) / num_heads;

            for (int dim = 0; dim < head_dim; dim++) {
                const int o_idx = o_base + global_row * (num_heads * head_dim) + head * head_dim + dim;

                // Compute P'ij @ Vj for this dimension
                T pv_sum = T(0.0);
                for (int c = 0; c < actual_Bc; c++) {
                    const int global_col = col_start + c;
                    const int v_idx = v_base + global_col * (num_kv_heads * head_dim) + kv_head * head_dim + dim;
                    pv_sum += S_block[r][c] * t_V[v_idx];
                }

                // Check for division by zero before updating output
                if (l_new_i[r] <= 0.0) {
                    t_O[o_idx] = T(0.0); // Set to zero to avoid NaN
                } else {
                    // Oi = (alpha * l_i * Oi + beta * P'ij @ Vj) / l_new_i
                    t_O[o_idx] = (T(alpha) * T(l_i[r]) * t_O[o_idx] + T(beta) * pv_sum) / T(l_new_i[r]);
                }
            }
        }

        // STEP 13: Update li, mi
        for (int r = 0; r < actual_Br; r++) {
            const int seq_pos = row_start + r;
            const int lm_idx = lm_base + head * seq_len + seq_pos;
            t_l[lm_idx] = l_new_i[r];
            t_m[lm_idx] = m_new_i[r];
        }
    }
}
